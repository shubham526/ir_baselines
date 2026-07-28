"""
Build re-ranking training and evaluation data from a candidate run.

    python -m ir_baselines.build_data \\
      --dataset disks45/nocr/trec-robust-2004 \\
      --run title.BM25_RM3_TUNED.run \\
      --folds folds.json \\
      --out data/robust04/title

writes, for each fold,

    data/robust04/title/fold-0/train.jsonl
    data/robust04/title/fold-0/dev.jsonl
    data/robust04/title/fold-0/dev.qrels
    data/robust04/title/fold-0/test.jsonl
    data/robust04/title/fold-0/test.qrels
    ...

The per-split qrels matter. Validation scores the dev run against dev.qrels,
and passing the full collection qrels instead means every topic in the other
folds counts as unretrieved -- which training refuses, since the checkpoint
would then be selected on a number that is not what it appears to be.

in the format the models consume:

    train  {"query_id", "query", "doc_id", "doc", "label"}
    dev    the same
    test   the same

`query_id` and `doc_id` are written everywhere. The models ignore them during
training except under `--loss ce-inbatch`, which needs `query_id` to avoid
scoring a second positive for the same query as a negative.

WHAT GOES IN EACH SPLIT

Training data is sampled: every judged positive in the candidate list, plus
`--negatives-per-positive` negatives drawn from it. Evaluation data is not
sampled -- test.jsonl carries the whole candidate list, because a run that
covers fewer documents than the systems it is compared against is not
comparable to them, and because `trec_eval -c` charges a run for the relevant
documents it does not rank.

Documents in the candidate list with no relevance judgment are treated as
negatives for training, following the usual convention, and are kept for
evaluation.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


def _by_bytes(path: str, desc: str):
    """
    A progress bar measured in bytes rather than lines.

    A corpus is several gigabytes and a line count gives no sense of how far
    through it is; the file size is known up front, so the bar can show a real
    percentage and time remaining.
    """
    total = os.path.getsize(path)
    bar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=desc)
    with open(path) as f:
        for line in f:
            bar.update(len(line))
            yield line
    bar.close()

NEGATIVE_SAMPLING = ('top', 'random')


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    {query_id: [(doc_id, score), ...]} in the order the run gives, which is
    rank order. That order is what `--negative-sampling top` depends on.
    """
    run: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    if True:
        for n, line in enumerate(_by_bytes(path, 'run'), start=1):
            parts = line.split()
            if len(parts) < 5:
                if line.strip():
                    raise ValueError(
                        f'{path}:{n}: expected six whitespace-separated fields, '
                        f'got {len(parts)}. TREC run format is '
                        f'"qid Q0 docid rank score tag".')
                continue
            qid, did, score = parts[0], parts[2], float(parts[4])
            if did in seen[qid]:
                continue                      # a duplicate ranking; keep the first
            seen[qid].add(did)
            run[qid].append((did, score))
    if not run:
        raise ValueError(f'{path}: no runs read')
    return dict(run)


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    if True:
        for n, line in enumerate(_by_bytes(path, 'qrels'), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(
                    f'{path}:{n}: expected four fields, got {len(parts)}')
            qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def read_folds(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path) as f:
        folds = json.load(f)
    for k, v in folds.items():
        missing = {'training', 'testing'} - set(v)
        if missing:
            raise ValueError(
                f'{path}: fold {k} is missing {sorted(missing)}. Expected '
                f'{{"training": [...], "testing": [...]}} per fold.')
    return folds


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

#: Query fields that hold text, in the order they are preferred when a dataset
#: exposes exactly one of them. A dataset exposing more than one is ambiguous
#: and --query-field is required, because on Robust04 that choice is the
#: difference between the title and description query sets -- two different
#: published rows.
QUERY_TEXT_FIELDS = ('text', 'title', 'description', 'query')

#: Document body fields, likewise. BEIR-style datasets expose `text`; native
#: TREC exposes `body`; CORD-19 and nfcorpus expose `abstract`.
DOC_BODY_FIELDS = ('text', 'body', 'abstract', 'contents')


def _flatten(value) -> str:
    """
    A field may be a string or a sequence of section objects. Formatting a
    sequence with f-string interpolation yields its repr, which looks like
    text and is not, so sequences are joined explicitly.
    """
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return ' '.join(_flatten(getattr(v, 'text', v)) for v in value)
    return str(value)


def available_text_fields(cls, candidates) -> List[str]:
    return [f for f in candidates if f in getattr(cls, '_fields', ())]


def query_text(q, field: Optional[str] = None) -> str:
    """
    The text of one query.

    With no `field`, a dataset exposing exactly one text field is used
    directly and one exposing several is refused. That refusal is the point:
    ir_datasets' own `default_text()` returns the topic title on Robust04, so
    a caller who wanted description queries and did not say so would get title
    queries and a number that looks plausible.
    """
    if field is not None:
        if not hasattr(q, field):
            raise SystemExit(
                f'queries have no field {field!r}. Available: '
                f'{", ".join(q._fields)}')
        return _flatten(getattr(q, field))

    present = available_text_fields(type(q), QUERY_TEXT_FIELDS)
    if len(present) == 1:
        return _flatten(getattr(q, present[0]))
    if not present:
        raise SystemExit(
            f'no recognised query text field on {type(q).__name__}; fields are '
            f'{", ".join(q._fields)}. Pass --query-field.')
    raise SystemExit(
        f'{type(q).__name__} has several query text fields ({", ".join(present)}), '
        f'so which to use is not obvious. Pass --query-field. On Robust04 this '
        f'is the difference between the title and description query sets.')


def doc_text(d, field: Optional[str] = None, with_title: bool = True) -> str:
    """
    The text of one document.

    `with_title` prepends the title where there is one, which is what
    ir_datasets' `default_text()` does for the datasets that have both and is
    what the published runs used. Turn it off to index the body alone.
    """
    if field is not None:
        if not hasattr(d, field):
            raise SystemExit(
                f'documents have no field {field!r}. Available: '
                f'{", ".join(d._fields)}')
        return _flatten(getattr(d, field))

    present = available_text_fields(type(d), DOC_BODY_FIELDS)
    if not present:
        raise SystemExit(
            f'no recognised document text field on {type(d).__name__}; fields '
            f'are {", ".join(d._fields)}. Pass --doc-field.')
    body = _flatten(getattr(d, present[0]))

    title = _flatten(getattr(d, 'title', '')) if with_title else ''
    # Only when the title is a separate field from the body: BEIR's `text` is
    # the body and its `title` is separate, but a dataset whose only text field
    # IS the title must not have it twice.
    if title and present[0] != 'title':
        return f'{title} {body}'.strip()
    return body.strip()


def read_jsonl(path: str, desc: Optional[str] = None):
    """Parsed objects from a JSONL file, blank lines skipped."""
    source = _by_bytes(path, desc) if desc else open(path)
    for n, line in enumerate(source, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f'{path}:{n}: {e}') from None


class TextSource:
    """
    Query and document text, from ir_datasets or from local files.

    Field names differ between datasets -- BEIR exposes `text`, native TREC
    exposes `title`/`body`, CORD-19 and nfcorpus expose `abstract` -- so the
    field is resolved per dataset rather than assumed. Where the choice is
    ambiguous, it is refused rather than guessed.
    """

    def __init__(self, dataset: Optional[str], queries_file: Optional[str],
                 docs_file: Optional[str], query_field: Optional[str],
                 doc_field: Optional[str], doc_title: bool = True,
                 needed_docs: Optional[set] = None):
        self.queries: Dict[str, str] = {}
        self._docs_local: Optional[Dict[str, str]] = None
        self._docs_store = None
        self._doc_field = doc_field
        self._doc_title = doc_title

        if dataset:
            try:
                import ir_datasets
            except ImportError:
                raise SystemExit(
                    '--dataset needs ir_datasets. Install it with\n'
                    '    pip install "ir-baselines[data]"\n'
                    'or supply --queries and --docs instead.') from None
            ds = ir_datasets.load(dataset)
            for q in tqdm(ds.queries_iter(), desc='queries'):
                self.queries[q.query_id] = query_text(q, query_field)
            self._docs_store = ds.docs_store()
            print(f'{len(self.queries):,} queries from {dataset}')

        if queries_file:
            for parts in (line.rstrip('\n').split('\t')
                          for line in _by_bytes(queries_file, 'queries')
                          if line.strip()):
                if len(parts) < 2:
                    raise SystemExit(
                        f'{queries_file}: expected "query_id<TAB>text"')
                self.queries[parts[0]] = parts[1]
            print(f'{len(self.queries):,} queries from {queries_file}')

        if docs_file:
            # Only the documents the run asks for. A collection corpus runs to
            # several gigabytes of JSON, and holding all of it costs far more
            # in memory than the candidates actually need -- CODEC's is 4.3 GB
            # on disk for a candidate set of about forty thousand documents.
            self._docs_local = {}
            seen = skipped = 0
            for d in read_jsonl(docs_file, desc='corpus'):
                seen += 1
                doc_id = d['doc_id']
                if needed_docs is not None and doc_id not in needed_docs:
                    skipped += 1
                    continue
                self._docs_local[doc_id] = self._local_doc_text(d)
            print(f'{len(self._docs_local):,} documents kept from {docs_file} '
                  f'({seen:,} read, {skipped:,} not in the run)')
            if needed_docs is not None:
                absent = len(needed_docs) - len(self._docs_local)
                if absent:
                    print(f'NOTE  {absent:,} candidate document(s) are not in the '
                          f'corpus and will be skipped.')

        if not self.queries:
            raise SystemExit('no queries: pass --dataset or --queries')
        if self._docs_local is None and self._docs_store is None:
            raise SystemExit('no documents: pass --dataset or --docs')

    def _local_doc_text(self, d: dict) -> str:
        if self._doc_field:
            if self._doc_field not in d:
                raise SystemExit(
                    f'--doc-field {self._doc_field!r} not in the document JSON; '
                    f'keys are {sorted(d)}')
            return _flatten(d[self._doc_field])
        for key in DOC_BODY_FIELDS:
            if key in d:
                body = _flatten(d[key])
                title = _flatten(d.get('title', '')) if self._doc_title else ''
                return f'{title} {body}'.strip() if title else body
        raise SystemExit(
            f'no document text found; expected one of '
            f'{"/".join(DOC_BODY_FIELDS)}, got keys {sorted(d)}. Use --doc-field.')

    def doc(self, doc_id: str) -> Optional[str]:
        if self._docs_local is not None:
            return self._docs_local.get(doc_id)
        try:
            d = self._docs_store.get(doc_id)
        except KeyError:
            return None
        return None if d is None else doc_text(d, self._doc_field, self._doc_title)


# ---------------------------------------------------------------------------
# example construction
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Collapse whitespace. A newline in a JSONL field is not an error, but a
    tab or newline surviving into the text makes the file harder to inspect."""
    return ' '.join(text.split())


def select(candidates: List[Tuple[str, float]],
           qrels: Dict[str, int],
           negatives_per_positive: int,
           sampling: str,
           rng: random.Random) -> List[Tuple[str, int]]:
    """
    -> [(doc_id, label)] for one query.

    `negatives_per_positive <= 0` keeps every candidate, which is what
    evaluation data uses.
    """
    positives = [(d, 1) for d, _ in candidates if qrels.get(d, 0) >= 1]
    negatives = [d for d, _ in candidates if qrels.get(d, 0) < 1]

    if negatives_per_positive <= 0:
        return [(d, 1 if qrels.get(d, 0) >= 1 else 0) for d, _ in candidates]

    if not positives:
        return []                     # nothing to learn from this query

    want = len(positives) * negatives_per_positive
    if sampling == 'random':
        chosen = rng.sample(negatives, min(want, len(negatives)))
    else:
        # A prefix of the run, so the negatives are the highest-ranked ones:
        # documents that looked relevant to the first stage and are not. Those
        # are the informative ones, and it is a stronger choice than the
        # 1:1 ratio itself.
        chosen = negatives[:want]
    return positives + [(d, 0) for d in chosen]


def build_split(topics: Iterable[str],
                run: Dict[str, List[Tuple[str, float]]],
                qrels: Dict[str, Dict[str, int]],
                text: TextSource,
                negatives_per_positive: int,
                sampling: str,
                rng: random.Random,
                stats: dict, desc: str = 'building') -> List[str]:
    lines = []
    topics = list(topics)
    for qid in tqdm(topics, desc=desc, leave=False):
        if qid not in text.queries:
            stats['topics_without_query_text'].add(qid)
            continue
        if qid not in run:
            stats['topics_absent_from_run'].add(qid)
            continue
        if qid not in qrels:
            stats['topics_without_judgments'].add(qid)
            continue

        query = normalise(text.queries[qid])
        chosen = select(run[qid], qrels[qid], negatives_per_positive, sampling, rng)
        if not chosen:
            stats['topics_with_no_positive'].add(qid)
            continue

        for doc_id, label in chosen:
            body = text.doc(doc_id)
            if body is None:
                stats['documents_without_text'] += 1
                continue
            lines.append(json.dumps({
                'query_id': qid,
                'query': query,
                'doc_id': doc_id,
                'doc': normalise(body),
                'label': label,
            }))
            stats['positives' if label else 'negatives'] += 1
        stats['topics_written'].add(qid)
    return lines


def write(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def write_qrels(path: str, qrels: Dict[str, Dict[str, int]],
                topics: Iterable[str]) -> int:
    """
    The judgments for one split's topics, and only those.

    Validation needs this. Scoring a dev run against the full qrels means the
    topics in the other folds count as unretrieved, so training refuses to
    proceed -- correctly, since under `trec_eval -c` those topics would drag
    the metric down and the checkpoint would be selected on a number that is
    not what it appears to be.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, 'w') as f:
        for qid in topics:
            for doc_id, rel in sorted(qrels.get(qid, {}).items()):
                f.write(f'{qid} 0 {doc_id} {rel}\n')
                n += 1
    return n


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        'Build re-ranking data from a candidate run.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('--run', required=True,
                   help='Candidate ranking in TREC format. This is what gets '
                        're-ranked, and the order matters: --negative-sampling '
                        'top takes a prefix of it.')
    p.add_argument('--out', required=True, help='Output directory.')

    p.add_argument('--dataset', default=None,
                   help='An ir_datasets id, e.g. disks45/nocr/trec-robust-2004. '
                        'Supplies queries, documents and qrels.')
    p.add_argument('--queries', default=None,
                   help='TSV of query_id<TAB>text, overriding --dataset.')
    p.add_argument('--docs', default=None,
                   help='JSONL with doc_id and text, overriding --dataset.')
    p.add_argument('--qrels', default=None,
                   help='TREC qrels, overriding --dataset.')

    p.add_argument('--query-field', default=None,
                   help='Which query field to use. Required when the dataset '
                        'exposes more than one -- on Robust04, title or '
                        'description, which are different published query sets.')
    p.add_argument('--doc-field', default=None,
                   help='Which document field to use. Default: the first of '
                        + '/'.join(DOC_BODY_FIELDS) + ' the dataset has.')
    p.add_argument('--no-doc-title', action='store_true',
                   help='Do not prepend the document title to its body. The '
                        'title is included by default, which is what the '
                        'published runs used.')

    p.add_argument('--folds', default=None,
                   help='folds.json with {"0": {"training": [...], '
                        '"testing": [...]}, ...}. Without it, one split is '
                        'written covering every topic in the run.')
    p.add_argument('--dev-fraction', type=float, default=0.1,
                   help='Fraction of each fold\'s TRAINING topics held out for '
                        'validation. Taken from training, never from testing: a '
                        'dev set drawn from the test fold selects checkpoints on '
                        'the data the fold is scored on. Default: 0.1')

    p.add_argument('--negatives-per-positive', type=int, default=1,
                   help='Negatives sampled per judged positive, for training '
                        'and validation data. 0 or less keeps every candidate. '
                        'Default: 1, which is the 1:1 balance the published '
                        'experiments used -- a convention, not a validated '
                        'choice.')
    p.add_argument('--negative-sampling', choices=NEGATIVE_SAMPLING, default='top',
                   help='top takes the highest-ranked negatives, which are the '
                        'hard ones; random samples from the whole candidate '
                        'list. Default: top.')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for --negative-sampling random and for the dev '
                        'split. Default: 42')
    args = p.parse_args()

    if not args.dataset and not (args.queries and args.docs):
        p.error('pass --dataset, or both --queries and --docs')
    if not args.dataset and not args.qrels:
        p.error('pass --dataset or --qrels')
    if not 0.0 <= args.dev_fraction < 1.0:
        p.error('--dev-fraction must be in [0, 1)')

    rng = random.Random(args.seed)

    print('Reading run...')
    run = read_run(args.run)
    print(f'  {len(run):,} topics')

    print('Reading qrels...')
    if args.qrels:
        qrels = read_qrels(args.qrels)
    else:
        import ir_datasets
        ds = ir_datasets.load(args.dataset)
        if not ds.has_qrels():
            raise SystemExit(f'{args.dataset} has no qrels; pass --qrels')
        qrels = defaultdict(dict)
        for q in tqdm(ds.qrels_iter(), desc='qrels'):
            qrels[q.query_id][q.doc_id] = q.relevance
        qrels = dict(qrels)
    print(f'  {len(qrels):,} topics')

    # Everything the run refers to, so the corpus can be filtered as it is
    # read rather than held whole.
    needed = {doc_id for candidates in run.values() for doc_id, _ in candidates}
    print(f'  {len(needed):,} distinct documents referenced')

    text = TextSource(args.dataset, args.queries, args.docs,
                      args.query_field, args.doc_field,
                      doc_title=not args.no_doc_title,
                      needed_docs=needed)

    def fresh_stats():
        return {'positives': 0, 'negatives': 0, 'documents_without_text': 0,
                'topics_written': set(), 'topics_absent_from_run': set(),
                'topics_without_query_text': set(), 'topics_without_judgments': set(),
                'topics_with_no_positive': set()}

    def report(name, path, lines, stats):
        print(f'  {name:<6} {len(lines):>9,} examples  '
              f'{stats["positives"]:>8,} positive  '
              f'{stats["negatives"]:>9,} negative  '
              f'{len(stats["topics_written"]):>4} topics  -> {path}')
        for key, label in (
                ('topics_absent_from_run', 'absent from the run'),
                ('topics_without_query_text', 'no query text'),
                ('topics_without_judgments', 'no relevance judgments'),
                ('topics_with_no_positive', 'no judged positive in the candidates')):
            missing = stats[key]
            if missing:
                shown = ', '.join(sorted(missing)[:5])
                more = ' ...' if len(missing) > 5 else ''
                print(f'         {len(missing)} topic(s) {label}: {shown}{more}')
        if stats['documents_without_text']:
            print(f'         {stats["documents_without_text"]:,} candidate(s) had no '
                  f'document text and were skipped')

    if args.folds:
        folds = read_folds(args.folds)
        print(f'\nBuilding {len(folds)} folds into {args.out}')
        for k in sorted(folds, key=lambda x: int(x) if str(x).isdigit() else x):
            train_topics = list(folds[k]['training'])
            test_topics = list(folds[k]['testing'])

            # Held out from training, never from testing.
            rng.shuffle(train_topics)
            n_dev = int(round(len(train_topics) * args.dev_fraction))
            if args.dev_fraction > 0 and n_dev == 0 and len(train_topics) > 1:
                n_dev = 1
            dev_topics, train_topics = train_topics[:n_dev], train_topics[n_dev:]

            print(f'\nfold-{k}  {len(train_topics)} train / {len(dev_topics)} dev '
                  f'/ {len(test_topics)} test topics')
            for name, topics, npp in (
                    ('train', train_topics, args.negatives_per_positive),
                    ('dev', dev_topics, args.negatives_per_positive),
                    ('test', test_topics, 0)):
                stats = fresh_stats()
                lines = build_split(topics, run, qrels, text, npp,
                                    args.negative_sampling, rng, stats,
                                    desc=f'fold-{k} {name}')
                path = os.path.join(args.out, f'fold-{k}', f'{name}.jsonl')
                write(path, lines)
                report(name, path, lines, stats)

                # Restricted to the topics actually written, not to the topics
                # requested: a topic dropped for want of judgments or text is
                # absent from the data and must be absent from the qrels too,
                # or validation refuses to score.
                if name != 'train':
                    qpath = os.path.join(args.out, f'fold-{k}', f'{name}.qrels')
                    n = write_qrels(qpath, qrels, sorted(stats['topics_written']))
                    print(f'         {n:,} judgments for these topics -> {qpath}')
    else:
        print(f'\nNo --folds given; writing one split over every topic in the run.')
        stats = fresh_stats()
        lines = build_split(sorted(run), run, qrels, text,
                            args.negatives_per_positive,
                            args.negative_sampling, rng, stats, desc='building')
        path = os.path.join(args.out, 'data.jsonl')
        write(path, lines)
        report('all', path, lines, stats)
        qpath = os.path.join(args.out, 'data.qrels')
        n = write_qrels(qpath, qrels, sorted(stats['topics_written']))
        print(f'         {n:,} judgments for these topics -> {qpath}')

    print('\nEvaluation data carries the whole candidate list, unsampled. A run')
    print('built from a sampled test set covers fewer documents than the systems')
    print('it is compared against, and trec_eval -c charges it for the relevant')
    print('ones it does not rank.')


if __name__ == '__main__':
    main()