"""
Build a candidate run with Pyserini, and optionally the corpus subset that
goes with it.

    python -m ir_baselines.retrieve \\
      --index msmarco-v1-passage \\
      --queries queries.tsv \\
      --method bm25+rm3 \\
      --out bm25_rm3.run --save-corpus corpus.jsonl

Why the corpus subset matters. `build_data` needs the text of every candidate,
and a collection corpus is usually far larger than the candidate set -- CODEC's
is 4.3 GB of JSON for about forty thousand retrieved documents. Worse, a corpus
assembled separately from the run can be missing documents the run refers to,
and those candidates are then silently dropped. Writing both from the same
index makes that impossible: every document in the run is in the corpus,
because both came out of the same search.

    retrieve  ->  run + corpus
    build_data ->  train/dev/test JSONL
    train / test

METHODS

    bm25            BM25 alone.
    bm25+rm3        BM25 with RM3 pseudo-relevance feedback. The usual
                    candidate ranking for the re-rankers in this package.
    bm25+rocchio    BM25 with Rocchio feedback, an alternative to RM3.
    bm25+rerank     BM25, then re-ranked by a trained checkpoint from this
                    package. Two stages in one command; --rerank-depth
                    controls how many candidates the re-ranker sees.

Pyserini needs a Java runtime (17 or later for recent versions). Neither it nor
Java is a dependency of this package, since most of it does not need them:

    pip install pyserini
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

METHODS = ('bm25', 'bm25+rm3', 'bm25+rocchio', 'bm25+rerank')


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def load_queries(dataset: Optional[str], queries_file: Optional[str],
                 query_field: Optional[str]) -> Dict[str, str]:
    """{query_id: text}, from ir_datasets or a TSV."""
    if queries_file:
        queries = {}
        with open(queries_file) as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    raise SystemExit(
                        f'{queries_file}:{n}: expected "query_id<TAB>text"')
                queries[parts[0]] = parts[1]
        return queries

    try:
        import ir_datasets
    except ImportError:
        raise SystemExit(
            '--dataset needs ir_datasets:  pip install "ir_baselines[data]"'
        ) from None

    # The same field resolution build_data uses, so a run and the data built
    # from it cannot disagree about which query set they are.
    from .build_data import query_text
    ds = ir_datasets.load(dataset)
    return {q.query_id: query_text(q, query_field)
            for q in tqdm(ds.queries_iter(), desc='queries')}


# ---------------------------------------------------------------------------
# searcher
# ---------------------------------------------------------------------------

def build_searcher(args):
    """
    A configured LuceneSearcher.

    `--index` is either a prebuilt index name that Pyserini downloads, or a
    path to one built locally with `python -m pyserini.index.lucene`.
    """
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        raise SystemExit(
            'This command needs Pyserini and a Java runtime:\n'
            '    pip install pyserini\n'
            'Pyserini requires Java 17 or later for recent versions.'
        ) from None

    if os.path.isdir(args.index):
        print(f'Opening local index {args.index}')
        searcher = LuceneSearcher(args.index)
    else:
        print(f'Loading prebuilt index {args.index}')
        searcher = LuceneSearcher.from_prebuilt_index(args.index)
        if searcher is None:
            raise SystemExit(
                f'no prebuilt index named {args.index!r}, and it is not a '
                f'directory. Run\n'
                f'    python -c "from pyserini.search.lucene import LuceneSearcher; '
                f'LuceneSearcher.list_prebuilt_indexes()"\n'
                f'to see what is available.')

    searcher.set_bm25(args.k1, args.b)
    print(f'BM25 k1={args.k1} b={args.b}')

    if args.method == 'bm25+rm3':
        searcher.set_rm3(fb_terms=args.fb_terms, fb_docs=args.fb_docs,
                         original_query_weight=args.original_query_weight)
        print(f'RM3 fb_terms={args.fb_terms} fb_docs={args.fb_docs} '
              f'original_query_weight={args.original_query_weight}')
    elif args.method == 'bm25+rocchio':
        searcher.set_rocchio(top_fb_terms=args.fb_terms, top_fb_docs=args.fb_docs,
                             alpha=args.rocchio_alpha, beta=args.rocchio_beta)
        print(f'Rocchio fb_terms={args.fb_terms} fb_docs={args.fb_docs} '
              f'alpha={args.rocchio_alpha} beta={args.rocchio_beta}')

    if args.language:
        searcher.set_language(args.language)
    return searcher


def search(searcher, queries: Dict[str, str], k: int, threads: int, batch: int
           ) -> Dict[str, List[Tuple[str, float]]]:
    """
    {query_id: [(doc_id, score), ...]} in rank order.

    Batched, because per-query search pays the JVM round trip every time.
    """
    qids = list(queries)
    results: Dict[str, List[Tuple[str, float]]] = {}

    for start in tqdm(range(0, len(qids), batch), desc='searching'):
        chunk = qids[start:start + batch]
        hits = searcher.batch_search([queries[q] for q in chunk], chunk,
                                     k=k, threads=threads)
        for qid in chunk:
            results[qid] = [(h.docid, float(h.score)) for h in hits.get(qid, [])]

    empty = [q for q, h in results.items() if not h]
    if empty:
        print(f'WARNING  {len(empty)} query/queries returned nothing '
              f'(e.g. {empty[:5]}). They will be absent from the run, and '
              f'trec_eval -c counts an absent topic as unretrieved.')
    return results


# ---------------------------------------------------------------------------
# document text
# ---------------------------------------------------------------------------

def raw_text(searcher, doc_id: str) -> Optional[str]:
    """
    The stored text of one document.

    A Lucene document's raw field is usually the JSON that was indexed, so the
    text is under `contents` or `text`. Where it is not JSON, the raw string is
    the text.
    """
    doc = searcher.doc(doc_id)
    if doc is None:
        return None
    raw = doc.raw()
    if raw is None:
        return doc.contents()
    raw = raw.strip()
    if raw.startswith('{'):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        for key in ('contents', 'text', 'body', 'abstract'):
            if key in d and d[key]:
                title = d.get('title', '')
                return f'{title} {d[key]}'.strip() if title else str(d[key])
        return raw
    return raw


def write_corpus(searcher, results: Dict[str, List[Tuple[str, float]]],
                 path: str) -> int:
    """
    {doc_id, text} for every document in the run, and nothing else.

    This is what makes the run and the corpus consistent by construction: both
    come from the same index in the same command, so `build_data` cannot find a
    candidate whose text is missing.
    """
    needed = {doc_id for hits in results.values() for doc_id, _ in hits}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    missing = 0
    with open(path, 'w') as f:
        for doc_id in tqdm(sorted(needed), desc='corpus'):
            text = raw_text(searcher, doc_id)
            if text is None:
                missing += 1
                continue
            f.write(json.dumps({'doc_id': doc_id, 'text': text}) + '\n')
    if missing:
        print(f'WARNING  {missing} document(s) retrieved but not readable from '
              f'the index. That should not happen and is worth investigating.')
    return len(needed) - missing


# ---------------------------------------------------------------------------
# optional re-ranking stage
# ---------------------------------------------------------------------------

def rerank(args, results: Dict[str, List[Tuple[str, float]]],
           queries: Dict[str, str], searcher
           ) -> Dict[str, List[Tuple[str, float]]]:
    """
    Re-rank the top `--rerank-depth` candidates with a trained checkpoint.

    Only the top slice is re-ranked; the remainder keeps its first-stage order
    and is appended below. That is the usual arrangement, and it means the run
    still covers the full candidate set -- a re-ranked run truncated to its
    top-k would be charged under `trec_eval -c` for everything it dropped.
    """
    import torch

    from . import build, evaluate, utils
    from .data import RankingDataLoader, RankingDataset

    device = torch.device(f'cuda:{args.cuda}'
                          if torch.cuda.is_available() and args.use_cuda else 'cpu')
    tokenizer, pretrained = build.build_tokenizer(args)
    model = build.build(args, tokenizer, pretrained)
    stored = utils.load_checkpoint(args.rerank, model, device, strict=True)
    conflicts = utils.check_config(stored.get('config', {}),
                                   build.architecture_config(args, model, pretrained),
                                   build.ARCHITECTURE_KEYS)
    if conflicts:
        for key, was, now in conflicts:
            print(f'ERROR  {key} was {was!r} at training time, {now!r} now.')
        raise SystemExit(
            'Re-ranking settings differ from the checkpoint\'s training '
            'settings. The weights do not mean what this architecture expects.')
    model.to(device)

    # One pass over the index for the text, rather than one lookup per pair.
    head = {qid: hits[:args.rerank_depth] for qid, hits in results.items()}
    needed = {doc_id for hits in head.values() for doc_id, _ in hits}
    print(f'Fetching text for {len(needed):,} candidates')
    texts = {}
    for doc_id in tqdm(sorted(needed), desc='text'):
        t = raw_text(searcher, doc_id)
        if t is not None:
            texts[doc_id] = ' '.join(t.split())

    import tempfile
    fd, tmp = tempfile.mkstemp(suffix='.jsonl')
    n = 0
    with os.fdopen(fd, 'w') as f:
        for qid, hits in head.items():
            for doc_id, _ in hits:
                if doc_id in texts:
                    f.write(json.dumps({
                        'query_id': qid, 'doc_id': doc_id,
                        'query': ' '.join(queries[qid].split()),
                        'doc': texts[doc_id], 'label': 0}) + '\n')
                    n += 1
    print(f'Scoring {n:,} pairs')

    ds = build.make_dataset(args, tmp, tokenizer, train=False, model=model)
    loader = RankingDataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
    scored = evaluate.evaluate(model, loader, device, amp_dtype=None)
    os.unlink(tmp)

    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, hits in results.items():
        by_doc = scored.get(qid, {})
        top = sorted(((d, by_doc[d][0]) for d, _ in hits[:args.rerank_depth]
                      if d in by_doc), key=lambda x: (-x[1], x[0]))
        # The tail keeps first-stage order, below everything re-ranked.
        floor = min((s for _, s in top), default=0.0)
        tail = [(d, floor - 1.0 - i)
                for i, (d, _) in enumerate(hits[args.rerank_depth:])]
        out[qid] = top + tail
    return out


# ---------------------------------------------------------------------------

def write_run(path: str, results: Dict[str, List[Tuple[str, float]]],
              tag: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    n = 0
    with open(path, 'w') as f:
        for qid in sorted(results):
            # Ties broken by document id, so the same scores always produce
            # the same ranking.
            ranked = sorted(results[qid], key=lambda x: (-x[1], x[0]))
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f'{qid} Q0 {doc_id} {rank} {score!r} {tag}\n')
                n += 1
    return n


def main():
    p = argparse.ArgumentParser(
        'Build a candidate run with Pyserini.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    p.add_argument('--index', required=True,
                   help='A Pyserini prebuilt index name, or a path to a local one.')
    p.add_argument('--out', required=True, help='Output run file.')
    p.add_argument('--save-corpus', default=None,
                   help='Also write {doc_id, text} for every retrieved document. '
                        'This is what build_data needs, and writing it here means '
                        'no candidate can be missing from it.')

    p.add_argument('--dataset', default=None, help='An ir_datasets id, for queries.')
    p.add_argument('--queries', default=None, help='TSV of query_id<TAB>text.')
    p.add_argument('--query-field', default=None,
                   help='Which query field to use, when the dataset has several.')

    p.add_argument('--method', choices=METHODS, default='bm25+rm3',
                   help=f'Retrieval method: {" | ".join(METHODS)}. Default: bm25+rm3')
    p.add_argument('--k', type=int, default=1000,
                   help='Candidates per query. Default: 1000')
    p.add_argument('--run-tag', default=None,
                   help='Run tag written in field 6. Default: the method.')

    p.add_argument('--k1', type=float, default=0.9, help='BM25 k1. Default: 0.9')
    p.add_argument('--b', type=float, default=0.4, help='BM25 b. Default: 0.4')
    p.add_argument('--fb-terms', type=int, default=10,
                   help='Feedback terms for RM3 or Rocchio. Default: 10')
    p.add_argument('--fb-docs', type=int, default=10,
                   help='Feedback documents. Default: 10')
    p.add_argument('--original-query-weight', type=float, default=0.5,
                   help='RM3 weight on the original query. Default: 0.5')
    p.add_argument('--rocchio-alpha', type=float, default=1.0, help='Default: 1.0')
    p.add_argument('--rocchio-beta', type=float, default=0.75, help='Default: 0.75')
    p.add_argument('--language', default=None,
                   help='Analyzer language for a non-English index.')

    p.add_argument('--rerank', default=None,
                   help='Checkpoint to re-rank with, for --method bm25+rerank.')
    p.add_argument('--rerank-depth', type=int, default=100,
                   help='How many first-stage candidates the re-ranker sees. The '
                        'rest keep their first-stage order below. Default: 100')

    p.add_argument('--threads', type=int, default=8, help='Search threads. Default: 8')
    p.add_argument('--batch', type=int, default=64,
                   help='Queries per batch_search call. Default: 64')
    args, _ = p.parse_known_args()

    # The re-ranking stage needs the model arguments; they are only added when
    # asked for, so `--help` stays readable for the common case.
    if args.method == 'bm25+rerank':
        from .build import add_common_args
        add_common_args(p)
        args = p.parse_args()
        if not args.rerank:
            p.error('--method bm25+rerank needs --rerank <checkpoint>')
    else:
        args = p.parse_args()

    if not args.dataset and not args.queries:
        p.error('pass --dataset or --queries')
    if args.k < 1:
        p.error('--k must be >= 1')

    queries = load_queries(args.dataset, args.queries, args.query_field)
    print(f'{len(queries):,} queries')

    searcher = build_searcher(args)
    results = search(searcher, queries, args.k, args.threads, args.batch)

    if args.method == 'bm25+rerank':
        results = rerank(args, results, queries, searcher)

    tag = args.run_tag or args.method.replace('+', '_')
    n = write_run(args.out, results, tag)
    topics = len([q for q, h in results.items() if h])
    print(f'\n{n:,} lines over {topics} topics -> {args.out}')
    print(f'mean depth {n / max(topics, 1):.1f}')

    if args.save_corpus:
        kept = write_corpus(searcher, results, args.save_corpus)
        print(f'{kept:,} documents -> {args.save_corpus}')
        print('\nEvery document in the run is in this corpus, so build_data will '
              'not\nfind a candidate whose text is missing.')


if __name__ == '__main__':
    main()