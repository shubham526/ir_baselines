"""
Build a Lucene index, and retrieve from it.

Two subcommands, in the order you need them:

    python -m ir_baselines.retrieve index \\
      --docs corpus.jsonl --index /path/to/lucene

    python -m ir_baselines.retrieve search \\
      --index /path/to/lucene --queries queries.tsv \\
      --method bm25+rm3 --out bm25_rm3.run --save-corpus corpus.subset.jsonl

and then the rest of the pipeline:

    python -m ir_baselines.build_data --run bm25_rm3.run --docs corpus.subset.jsonl ...
    python -m ir_baselines.train ...
    python -m ir_baselines.test ...

WHY --save-corpus MATTERS

`build_data` needs the text of every candidate. A corpus assembled separately
from the run can be missing documents the run refers to, and those candidates
are then dropped without comment. Writing both from the same index in the same
command makes that impossible. It also avoids reading a whole collection to use
a small part of it: a collection corpus runs to several gigabytes, and the
candidate set is usually a few tens of thousands of documents.

RETRIEVAL METHODS

    bm25            BM25 alone.
    bm25+rm3        BM25 with RM3 pseudo-relevance feedback. The usual
                    candidate ranking for the re-rankers in this package.
    bm25+rocchio    BM25 with Rocchio feedback, an alternative to RM3.
    bm25+rerank     BM25, then re-ranked by a trained checkpoint from this
                    package. --rerank-depth controls how many candidates the
                    re-ranker sees; the rest keep their first-stage order.

REQUIREMENTS

Pyserini needs a Java runtime -- 21 for recent versions, 11 for older ones.
Neither it nor Java is a dependency of this package, since nothing else here
needs them:

    pip install "ir_baselines[retrieval]"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

METHODS = ('bm25', 'bm25+rm3', 'bm25+rocchio', 'bm25+rerank')


def _require_pyserini():
    try:
        import pyserini  # noqa: F401
    except ImportError:
        raise SystemExit(
            'This command needs Pyserini and a Java runtime:\n'
            '    pip install "ir_baselines[retrieval]"\n'
            'Pyserini needs Java 21 for recent versions, 11 for older ones.\n'
            '    java -version'
        ) from None


# ===========================================================================
# index
# ===========================================================================

def _write_pyserini_jsonl(out_dir: str, docs, total: Optional[int] = None) -> int:
    """
    Pyserini's JsonCollection expects {"id", "contents"} per line, so a corpus
    keyed on doc_id/text has to be rewritten. Done as a stream, since the
    input is usually far larger than memory.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'docs.jsonl')
    n = 0
    with open(path, 'w') as out:
        for doc_id, text in tqdm(docs, total=total, desc='converting'):
            if not text:
                continue
            out.write(json.dumps({'id': doc_id, 'contents': text}) + '\n')
            n += 1
    return n


def _docs_from_jsonl(path: str, field: Optional[str]):
    from .build_data import DOC_BODY_FIELDS, _flatten, read_jsonl
    for d in read_jsonl(path, desc='corpus'):
        if field:
            text = _flatten(d.get(field, ''))
        else:
            text = ''
            for key in DOC_BODY_FIELDS:
                if key in d:
                    text = _flatten(d[key])
                    break
            title = _flatten(d.get('title', ''))
            if title and text:
                text = f'{title} {text}'
        yield d['doc_id'], text


def _docs_from_dataset(dataset: str, field: Optional[str]):
    try:
        import ir_datasets
    except ImportError:
        raise SystemExit(
            '--dataset needs ir_datasets:  pip install "ir_baselines[data]"'
        ) from None
    from .build_data import doc_text
    ds = ir_datasets.load(dataset)
    total = ds.docs_count() if hasattr(ds, 'docs_count') else None
    return ((d.doc_id, doc_text(d, field)) for d in ds.docs_iter()), total


def cmd_index(args) -> None:
    _require_pyserini()

    if args.docs:
        source, total = _docs_from_jsonl(args.docs, args.doc_field), None
    else:
        source, total = _docs_from_dataset(args.dataset, args.doc_field)

    workdir = args.staging or tempfile.mkdtemp(prefix='ir_baselines_index_')
    print(f'Staging Pyserini-format JSONL in {workdir}')
    n = _write_pyserini_jsonl(workdir, source, total)
    print(f'{n:,} documents staged')
    if n == 0:
        raise SystemExit('nothing to index: every document had empty text')

    os.makedirs(args.index, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', workdir,
        '--index', args.index,
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', str(args.threads),
        '--storePositions', '--storeDocvectors',
        # storeRaw is what lets `search --save-corpus` read document text back
        # out of the index. Without it the corpus subset cannot be written.
        '--storeRaw',
    ]
    if args.language:
        cmd += ['--language', args.language]
    if args.extra:
        cmd += args.extra.split()

    print('\n' + ' '.join(cmd) + '\n')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f'indexing failed (exit {result.returncode}). The staged JSONL is '
            f'in {workdir} if you want to retry by hand.')

    if args.staging is None and not args.keep_staging:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f'Removed staging directory. Pass --keep-staging to retain it.')

    print(f'\nIndex written to {args.index}')
    print(f'Retrieve from it with:\n'
          f'    python -m ir_baselines.retrieve search --index {args.index} \\\n'
          f'      --queries queries.tsv --method bm25+rm3 \\\n'
          f'      --out bm25_rm3.run --save-corpus corpus.subset.jsonl')


# ===========================================================================
# search
# ===========================================================================

def load_queries(dataset: Optional[str], queries_file: Optional[str],
                 query_field: Optional[str]) -> Dict[str, str]:
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


def build_searcher(args):
    _require_pyserini()
    from pyserini.search.lucene import LuceneSearcher

    if os.path.isdir(args.index):
        print(f'Opening local index {args.index}')
        searcher = LuceneSearcher(args.index)
    else:
        print(f'Loading prebuilt index {args.index}')
        searcher = LuceneSearcher.from_prebuilt_index(args.index)
        if searcher is None:
            raise SystemExit(
                f'no prebuilt index named {args.index!r}, and it is not a '
                f'directory. Build one with\n'
                f'    python -m ir_baselines.retrieve index --docs corpus.jsonl '
                f'--index {args.index}')

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
    """{query_id: [(doc_id, score), ...]} in rank order."""
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


def raw_text(searcher, doc_id: str) -> Optional[str]:
    """
    The stored text of one document. Needs the index to have been built with
    --storeRaw, which `index` above does.
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
            if d.get(key):
                title = d.get('title', '')
                return f'{title} {d[key]}'.strip() if title else str(d[key])
        return raw
    return raw


def write_corpus(searcher, results, path: str) -> int:
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
        print(f'WARNING  {missing} document(s) were retrieved but could not be '
              f'read back. If the index was built without --storeRaw, that is '
              f'the reason.')
    return len(needed) - missing


def rerank(args, results, queries, searcher):
    """
    Re-rank the top --rerank-depth with a trained checkpoint.

    Only the head is reordered; the tail keeps first-stage order below it, so
    the run still covers the full candidate set. A run truncated to its
    re-ranked head would be charged under trec_eval -c for everything dropped.
    """
    import torch

    from . import build, evaluate, utils
    from .data import RankingDataLoader

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
            "Re-ranking settings differ from the checkpoint's training settings. "
            'The weights do not mean what this architecture expects.')
    model.to(device)

    head = {qid: hits[:args.rerank_depth] for qid, hits in results.items()}
    needed = {doc_id for hits in head.values() for doc_id, _ in hits}
    print(f'Fetching text for {len(needed):,} candidates')
    texts = {}
    for doc_id in tqdm(sorted(needed), desc='text'):
        t = raw_text(searcher, doc_id)
        if t is not None:
            texts[doc_id] = ' '.join(t.split())

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

    out = {}
    for qid, hits in results.items():
        by_doc = scored.get(qid, {})
        top = sorted(((d, by_doc[d][0]) for d, _ in hits[:args.rerank_depth]
                      if d in by_doc), key=lambda x: (-x[1], x[0]))
        floor = min((s for _, s in top), default=0.0)
        tail = [(d, floor - 1.0 - i)
                for i, (d, _) in enumerate(hits[args.rerank_depth:])]
        out[qid] = top + tail
    return out


def write_run(path: str, results, tag: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    n = 0
    with open(path, 'w') as f:
        for qid in sorted(results):
            # Ties broken by document id, so the same scores always produce the
            # same ranking. Matches utils.save_trec.
            ranked = sorted(results[qid], key=lambda x: (-x[1], x[0]))
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f'{qid} Q0 {doc_id} {rank} {score!r} {tag}\n')
                n += 1
    return n


def cmd_search(args) -> None:
    if not args.dataset and not args.queries:
        raise SystemExit('pass --dataset or --queries')
    if args.method == 'bm25+rerank' and not args.rerank:
        raise SystemExit('--method bm25+rerank needs --rerank <checkpoint>')
    if args.method == 'bm25+rerank' and not args.model:
        raise SystemExit('--method bm25+rerank needs --model, matching the '
                         'checkpoint. See: python -m ir_baselines.train --list-models')

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
        print('\nEvery document in the run is in this corpus, so build_data '
              'cannot\nfind a candidate whose text is missing.')


# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        'Build a Lucene index and retrieve from it.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    # ---------------------------------------------------------------- index
    i = sub.add_parser('index', help='Build a Lucene index from a corpus.')
    i.add_argument('--index', required=True, help='Directory to write the index to.')
    i.add_argument('--docs', default=None,
                   help='Corpus JSONL with doc_id and text.')
    i.add_argument('--dataset', default=None,
                   help='An ir_datasets id to index instead.')
    i.add_argument('--doc-field', default=None,
                   help='Which document field to index. Default: the first '
                        'text field the corpus has, with the title prepended.')
    i.add_argument('--threads', type=int, default=8, help='Default: 8')
    i.add_argument('--language', default=None,
                   help='Analyzer language for a non-English collection.')
    i.add_argument('--staging', default=None,
                   help='Where to write the converted JSONL. Default: a '
                        'temporary directory, removed afterwards.')
    i.add_argument('--keep-staging', action='store_true',
                   help='Keep the converted JSONL after indexing.')
    i.add_argument('--extra', default=None,
                   help='Further arguments passed to pyserini.index.lucene, '
                        'as one quoted string.')

    # --------------------------------------------------------------- search
    s = sub.add_parser('search', help='Retrieve a candidate run from an index.')
    s.add_argument('--index', required=True,
                   help='Index directory, or a Pyserini prebuilt index name.')
    s.add_argument('--out', required=True, help='Output run file.')
    s.add_argument('--save-corpus', default=None,
                   help='Also write {doc_id, text} for every retrieved document, '
                        'which is what build_data needs.')

    s.add_argument('--dataset', default=None, help='An ir_datasets id, for queries.')
    s.add_argument('--queries', default=None, help='TSV of query_id<TAB>text.')
    s.add_argument('--query-field', default=None,
                   help='Which query field to use, when the dataset has several.')

    s.add_argument('--method', choices=METHODS, default='bm25+rm3',
                   help=f'{" | ".join(METHODS)}. Default: bm25+rm3')
    s.add_argument('--k', type=int, default=1000,
                   help='Candidates per query. Default: 1000')
    s.add_argument('--run-tag', default=None,
                   help='Run tag in field 6. Default: the method.')

    s.add_argument('--k1', type=float, default=0.9, help='BM25 k1. Default: 0.9')
    s.add_argument('--b', type=float, default=0.4, help='BM25 b. Default: 0.4')
    s.add_argument('--fb-terms', type=int, default=10, help='Default: 10')
    s.add_argument('--fb-docs', type=int, default=10, help='Default: 10')
    s.add_argument('--original-query-weight', type=float, default=0.5,
                   help='RM3 weight on the original query. Default: 0.5')
    s.add_argument('--rocchio-alpha', type=float, default=1.0, help='Default: 1.0')
    s.add_argument('--rocchio-beta', type=float, default=0.75, help='Default: 0.75')
    s.add_argument('--language', default=None, help='Analyzer language.')

    s.add_argument('--threads', type=int, default=8, help='Default: 8')
    s.add_argument('--batch', type=int, default=64,
                   help='Queries per batch_search call. Default: 64')

    # Re-ranking. --model is not required here, unlike train and test, because
    # three of the four methods do not use a model at all.
    r = s.add_argument_group('re-ranking (--method bm25+rerank)')
    r.add_argument('--rerank', default=None, help='Checkpoint to re-rank with.')
    r.add_argument('--rerank-depth', type=int, default=100,
                   help='Candidates the re-ranker sees. The rest keep their '
                        'first-stage order below. Default: 100')
    r.add_argument('--model', default=None,
                   help='Model name, matching the checkpoint.')
    r.add_argument('--pretrain', default=None, help='Encoder, overriding the default.')
    r.add_argument('--t5-pooling', default='mean-all',
                   choices=('mean-all', 'masked-mean'))
    r.add_argument('--poly-m', type=int, default=16)
    r.add_argument('--me-bert-m', type=int, default=8)
    r.add_argument('--me-bert-proj-dim', type=int, default=None)
    r.add_argument('--shared-encoder', default=None,
                   type=lambda v: v.strip().lower() in ('1', 'true', 'yes', 'y', 't'))
    r.add_argument('--logit-scale', action='store_true')
    r.add_argument('--no-validate-inputs', action='store_true')
    r.add_argument('--max-len', type=int, default=512)
    r.add_argument('--max-query-len', type=int, default=20)
    r.add_argument('--max-doc-len', type=int, default=512)
    r.add_argument('--batch-size', type=int, default=8)
    r.add_argument('--num-workers', type=int, default=0)
    r.add_argument('--use-cuda', action='store_true')
    r.add_argument('--cuda', type=int, default=0)

    args = p.parse_args()

    if args.command == 'index':
        if not args.docs and not args.dataset:
            p.error('index needs --docs or --dataset')
        if args.docs and args.dataset:
            p.error('pass --docs or --dataset, not both')
        cmd_index(args)
    else:
        cmd_search(args)


if __name__ == '__main__':
    main()