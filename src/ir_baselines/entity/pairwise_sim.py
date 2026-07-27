"""
Pairwise entity similarity: rank documents by how similar their entities are
to the query's.

The query's entities come from its entity links, not from an entity ranking,
so unlike ExactMatch this baseline does not depend on the entity-ranking
stage.

Each query entity and each document entity is looked up in a Wikipedia2Vec
embedding table, and a similarity matrix is formed between the two sets:

    --metric cos   cosine similarity (default)
    --metric dot   inner product

    --method max   take the maximum over document entities for each query
                   entity, then sum those maxima (default in the papers'
                   MaxSimCos row)
    --method sum   sum the whole matrix

The confidence score attached to each query entity by the entity linker is
read but not used in the score.

Documents scoring zero or less are not written to the run.
"""

import argparse
import gzip
import json
import operator
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def calculate_document_score(
        query_id: str,
        doc_id: str,
        query_entities: Dict[str, float],
        doc_entities: List[str],
        embeddings: Dict[str, List[float]],
        metric: str,
        method: str
) -> float:
    query_vectors = None
    doc_vectors = None
    try:
        query_vectors = np.array(
            [embeddings[e] for e in query_entities.keys() if e in embeddings])
        doc_vectors = np.array(
            [embeddings[str(e)] for e in doc_entities if str(e) in embeddings])

        if query_vectors.size == 0 or doc_vectors.size == 0:
            return 0.0

        if metric == 'dot':
            similarities = np.dot(query_vectors, doc_vectors.T)
        else:
            similarities = cosine_similarity(query_vectors, doc_vectors)

        if method == 'max':
            return float(np.sum(np.max(similarities, axis=1)))
        return float(np.sum(similarities))
    except Exception as e:
        # The shapes are reported only if the arrays were built. Referring to
        # them unconditionally raised NameError inside the handler when the
        # failure happened during construction.
        shapes = ''
        if query_vectors is not None and doc_vectors is not None:
            shapes = ' query={} doc={}'.format(query_vectors.shape, doc_vectors.shape)
        print('Exception on {}/{}: {}{}'.format(query_id, doc_id, e, shapes))
        return 0.0


def get_query_annotations(query_annotations: str) -> Dict[str, float]:
    annotations = json.loads(query_annotations)
    return {str(ann['entity_id']): ann['score'] for ann in annotations}


def rank_docs_for_query(
        query_id: str,
        candidate_docs: List[str],
        docs: Dict[str, List[str]],
        query_annotations: Dict[str, float],
        embeddings: Dict[str, List[float]],
        metric: str,
        method: str
) -> Dict[str, float]:
    ranking: Dict[str, float] = {
        doc_id: calculate_document_score(
            query_id=query_id,
            doc_id=doc_id,
            query_entities=query_annotations,
            doc_entities=docs[doc_id],
            embeddings=embeddings,
            metric=metric,
            method=method,
        )
        for doc_id in candidate_docs if doc_id in docs
    }
    return dict(sorted(ranking.items(), key=operator.itemgetter(1), reverse=True))


def to_run_file_strings(query: str, doc_ranking: Dict[str, float],
                        metric: str, method: str) -> List[str]:
    run_file_strings: List[str] = []
    tag = metric + '_' + method
    rank: int = 1
    for doc_id, score in doc_ranking.items():
        if score > 0.0:
            run_file_strings.append(
                query + ' Q0 ' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' ' + tag)
            rank += 1
    return run_file_strings


def re_rank(
        run: Dict[str, List[str]],
        docs: Dict[str, List[str]],
        query_annotations: Dict[str, str],
        embeddings: Dict[str, List[float]],
        out_file: str,
        metric: str,
        method: str
) -> None:
    # Opened once, in write mode. Earlier versions appended per query, so
    # running twice without deleting the output doubled the run file.
    n_written = n_scored = 0
    no_annotations = []
    empty = []

    with open(out_file, 'w') as f:
        for query_id, candidate_docs in tqdm(run.items(), total=len(run), desc='ranking'):
            if query_id not in query_annotations:
                no_annotations.append(query_id)
                continue
            ranked_docs = rank_docs_for_query(
                query_id=query_id,
                candidate_docs=candidate_docs,
                docs=docs,
                query_annotations=get_query_annotations(query_annotations[query_id]),
                embeddings=embeddings,
                metric=metric,
                method=method,
            )
            n_scored += len(ranked_docs)
            lines = to_run_file_strings(query_id, ranked_docs, metric, method)
            if not lines:
                empty.append(query_id)
                continue
            for line in lines:
                f.write('%s\n' % line)
            n_written += len(lines)

    print()
    print('Topics in the run : {}'.format(len(run)))
    print('Candidates scored : {:,}'.format(n_scored))
    print('Pairs written     : {:,}'.format(n_written))
    if n_scored:
        print('Dropped, score <=0: {:,} ({:.1f}%)'.format(
            n_scored - n_written, 100 * (n_scored - n_written) / n_scored))
    if no_annotations:
        print('WARNING: {} topic(s) have no entity annotations and are absent '
              'from this run: {}'.format(
                  len(no_annotations), ', '.join(no_annotations[:10])
                  + (' ...' if len(no_annotations) > 10 else '')))
    if empty:
        print('NOTE: {} topic(s) produced no positively scored document: {}'.format(
            len(empty), ', '.join(empty[:10]) + (' ...' if len(empty) > 10 else '')))


def load_embeddings(embedding_file: str) -> Dict[str, List[float]]:
    emb = {}
    with gzip.open(embedding_file, 'rt') as f:
        for line in tqdm(f, desc='embeddings'):
            d = json.loads(line)
            emb[d['entity_id']] = d['embedding'][:300]
    return emb


def read_docs(file: str) -> Dict[str, List[str]]:
    docs: Dict[str, List[str]] = {}
    with open(file, 'r') as f:
        for line in tqdm(f, desc='corpus'):
            d = json.loads(line)
            docs[d['doc_id']] = d['entities']
    return docs


def read_run(run_file: str) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            run.setdefault(parts[0], []).append(parts[2])
    return run


def read_tsv_to_dict(file_path: str, key_index: int = 0, value_index: int = 1) -> Dict[str, str]:
    """Read a TSV file into a dictionary mapping one column to another."""
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for n, line in enumerate(tqdm(f, desc=file_path.split('/')[-1]), start=1):
            parts = line.strip().split('\t')
            try:
                result[parts[key_index].strip()] = parts[value_index].strip()
            except IndexError:
                print('Skipping malformed line {} of {}'.format(n, file_path))
    return result


def main():
    parser = argparse.ArgumentParser('Rank documents by pairwise entity similarity.')
    parser.add_argument('--run', help='Document run file to re-rank.', required=True)
    parser.add_argument('--docs', help='Entity-linked corpus (JSONL with doc_id and entities).',
                        required=True)
    parser.add_argument('--annotations', help='TSV of query entity annotations.', required=True)
    parser.add_argument('--embeddings', help='Wikipedia2Vec entity embeddings (gzipped JSONL).',
                        required=True)
    parser.add_argument('--metric', help='Similarity metric (cos|dot). Default: cos',
                        default='cos', choices=('cos', 'dot'), type=str)
    parser.add_argument('--method', help='Score method (sum|max). Default: sum',
                        default='sum', choices=('sum', 'max'), type=str)
    parser.add_argument('--save', help='Output run file.', required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading run file...')
    run = read_run(run_file=args.run)
    print('[Done]. {} topics.'.format(len(run)))

    print('Loading corpus...')
    docs = read_docs(file=args.docs)
    print('[Done]. {:,} documents.'.format(len(docs)))

    print('Loading query annotations...')
    query_annotations = read_tsv_to_dict(file_path=args.annotations)
    print('[Done]. {} topics.'.format(len(query_annotations)))

    print('Loading entity embeddings...')
    embeddings = load_embeddings(embedding_file=args.embeddings)
    print('[Done]. {:,} entities.'.format(len(embeddings)))

    print('Re-ranking with metric={} method={}...'.format(args.metric, args.method))
    re_rank(run=run, docs=docs, query_annotations=query_annotations,
            embeddings=embeddings, out_file=args.save,
            metric=args.metric, method=args.method)
    print('[Done].')
    print('Run file written to ==> {}'.format(args.save))


if __name__ == '__main__':
    main()