"""
ExactMatch: rank documents by the query entities they contain.

For each query, take the top-k entities of its entity ranking. A document
scores the sum, over those entities that appear in it, of

    (1 / rank of the entity) * (score of the entity)

Documents scoring zero -- those sharing no entity with the top-k -- are not
written to the run. That is the method rather than a fault, and it is why the
run is much shallower than the candidate set it was given: on Robust04 title
queries it ranks about 17 documents per topic against the candidate set's
1000. Under `trec_eval -c` the run is charged for the relevant documents it
does not rank.

Note that the query entities come from the entity ranking, so this baseline
depends on the entity-ranking stage in the same way the systems it is compared
against do.
"""

import argparse
import collections
import json
import operator
import sys
from typing import Dict, List

from tqdm import tqdm


def read_run(run_file: str) -> Dict[str, Dict[str, float]]:
    run = collections.defaultdict(dict)
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id, object_id, score = parts[0], parts[2], parts[4]
            if object_id not in run[query_id]:
                run[query_id][object_id] = float(score)
    return run


def read_docs(file: str) -> Dict[str, List[str]]:
    docs: Dict[str, List[str]] = {}
    with open(file, 'r') as f:
        for line in tqdm(f, desc='corpus'):
            d = json.loads(line)
            docs[d['doc_id']] = [str(e) for e in d['entities']]
    return docs


def calculate_document_score(entities: Dict[str, float], doc_entities: List[str]) -> float:
    doc_entity_set = set(doc_entities)
    score = 0.0
    # rank comes from the position in the (already ordered) top-k mapping.
    for rank, (entity_id, entity_score) in enumerate(entities.items(), start=1):
        if entity_id in doc_entity_set:
            score += 1 / rank * entity_score
    return score


def rank_docs_for_query(
        candidate_docs: Dict[str, float],
        docs: Dict[str, List[str]],
        entities: Dict[str, float]
) -> Dict[str, float]:
    ranking: Dict[str, float] = {
        doc_id: calculate_document_score(entities=entities, doc_entities=docs[doc_id])
        for doc_id in candidate_docs if doc_id in docs
    }
    return dict(sorted(ranking.items(), key=operator.itemgetter(1), reverse=True))


def to_run_file_strings(query: str, doc_ranking: Dict[str, float]) -> List[str]:
    run_file_strings: List[str] = []
    tag = 'ExactMatch'
    rank: int = 1
    for doc_id, score in doc_ranking.items():
        if score > 0.0:
            run_file_strings.append(
                query + ' Q0 ' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' ' + tag)
            rank += 1
    return run_file_strings


def re_rank(
        doc_run: Dict[str, Dict[str, float]],
        entity_run: Dict[str, Dict[str, float]],
        docs: Dict[str, List[str]],
        out_file: str,
        k: int
) -> None:
    # Opened once, in write mode. Earlier versions appended per query, so
    # running twice without deleting the output doubled the run file.
    n_written = n_scored = 0
    no_entity_run = []
    empty = []

    with open(out_file, 'w') as f:
        for query_id, candidate_docs in tqdm(doc_run.items(), total=len(doc_run), desc='ranking'):
            if query_id not in entity_run:
                no_entity_run.append(query_id)
                continue
            entities = dict(list(entity_run[query_id].items())[:k])
            ranked_docs = rank_docs_for_query(
                candidate_docs=candidate_docs, docs=docs, entities=entities)
            n_scored += len(ranked_docs)
            if not ranked_docs:
                empty.append(query_id)
                continue
            lines = to_run_file_strings(query_id, ranked_docs)
            if not lines:
                empty.append(query_id)
                continue
            for line in lines:
                f.write('%s\n' % line)
            n_written += len(lines)

    print()
    print('Topics in the document run   : {}'.format(len(doc_run)))
    print('Topics with an entity ranking: {}'.format(len(doc_run) - len(no_entity_run)))
    print('Candidates scored            : {:,}'.format(n_scored))
    print('Pairs written                : {:,}'.format(n_written))
    if n_scored:
        print('Dropped for scoring zero     : {:,} ({:.1f}%)'.format(
            n_scored - n_written, 100 * (n_scored - n_written) / n_scored))
    if no_entity_run:
        print('WARNING: {} topic(s) absent from the entity run and therefore '
              'absent from this run: {}'.format(
                  len(no_entity_run), ', '.join(no_entity_run[:10])
                  + (' ...' if len(no_entity_run) > 10 else '')))
    if empty:
        print('NOTE: {} topic(s) had no candidate sharing an entity with the '
              'top-{}: {}'.format(len(empty), k, ', '.join(empty[:10])
                                  + (' ...' if len(empty) > 10 else '')))


def main():
    parser = argparse.ArgumentParser(
        'Rank documents by the top-k query entities they contain.')
    parser.add_argument('--doc-run', help='Document run file to re-rank.', required=True)
    parser.add_argument('--docs', help='Entity-linked corpus (JSONL with doc_id and entities).',
                        required=True)
    parser.add_argument('--entity-run', help='Entity run file.', required=True, type=str)
    parser.add_argument('--save', help='Output run file.', required=True, type=str)
    parser.add_argument('--k', help='Top-K entities to consider. Default: 20', type=int, default=20)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading entity run file...')
    entity_ranking = read_run(run_file=args.entity_run)
    print('[Done]. {} topics.'.format(len(entity_ranking)))

    print('Loading document run file...')
    doc_ranking = read_run(run_file=args.doc_run)
    print('[Done]. {} topics.'.format(len(doc_ranking)))

    print('Loading corpus...')
    docs = read_docs(file=args.docs)
    print('[Done]. {:,} documents.'.format(len(docs)))

    print('Re-ranking...')
    re_rank(doc_run=doc_ranking, entity_run=entity_ranking, docs=docs,
            out_file=args.save, k=args.k)
    print('[Done].')
    print('Run file written to ==> {}'.format(args.save))


if __name__ == '__main__':
    main()