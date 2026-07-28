"""
Evaluation.

Built on ir_measures, which wraps `trec_eval` and -- importantly -- averages
over the topics in the QRELS rather than over the topics present in the run.
That is `trec_eval -c`, and it is the convention every figure in this area is
reported under.

The distinction is not cosmetic. Given a run covering two of four topics, and
covering those two perfectly:

    ir_measures    AP 0.5     two perfect topics over four judged
    pytrec_eval    AP 1.0     two perfect topics over two scored

Under the second, a run that loses its hard topics reports a BETTER number,
not a worse one. Where that number selects a checkpoint, the model that lost
the most topics wins. `require_full_coverage` below exists to guard exactly
that; with `-c` semantics it is a second line of defence rather than the only
one, and it is kept because a missing topic is still worth refusing rather
than discovering later.

Measure names are ir_measures strings -- `nDCG@20`, `AP`, `P@20`, `RR` -- with
the `trec_eval` spellings (`ndcg_cut_20`, `map`, `P_20`, `recip_rank`)
accepted as aliases, since those are what published tables and older scripts
use.
"""

import os
from functools import lru_cache
from typing import Dict, Iterable, Set, Tuple

import ir_measures
from ir_measures import parse_measure

#: trec_eval's internal spellings, which appear throughout published tables
#: and in anything that previously drove this module.
ALIASES = {
    'map': 'AP',
    'ndcg': 'nDCG',
    'ndcg_cut_10': 'nDCG@10',
    'ndcg_cut_20': 'nDCG@20',
    'ndcg_cut_100': 'nDCG@100',
    'P_10': 'P@10',
    'P_20': 'P@20',
    'P_100': 'P@100',
    'recip_rank': 'RR',
    'recall_100': 'R@100',
    'recall_1000': 'R@1000',
}

DEFAULT_MEASURES = ('AP', 'nDCG@20', 'P@20', 'RR')


def _with_judged_only(name: str) -> str:
    """
    Insert `judged_only=True` into an ir_measures name.

    `nDCG@20` becomes `nDCG(judged_only=True)@20`; `AP` becomes
    `AP(judged_only=True)`. The cutoff stays outside the parameter list, which
    is where ir_measures expects it.
    """
    if '(' in name:                      # already parameterised
        head, rest = name.split('(', 1)
        return f'{head}(judged_only=True,{rest}'
    if '@' in name:
        head, cutoff = name.split('@', 1)
        return f'{head}(judged_only=True)@{cutoff}'
    return f'{name}(judged_only=True)'


def resolve(name: str, judged_only: bool = False):
    """
    A measure object from an ir_measures name or a trec_eval spelling.

    `judged_only` is `trec_eval -J`: unjudged documents are removed from the
    ranking rather than counted as non-relevant. Some collections are scored
    that way -- shallow judgment pools make the difference large -- and a
    figure computed one way is not comparable to one computed the other.
    """
    canonical = ALIASES.get(name, name)
    if judged_only:
        canonical = _with_judged_only(canonical)
    try:
        return parse_measure(canonical)
    except (NameError, ValueError) as e:
        # Only the two ir_measures raises for a name it cannot parse. A broader
        # catch would turn any unrelated fault into "unknown measure", which
        # sends the reader looking in the wrong place.
        raise KeyError(
            f'unknown measure {name!r} ({e}). Use an ir_measures name such as '
            f'nDCG@20, AP, P@20 or RR; the trec_eval spellings '
            f'({", ".join(sorted(ALIASES))}) are accepted as aliases.'
        ) from None


def _fingerprint(path: str, measures: Tuple[str, ...], judged_only: bool):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size, measures, judged_only)


@lru_cache(maxsize=8)
def _cached_evaluator(fingerprint, path: str, measures: Tuple[str, ...],
                      judged_only: bool = False):
    """
    An ir_measures Evaluator, built once per (qrels file, measure set).

    Validation runs every epoch against the same qrels, and rebuilding the
    evaluator each time re-reads and re-processes the whole file. The
    fingerprint includes the file's mtime and size, so editing the qrels
    invalidates the cache rather than silently scoring against a stale copy.
    """
    with open(path) as f:
        qrels_list = list(ir_measures.read_trec_qrels(f))
    resolved = [resolve(name, judged_only) for name in measures]
    return ir_measures.evaluator(resolved, qrels_list), qrels_list


def _load(qrels: str, run: str):
    with open(qrels) as f:
        qrels_list = list(ir_measures.read_trec_qrels(f))
    with open(run) as f:
        run_list = list(ir_measures.read_trec_run(f))
    return qrels_list, run_list


def _read_run(run: str):
    with open(run) as f:
        return list(ir_measures.read_trec_run(f))


def _topics(qrels_list, run_list) -> Tuple[Set[str], Set[str]]:
    return ({q.query_id for q in qrels_list}, {r.query_id for r in run_list})


def _calc(qrels_list, run_list, measures: Iterable[str],
          judged_only: bool = False) -> Dict[str, float]:
    """Keys come back as the caller spelled them, aliases included."""
    resolved = {name: resolve(name, judged_only) for name in measures}
    agg = ir_measures.calc_aggregate(list(resolved.values()), qrels_list, run_list)
    return {name: agg[m] for name, m in resolved.items()}


def _calc_cached(qrels: str, run_list, measures: Tuple[str, ...],
                 judged_only: bool = False) -> Dict[str, float]:
    """As _calc, but reusing an Evaluator built once per qrels file."""
    ev, _ = _cached_evaluator(_fingerprint(qrels, measures, judged_only),
                              qrels, measures, judged_only)
    resolved = {name: resolve(name, judged_only) for name in measures}
    agg = ev.calc_aggregate(run_list)
    return {name: agg[m] for name, m in resolved.items()}


def get_all_metrics(qrels: str, run: str,
                    measures: Iterable[str] = DEFAULT_MEASURES,
                    judged_only: bool = False
                    ) -> Tuple[Dict[str, float], Set[str], Set[str], Set[str]]:
    """
    Returns (metrics, topics scored, topics in qrels, topics in run).

    All three topic sets are returned because they differ in ways that matter.
    A topic in the run but not in the qrels contributes nothing and cannot be
    detected from the results alone, so the raw run set is needed to find it.
    A topic in the qrels but not in the run counts as unretrieved, which is
    what `-c` means.
    """
    qrels_list, run_list = _load(qrels, run)
    qrels_topics, run_topics = _topics(qrels_list, run_list)
    return (_calc(qrels_list, run_list, measures, judged_only),
            qrels_topics & run_topics, qrels_topics, run_topics)


def require_full_coverage(qrels: str, run: str, context: str = 'run',
                          measures: Iterable[str] = DEFAULT_MEASURES,
                          judged_only: bool = False) -> Dict[str, float]:
    """
    Scores a run and refuses to return a metric unless its topic set matches
    the qrels exactly.

    Under `-c` an incomplete run reports a depressed figure rather than an
    inflated one, so this is no longer the difference between a plausible
    number and a wrong one. It is kept because a checkpoint should not be
    selected on a run that is quietly missing topics, and because a topic
    present in the run but absent from the qrels -- the right count with the
    wrong ids -- is caught here and nowhere else.
    """
    measures = tuple(measures)
    _, qrels_list = _cached_evaluator(_fingerprint(qrels, measures, judged_only),
                                      qrels, measures, judged_only)
    run_list = _read_run(run)
    qrels_topics, run_topics = _topics(qrels_list, run_list)

    missing = qrels_topics - run_topics
    extra = run_topics - qrels_topics
    if missing or extra:
        parts = []
        if missing:
            parts.append(f'{len(missing)} qrels topics absent from the {context} '
                         f'(e.g. {sorted(missing)[:5]})')
        if extra:
            parts.append(f'{len(extra)} {context} topics absent from the qrels '
                         f'(e.g. {sorted(extra)[:5]})')
        raise RuntimeError(
            'Topic sets do not match: ' + '; '.join(parts) + '. '
            'Scoring would treat the absent topics as unretrieved, so the '
            'figure would not mean what it appears to. If this is a validation '
            'run, the qrels should be the ones for that split -- '
            'ir_baselines.build_data writes them per fold.'
        )

    return _calc_cached(qrels, run_list, measures, judged_only)


def get_metric(qrels: str, run: str, metric: str = 'AP',
               strict: bool = True, judged_only: bool = False) -> float:
    """
    `strict=True` refuses to return a figure from a run whose topic set does
    not match the qrels. Set it to False only for exploratory scoring, never
    for checkpoint selection or for anything that goes in a table.
    """
    measures = tuple(dict.fromkeys((metric, *DEFAULT_MEASURES)))
    if strict:
        agg = require_full_coverage(qrels, run, measures=measures,
                                    judged_only=judged_only)
    else:
        agg, scored, qrels_topics, _ = get_all_metrics(
            qrels, run, measures=measures, judged_only=judged_only)
        if scored != qrels_topics:
            print(f'WARNING  {len(qrels_topics - scored)} of {len(qrels_topics)} '
                  f'qrels topics are absent from the run and count as '
                  f'unretrieved.')
    return agg[metric]