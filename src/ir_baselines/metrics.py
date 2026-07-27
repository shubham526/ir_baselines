from typing import Dict, Set, Tuple

import pytrec_eval


def _load(qrels: str, run: str):
    with open(qrels, 'r') as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)
    with open(run, 'r') as f_run:
        run_dict = pytrec_eval.parse_run(f_run)
    return qrel_dict, run_dict


def _aggregate(qrel_dict, run_dict) -> Tuple[Dict[str, float], Set[str]]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run_dict)
    if not results:
        return {}, set()
    measures = next(iter(results.values())).keys()
    # Once per measure, not once per (query, measure) pair. The original nested
    # loop recomputed every aggregate for every query -- quadratic in the number
    # of topics, on every validation epoch.
    agg = {
        measure: pytrec_eval.compute_aggregated_measure(
            measure, [qm[measure] for qm in results.values()]
        )
        for measure in measures
    }
    return agg, set(results)


def get_all_metrics(qrels: str, run: str) -> Tuple[Dict[str, float], Set[str], Set[str], Set[str]]:
    """
    Returns (aggregated metrics, topics scored, topics in qrels, topics in run).

    All three topic sets are returned because they differ in ways that matter.
    pytrec_eval scores only the intersection: a topic in the run but not in the
    qrels is dropped from the evaluator's output entirely, so the scored set can
    never contain it. Detecting an unexpected topic therefore requires the raw
    run set, not the evaluator's result keys.

    Scoring only the topics present is the equivalent of trec_eval WITHOUT `-c`,
    which means a run that loses topics reports a *higher* metric, not a lower
    one.
    """
    qrel_dict, run_dict = _load(qrels, run)
    agg, scored = _aggregate(qrel_dict, run_dict)
    return agg, scored, set(qrel_dict), set(run_dict)


def require_full_coverage(qrels: str, run: str, context: str = 'run') -> Dict[str, float]:
    """
    Scores a run and refuses to return a metric unless its topic set matches the
    qrels exactly.

    The comparison is against the raw parsed run, so a topic present in the run
    but absent from the qrels is caught. An incomplete or misaligned run still
    evaluates without error and reports a plausible, inflated number; where that
    number selects a checkpoint or lands in a table, a warning is not enough.
    """
    qrel_dict, run_dict = _load(qrels, run)
    qrels_topics, run_topics = set(qrel_dict), set(run_dict)

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
            'pytrec_eval averages over the topics it can score, so the reported '
            'figure would be inflated relative to trec_eval -c.'
        )

    agg, _ = _aggregate(qrel_dict, run_dict)
    return agg


def get_metric(qrels: str, run: str, metric: str = 'map', strict: bool = True) -> float:
    """
    `strict=True` refuses to return a figure from a run whose topic set does not
    match the qrels. Set it to False only for exploratory scoring, never for
    checkpoint selection or for anything that goes in a table.
    """
    if strict:
        agg = require_full_coverage(qrels, run)
    else:
        agg, scored, qrels_topics, run_topics = get_all_metrics(qrels, run)
        if scored != qrels_topics:
            print(f'WARNING  scored {len(scored)} of {len(qrels_topics)} qrels topics; '
                  f'{metric} is averaged over those only.')
    if metric not in agg:
        raise KeyError(f'metric {metric!r} not available; got {sorted(agg)[:10]}...')
    return agg[metric]