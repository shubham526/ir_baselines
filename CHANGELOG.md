# Changelog

## 2.0.0

A single package for both model families, and a reproducibility layer.

**This release does not reproduce figures from papers that used earlier
versions.** Use the `v1.0-as-published` tag for those, and see
[docs/reproducing-papers.md](docs/reproducing-papers.md).

### Added

- **`build_data`**, turning a candidate run into the training, validation and
  test JSONL the models consume, one fold tree per invocation. Queries and
  documents come from ir_datasets or from local files. Per-split qrels are
  written beside each fold, because validation scores the dev run and the full
  collection qrels would count every other fold's topics as unretrieved.
  Sampling is two flags rather than one: `--negatives-per-positive` for the
  ratio, `--negative-sampling` for which negatives. Test data is never sampled.
- **`retrieve`**, with `index` and `search` subcommands: build a Lucene index
  with Pyserini and retrieve a candidate run from it, by BM25, BM25+RM3,
  BM25+Rocchio, or BM25 followed by a trained checkpoint from this package.
  `--save-corpus` writes the corpus subset alongside the run, from the same
  index in the same command, so `build_data` cannot find a candidate whose text
  is missing.
- **`inspect`**, reading the provenance of a checkpoint or a run and warning
  when a run has changed since its sibling was written.
- **`--judged-only`**, scoring with unjudged documents removed from the ranking
  rather than counted as non-relevant — `trec_eval -J`. Some collections are
  reported that way, and where the judgment pool is shallow the conventions are
  far apart: on one CODEC fold the same run reads AP 0.083 under `-c` and 0.322
  under `-Jc`. Applies to validation as well as evaluation, and is recorded in
  the checkpoint, since it changes what the metric means.
- **`--pretrain` accepts a path or a hub id**, not only a registered short
  name, and further names can be registered through `IR_BASELINES_ENCODERS`.
  Local and fine-tuned encoders were previously unusable without editing the
  package.
- **Provenance in every checkpoint.** Git commit, branch and dirty flag with
  the modified file list; python, torch, CUDA, cuDNN, transformers and numpy
  versions; GPU names, hostname and platform; the full command line, working
  directory and a UTC timestamp; and per input file its size, line count and
  SHA-256. `test.py` prints a summary when it loads a checkpoint.
- **`--resume`**, restoring the optimiser, scheduler, scaler, epoch counter,
  best metric, history and the python/numpy/torch/CUDA random state. Refuses
  when the input files' digests differ, unless `--allow-data-change`.
- **`last.bin`**, written at every evaluation. The main checkpoint is written
  only when the metric improves, so it is the best model rather than the
  latest; resuming from it would discard every epoch since. Disable with
  `--save-last ''`.
- **Installable package.** `pip install -e .`, then
  `python -m ir_baselines.train`. src-layout, so the working copy cannot
  shadow the installed one.
- **Provenance for run files.** `test.py` writes a sibling
  `<run>.provenance.json` recording the run's own digest, topic and pair
  counts; the git commit, environment and test-data digests at inference time;
  and the checkpoint's path, digest, configuration and training-time
  provenance. A run file cannot carry its own record: `trec_eval` and
  `pytrec_eval` both reject a comment line. `--tag-commit` additionally writes
  the short commit into field 6, for a run separated from its sibling; it is
  skipped when tracked files are modified.
- **`python -m ir_baselines.inspect`**, reading the provenance of a checkpoint
  or a run and warning when a run has changed since its sibling was written.
- **`--pretrain` accepts a local directory or a hub id**, not only a short
  name, and further short names can be registered through the
  `IR_BASELINES_ENCODERS` environment variable so they reach subprocesses.
- **A pytest suite**: 125 tests across six files, of which 103 run in seven
  seconds and the rest train a small model. Nothing downloads -- the
  end-to-end tests build a two-layer BERT from a config.
- **`--list-models`**, printing each model with its encoding, objective and
  in-batch support.
- **Entity baselines** as modules: `ir_baselines.entity.exact_match` and
  `ir_baselines.entity.pairwise_sim`.
- **`tests/test_dispatch.py`**, checking that each `forward` signature matches
  the batch key order the trainer uses. The trainer passes tensors
  positionally, so a mismatch feeds them into the wrong parameters.
- **`tests/test_provenance.py`**, and eight more assertions in the smoke suite.
- **Progress on long reads.** The corpus read in `build_data` is minutes of
  silence on a real collection, and hashing a multi-gigabyte training file at
  startup looked like a hang. Both report progress, measured in bytes rather
  than lines.

### Changed

- **Evaluation moved from pytrec_eval to ir_measures**, which averages over the
  topics in the qrels rather than the topics present in the run. That is
  `trec_eval -c`, and it is the safe default: under the other convention a run
  that loses its hard topics reports a *better* number, and where that number
  selects a checkpoint, the model that lost the most topics wins. The two agree
  exactly on a complete run. Measures are now ir_measures names — `nDCG@20`,
  `AP` — with the trec_eval spellings kept as aliases. The qrels evaluator is
  cached per file, since validation re-scored against the same qrels every
  epoch.
- **One codebase for both families.** Each model declares `ENCODING`, `LOSS`
  and `SUPPORTS_INBATCH` and implements `score()`; the dataset, trainer and
  evaluator read those rather than branching on the model name. Previously the
  cross-encoders and the multi-vector models had separate copies of the
  dataset, loader, trainer, evaluator and run writer.
- **`--model` names a system**, not a class: `rankt5`, `me-bert`, `bert`.
  `--pretrain` overrides the encoder; `--loss` overrides the objective where
  the model supports more than one. Neither is needed for the usual setup.
- **`load_checkpoint` returns everything the checkpoint carries**, not just
  the config. The config is under `['config']`.
- **Optimiser is AdamW** with weight decay, was Adam.
- The package no longer refers to any particular paper. Model citations
  remain; anything paper-specific is in
  [docs/reproducing-papers.md](docs/reproducing-papers.md).

### Fixed

Each of these could change a number, which is why they are here and not in
`v1.0-as-published`:

- **ME-BERT scoring** took the maximum over query positions rather than over
  the *m* document vectors, so the multi-vector maximum never happened.
- **Poly-encoder scoring** contracted the code index and the sequence index in
  one einsum, so the softmax summed to one across the whole tensor and the
  learned context codes had no effect on the score.
- **Nothing was seeded.** Shuffle order, dropout and initialisation all varied
  between runs. Two runs with the same `--seed` are now byte-identical.
- **Run-file ties were broken by dictionary order.** Now by document id, so
  equal scores always rank the same way. Scores are written at full `repr`
  precision, so re-scoring a written run reproduces the metric.
- **Validation scored only the topics present**, which is `trec_eval` without
  `-c`: a run that lost topics reported a *higher* figure and was more likely
  to be selected as the best checkpoint. Now refused outright.
- **The final epoch was not evaluated** unless it happened to fall on
  `--eval-every`, so `--epoch 5 --eval-every 2` never scored epoch 5.
- **Warmup was a fixed step count** that could exceed the total number of
  steps on a small fold, leaving the learning rate on the ramp for the whole
  run. Now a fraction of total steps, with a clamp.
- **Checkpoint selection used `>=`**, keeping the later of two tied epochs.
  Now `>`, keeping the first to reach the best score.
- **Training loss was summed**, which is not comparable across epochs or
  across folds of different sizes. Now the mean.
- **`metrics.py` aggregated once per (query, measure) pair** rather than once
  per measure — quadratic in the number of topics, on every validation epoch.

### Removed

- **ColBERT.** Its query augmentation depends on the appended `[MASK]`
  positions having their attention mask set to zero while still producing
  output embeddings. Under fused attention kernels those positions return zero
  vectors, so the augmentation contributes nothing to MaxSim and the model
  silently becomes a truncated bi-encoder. Use the reference implementation.
- **KNRM, ConvKNRM and EDRM.** Use
  [OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1), which
  implements all three. They take word-level input from a separate vocabulary,
  and EDRM additionally takes entity and description ids, so they fit neither
  encoding here.

---

## 1.0.0-as-published

The code that produced the baseline runs released with two papers. Contains
only fixes that cannot change a score; everything that would is listed in that
tag's `docs/known-issues.md`.

Verified by running inference from one checkpoint under both the original and
the merged code: on a CODEC T5 fold the runs are identical across all 8,995
pairs in topic, document, rank and score.