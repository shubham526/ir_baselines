# Design notes

Why the code is shaped the way it is. Most of this exists because the failure
it prevents is silent — the pipeline runs, writes a file, and reports a number
that looks fine.

---

## One dataset, one trainer, one evaluator

Two families of model live here and they differ in exactly two ways: how a
`(query, document)` pair reaches the model, and what the score means. Every
other difference between a cross-encoder and a multi-vector model is internal
to the model.

So each model declares `ENCODING` and `LOSS`, implements `score()`, and the
rest of the pipeline reads those. The alternative — branching on the model
name in the dataset, the trainer and the evaluator — is how two codebases end
up duplicating a dataset, a loader and a run writer, and then drifting.

One consequence to be aware of: the trainer calls
`model(*[batch[k] for k in TENSOR_KEYS[ENCODING]])`, passing tensors
**positionally**. If `TENSOR_KEYS` and a `forward` signature ever disagree,
tensors land in the wrong parameters, possibly without a shape error.
`tests/test_dispatch.py` checks the correspondence by introspection.

---

## Failures that are silent, and what catches them

### A run that is missing topics scores *higher*

`pytrec_eval` averages over the topics it can score. A run missing the hard
topics therefore reports a better number than a complete one, not a worse one.
This is `trec_eval` without `-c`.

Where that number selects a checkpoint, the effect is not cosmetic: the model
that lost the most topics wins. `metrics.require_full_coverage` refuses to
return a figure unless the run's topic set matches the qrels exactly, and
validation uses it. `test.py` offers `--expected-topics` and, with `--qrels`,
compares the topic *sets* — the right count with the wrong ids passes a count
check and fails this one.

### A metric computed under the wrong convention

Two independent conventions decide what a number means, and neither is visible
in the number.

**Which topics are averaged over.** `trec_eval -c` averages over the topics in
the qrels; without it, only the topics present in the run. Under the second, a
run that loses its hard topics reports a BETTER figure, so where that figure
selects a checkpoint, the model that lost the most topics wins. Scoring here is
through ir_measures, which does `-c` by default; `require_full_coverage` is a
second line rather than the only one.

**Whether unjudged documents count.** `trec_eval -J` removes them from the
ranking; without it they count as non-relevant. Where the judgment pool is
shallow the difference is large — on one CODEC fold the same run reads AP 0.083
one way and 0.322 the other. `--judged-only` selects it, and it applies to
validation as well as evaluation: a checkpoint chosen under one convention and
reported under the other is not the checkpoint the number describes. The choice
is recorded in the checkpoint config, since it changes what `best_metric` means.

Neither convention is right in general. Each collection is reported under one
of them, and the figure has to be computed the same way to be comparable.

### A candidate whose text is missing

`build_data` needs the text of every candidate. A corpus assembled separately
from the run can be missing documents the run refers to, and those pairs are
then absent from the training data and from the run that follows — quietly, and
in a way that looks like a shorter collection rather than an error.

`retrieve search --save-corpus` writes the run and the corpus from the same
index in the same command, so the two cannot disagree. Where a corpus is
supplied separately, the count of candidates that had no text is reported
rather than left to be inferred.

### A checkpoint that loads cleanly into the wrong architecture

`load_state_dict(strict=True)` catches renamed and missing parameters. It does
not catch:

- **Tower topology.** With `shared_encoder=True` the query and document
  encoders are the same module object, so `state_dict()` still emits both key
  names. A separate-tower checkpoint loads into a tied model with every key
  matched, silently overwriting one tower with the other.
- **T5 pooling.** Not a parameter, so nothing in the state dict reflects it.
- **Sequence lengths.** Also not parameters. A model trained at 512 and
  evaluated at 250 sees truncated documents.

The architecture configuration is therefore stored inside the checkpoint
rather than in a sibling file, so a checkpoint that is copied or renamed keeps
it, and `test.py` compares it against the current settings before running. A
checkpoint carrying no configuration is refused unless
`--allow-unverified-checkpoint` is passed.

### A file of weights nobody can trace

The expensive question about any artifact is *what produced it* — which code,
which environment, which data. Answered at the time it is recorded, it costs
milliseconds. Answered two years later from a directory of `.bin` files, it
may not be answerable at all.

So every checkpoint carries a `provenance` record:

| | Why it is there |
|---|---|
| git commit, branch, describe | which code |
| **dirty flag, and which files** | a clean hash identifies the code exactly; a dirty one means the hash is a lower bound |
| torch, CUDA, cuDNN, transformers, numpy versions | `encode_plus` was removed in transformers v5, and T5 tokenisation and attention defaults have both changed within v4 |
| GPU names, hostname, platform | non-determinism and kernel differences |
| the full command line, working directory | which invocation |
| UTC timestamp | ordering |
| per input file: size, line count, **SHA-256** | which data |

The digest is the one that earns its place. Regenerating training data with a
different negative sample produces a file with the same path, the same size
and the same line count. Only the digest differs, and `--resume` compares them
before continuing an optimiser trajectory fitted to different data.

`test.py` prints a summary when it loads a checkpoint. It is not a check —
a different commit is not an error — but it is the first thing worth knowing
when a number does not match.

### A run file that cannot be traced

Run files are what get shared — emailed, attached to a submission, dropped in
a shared directory. They are also the artifact with the least provenance: six
fields per line, one of which is a free-text tag.

Putting the record inside the file is not available. `trec_eval` and
`pytrec_eval` both parse every line as six whitespace-separated fields and
reject a comment, so a `#` line makes the run unscoreable. The record
therefore goes in a sibling `<run>.provenance.json`, and the run's own SHA-256
goes inside it — which is what lets a reader tell whether the sibling belongs
to the file in front of them.

For the case where the sibling has been left behind, `--tag-commit` writes the
short commit into field 6. It refuses to do so when tracked files are modified:
a hash that does not identify the code is worse than no hash, because it looks
like it does.

### A resumed run that is not the run that was interrupted

Restoring the seed reproduces the state at step zero, not the state at the step
training stopped at, so a run resumed from the seed alone sees a different data
order. The checkpoint therefore stores the python, numpy, torch and CUDA
generator states alongside the optimiser, scheduler and scaler state.

There is a subtler version of the same problem. The main checkpoint is written
only when the validation metric improves, so it is the *best* model rather than
the *latest* — resuming from it silently discards every epoch since. A separate
`last.bin` is written at every evaluation for that purpose, and resuming from
the best checkpoint says what it is discarding.

### A fold-concatenated run that is short

`cat fold-*.run > master.run` with one fold missing produces a run that
evaluates without error and reports a plausible figure. `scripts/run_5fold.sh`
checks the topic count against the qrels before reporting anything, and names
the missing topics when it fails.

### Padding that wins a maximum

An all-padding ME-BERT document makes every slot ineligible for the maximum,
so `max` returns the masking sentinel itself — a large negative score that
ranks last and looks like a confident prediction rather than a malformed
batch. `_validate_attention_mask` rejects fully padded sequences at the point
they enter the model.

The sentinel is chosen from the tensor's own dtype rather than hard-coded:
`torch.finfo(torch.float32).min` overflows float16 and raises at
`masked_fill` time, which breaks mixed-precision training. Using `-inf`
instead would risk NaN if a row were ever fully masked.

### NaN that propagates

A non-finite loss makes every parameter NaN on the next backward pass, and
training continues without complaint. A non-finite score ranks its documents
arbitrarily. Both are raised rather than warned about.

### Ties broken by dictionary order

`save_trec` sorts by `(-score, doc_id)`, so equal-scoring documents always
rank the same way. Left to insertion order, the same scores produce different
rankings between runs.

Scores are written at full `repr` precision, so re-scoring a written run
reproduces the metric rather than a rounded version of it.

---

## Choices that cost something

**Deterministic kernels.** `utils.set_seed` disables cuDNN autotuning and
requests deterministic algorithms. That costs throughput. The trade is
deliberate: a result that cannot be reproduced is worth less than the time it
saves. Two runs with the same seed produce byte-identical run files.

**Input validation forces a sync.** `_validate_attention_mask` reads a tensor
on the host, which serialises the step boundary. `--no-validate-inputs` turns
it off once a configuration has been smoke-tested.

**Validation always runs in full precision**, regardless of `--amp`. The number
selects a checkpoint, so it should not be quantised. `--amp` affects training
only, and `test.py` says so if it is passed there.

**Warmup is a fraction of total steps**, not a fixed count. A fixed warmup can
exceed the total step count on a small fold, in which case the learning rate
never leaves the ramp. Steps are optimiser steps — batches, not examples.

**The final epoch is always evaluated.** Otherwise `--epoch 5 --eval-every 2`
never scores epoch 5, and `--eval-every` greater than `--epoch` saves no
checkpoint at all.

**Checkpoint selection uses `>`, not `>=`.** A tie keeps the earlier
checkpoint, so the saved model is the first to reach the best score rather than
the last.

**Training loss is the mean, not the sum.** A summed loss is not comparable
across epochs or across folds of different sizes.

---

## Arguments that are refused rather than warned about

Where proceeding would produce a number that looks fine and is not:

| | Why |
|---|---|
| `--positives-only` with a pointwise objective | Filtering happens before the label check, so it would train against an all-ones target. |
| `--loss ce-inbatch` without `--positives-only` | A labelled negative would be treated as another query's positive. |
| `--loss ce-inbatch` on a cross-encoder | There are no separate representations to cross-score. |
| an objective the model's output cannot feed | A shape error mid-training at best. |
| inference without `--checkpoint` | An untrained model writes a run file indistinguishable from a real one at a glance. |
| a checkpoint whose configuration disagrees | See above. |
| graded labels under a pointwise objective | `BCEWithLogitsLoss` accepts a target of 2 and optimises towards it. |
| `query_id` on some examples but not others | `collate` decides from the first example in the batch, so the symptom would depend on the shuffle seed. |
| `--resume` when the input files' digests differ | The optimiser trajectory was fitted to different data. |
| `--resume` from a checkpoint with no training state | The optimiser and schedule would restart, which is a different run. |
| `--resume` together with `--init-checkpoint` | They mean opposite things: continue a run, or start one from existing weights. |
| an ambiguous query field | On Robust04, title and description are two different published query sets; defaulting either way produces the wrong experiment silently. |
| `--positives-only` on evaluation data | A test run covering fewer documents than the systems it is compared against is not comparable to them. |