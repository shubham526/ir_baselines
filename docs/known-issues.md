# Known issues in this release

`v1.0-as-published` is the code that produced the run files released with the
DREQ (ECIR 2024) and QDER (SIGIR 2025) papers. It is tagged separately from
later releases so that those runs remain reproducible from the code that made
them.

Issues that would change a score are therefore **not** fixed here. They are
listed below, and fixed in `v2.0`.

---

## Fixed in this release

These prevented the code from running at all, or from running on a current
environment. None of them can change a score.

| | |
|---|---|
| `--models` where the code read `args.model` | Both entry points raised `AttributeError` on the first line of `main()`. |
| `tokenizer.encode_plus` | Removed in transformers v5. The `__call__` form takes the same arguments and returns the same tokens. |
| `token_type_ids` assumed present | The T5 and DistilBERT tokenizers emit none, so `--pretrain t5` — the configuration behind the published T5 rows — raised `KeyError` on the first batch. Zeros are supplied; neither model uses them. |
| `load_checkpoint` read `state_dict['model_state_dict']` | `save_checkpoint` writes a bare state dict, so this raised `KeyError` on every checkpoint the code produces. |
| Checkpoints prefixed `t5.` | The released checkpoints name the encoder attribute `t5.`; the model calls it `encoder.`. Same tensors, same shapes. Without the remap a strict load fails; without strict loading it would silently leave the encoder randomly initialised. |
| `train.py` loaded checkpoints inline | No `map_location`, no unwrap, no remap — so resuming training failed on checkpoints inference could read. Both paths now go through `utils`. |
| `config.json` written to a directory that may not exist | Crashed on a fresh output directory. |

---

## Not fixed in this release

Each of these can change a score, so fixing them here would mean the tag no
longer reproduces the published runs.

### ME-BERT and Poly-encoder scoring

Neither implementation computes the expression given in its paper.

**Poly-encoder.** The attention over the learned context codes contracts both
the code index and the sequence index in one `einsum`, so the softmax sums to
one across the whole tensor and the codes have no effect on the score. The
document representation is an unweighted sum over all positions, padding
included. The learned codes — the mechanism the method is named for — are
inert.

**ME-BERT.** The query is represented by *m* vectors rather than one, and the
maximum is taken over query positions rather than over the *m* document
vectors, so the multi-vector maximum that defines the method never happens.

`tests/test_scoring.py` compares both against explicit implementations of the
published expressions and fails on both. That failure is the record of this
issue, not a broken test.

Revised implementations, and the figures they give, are in `v2.0`. Both
methods remain far below every other baseline under either implementation, so
no ranking or significance annotation changes.

### T5 pooling over padding

`CrossEncoder` averages T5's hidden states over every position, padding
included. On a 185-token document padded to 512, roughly two thirds of the
averaged positions are padding.

Both poolings are available here through `--t5-pooling`, and the default is
`mean-all`, which is what produced the published runs. The setting is stored
in the checkpoint and checked at inference: a checkpoint trained with one
pooling and evaluated with the other loads with every key matched and produces
different scores throughout.

### Non-determinism

Nothing is seeded — shuffle order, dropout and initialisation all vary between
runs, so two identically configured runs give different numbers.

### Tie-breaking in the run file

`save_trec` sorts by score alone, so documents with equal scores are ordered
by dictionary insertion. Two runs over the same scores can produce different
rankings.

### Validation scoring

`metrics.py` scores only the topics present in the run, which is `trec_eval`
without `-c`. A run that loses topics therefore reports a **higher**
validation figure and is more likely to be selected as the best checkpoint.
It also aggregates once per (query, measure) pair rather than once per
measure, which is quadratic in the number of topics and runs on every
validation epoch.

### Final epoch not evaluated

Validation runs only when `(epoch + 1) % eval_every == 0`, so `--epoch 20
--eval-every 3` never scores the last epoch.

### Fixed warmup

`--n-warmup-steps` defaults to 1000 regardless of dataset size. On a small
fold that can exceed the total number of optimiser steps, leaving the learning
rate on the warmup ramp for the entire run.

### Checkpoint selection on ties

`>=` keeps the later checkpoint when two epochs score identically, rather than
the first to reach that score.
