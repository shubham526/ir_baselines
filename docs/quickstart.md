# Quickstart

## Install

```bash
git clone https://github.com/<user>/ir-baselines.git
cd ir-baselines
pip install -e .
```

Everything runs as a module, so it works from any directory once installed:

```bash
python -m ir_baselines.train --list-models
```

## Data

One JSON object per line.

```
train   {"query": "...", "doc": "...", "label": 0}
test    {"query_id": "301", "doc_id": "FBIS3-10082",
         "query": "...", "doc": "...", "label": 0}
```

`query_id` is optional in training and required at test time along with
`doc_id`, so that a run file can be written. Including `query_id` in training
matters for `--loss ce-inbatch`: without it, a second positive for the same
query inside a batch is scored as a negative.

Labels must be 0 or 1 for the pointwise objectives. Graded relevance values
are rejected at load time rather than silently optimised towards.

## Train and test

```bash
python -m ir_baselines.train \
  --model rankt5 \
  --train fold-0/train.jsonl \
  --dev   fold-0/dev.jsonl \
  --qrels fold-0/dev.qrels \
  --save-dir out/rankt5/fold-0 \
  --epoch 10 --batch-size 16 --learning-rate 1e-5 \
  --use-cuda

python -m ir_baselines.test \
  --model rankt5 \
  --test fold-0/test.jsonl \
  --checkpoint out/rankt5/fold-0/model.bin \
  --save-dir runs/rankt5 --run fold-0.run \
  --qrels fold-0/test.qrels \
  --use-cuda
```

`--model` names the system. `--pretrain` overrides the encoder it wraps, and
`--loss` overrides the objective where the model supports more than one;
neither is needed for the usual configuration.

Five folds at once:

```bash
MODEL=rankt5 DATA=./fold_data QRELS=./qrels.txt OUT=./runs \
  bash scripts/run_5fold.sh
```

## Which arguments matter for which model

`python -m ir_baselines.train --list-models` prints the encoding and objective
for each model, which is what determines the relevant flags.

| Encoding | Length flags | Objective |
|---|---|---|
| `pair` (cross-encoders) | `--max-len` | `cross-entropy` |
| `dual` (multi-vector) | `--max-query-len`, `--max-doc-len` | `bce` or `ce-inbatch` |

Passing a flag that does not apply prints a note rather than failing, except
where it would change the result: an objective the model cannot consume is
refused.

## What a checkpoint carries

A checkpoint is not just weights. Alongside `model_state_dict` it stores:

| | |
|---|---|
| `config` | architecture settings, **verified on load** |
| `provenance` | git commit and dirty flag, python/torch/CUDA/transformers versions, GPU, hostname, command line, timestamp |
| `provenance['data']` | size, line count and **SHA-256** of every input file |
| `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` | training state, for `--resume` |
| `epoch`, `best_metric`, `history` | where the run got to |
| `rng_state` | python, numpy, torch and CUDA generators |

All of it travels inside the checkpoint rather than in sibling files, so a
checkpoint that is copied or renamed keeps its provenance.

Inspecting one:

```python
import torch
ck = torch.load('out/model.bin', weights_only=False)
print(ck['provenance']['git']['describe'])       # which code
print(ck['provenance']['environment']['transformers'])
print(ck['provenance']['data'])                  # which data, by digest
print(ck['epoch'], ck['best_metric'])
```

`test.py` prints a summary of the provenance when it loads a checkpoint. That
is not a check — a different commit or environment is not an error — but it is
the first thing worth knowing when a number does not match.

### The configuration is verified, and that matters

Several mismatches load with every key matched and no warning:

- **Tower topology.** With `--shared-encoder true` the query and document
  encoders are the same module, so `state_dict()` still emits both key names.
  A separate-tower checkpoint loads into a tied model with every key matched,
  silently overwriting one tower with the other.
- **T5 pooling.** Not a parameter at all, so nothing in the state dict
  reflects it. A checkpoint trained with `mean-all` and evaluated with
  `masked-mean` produces different scores throughout.
- **Sequence lengths.** A model trained at 512 tokens and evaluated at 250
  sees truncated documents and reports a lower figure with no indication why.

If a checkpoint carries no configuration, inference refuses rather than
guessing. `--allow-unverified-checkpoint` overrides that.

### Why the data digest

Regenerating training data with a different negative sample gives a file of
the same size and line count and different contents. A path, a size and a line
count all match; only the digest does not. `--resume` compares them and
refuses to continue an optimiser trajectory fitted to different data unless
`--allow-data-change` is passed.

`--no-data-fingerprint` skips the hashing for very large inputs.

## What a run file carries

A run file cannot carry its own provenance. The TREC format is six
whitespace-separated fields per line, and every parser — `trec_eval` and
`pytrec_eval` alike — rejects a comment line. So `test.py` writes a sibling:

```
runs/fold-0.run
runs/fold-0.run.provenance.json
```

The sibling records the run's own topic count, pair count and **SHA-256**;
the git commit, environment and test-data digests at inference time; and the
checkpoint's path, digest, configuration and its own training-time provenance.
Training and inference provenance are kept apart because they can differ, and
the difference is often the answer — the same checkpoint scored on a different
machine, or under a different transformers version, does not always produce
the same run.

The run's digest is what lets a reader tell whether the sibling belongs to
the file in front of them or to an earlier version of it.

`--no-run-provenance` skips it.

### For a run file on its own

A sibling gets separated from its run the moment someone emails one file.
`--tag-commit` puts the short commit in field 6:

```
t0 Q0 t0_d0 1 0.5102885365486145 bert.53ccb90
```

It is skipped when the working tree has modified tracked files, since a hash
that does not identify the code is worse than no hash. Untracked files do not
count: a repository almost always has some, and treating those as dirty would
mean the hash was never usable.

## Reading a provenance record

```bash
python -m ir_baselines.inspect out/model.bin
python -m ir_baselines.inspect runs/fold-0.run
python -m ir_baselines.inspect runs/fold-0.run --json
```

For a checkpoint this prints the training state, the configuration and the
provenance. For a run it prints the run's shape, what produced it, and what
trained the checkpoint behind it — and warns if the run has changed since its
sibling was written.

## Resuming an interrupted run

Two checkpoints are written:

| | |
|---|---|
| `model.bin` | the **best** model, written only when the validation metric improves. Use this for inference. |
| `last.bin` | the **latest** state, written at every evaluation. Use this for `--resume`. |

```bash
python -m ir_baselines.train ... --epoch 20 --resume out/rankt5/fold-0/last.bin
```

Resuming restores the optimiser, scheduler, scaler, epoch counter, best metric,
history and random state, so the continued run sees the data order the original
would have seen. Restoring the seed alone would reproduce step zero, not the
step training stopped at.

Resuming from `model.bin` works but discards every epoch since the best one,
and says so.

`--save-last ''` disables the second checkpoint if disk is tight; `--resume`
then has only the best checkpoint to work from.

`--init-checkpoint` is a different thing: it starts a **new** run from existing
weights, with the optimiser, schedule and epoch counter all fresh.

## What the output tells you

`test.py` prints the number of examples in the test file against the number of
pairs written, and warns if they differ. A run that silently loses pairs still
scores without error and reports a plausible figure, so it is worth reading:

```
Run written to runs/rankt5/fold-0.run
  examples in test file : 8995
  pairs written         : 8995 across 50 topics
```

`--expected-topics` makes a topic-count mismatch fatal, which is worth using
when concatenating folds. `--qrels` additionally checks that the topic *sets*
match, not just the counts: the right number of topics with the wrong ids
passes a count check and fails this one.

## Using your own encoder

`--pretrain` takes a short name, a path to a local model directory, or a hub
id:

```bash
python -m ir_baselines.train --model bert --pretrain roberta ...
python -m ir_baselines.train --model bert --pretrain /models/my-bert ...
python -m ir_baselines.train --model bert --pretrain allenai/scibert_scivocab_uncased ...
```

To give a local encoder a short name, including for subprocesses:

```bash
export IR_BASELINES_ENCODERS='{"my-bert": "/models/my-bert"}'
python -m ir_baselines.train --model bert --pretrain my-bert ...
```

## Tests

```bash
pip install -e ".[test]"
pytest                    # everything
pytest -m "not slow"      # skip the ones that train
```

Nothing downloads: the end-to-end tests build a two-layer BERT from a config
and register it through `IR_BASELINES_ENCODERS`.

## Reproducibility

Training is seeded. Two runs with the same `--seed` on the same data produce
byte-identical run files, including identical per-epoch losses.

`utils.set_seed` also requests deterministic kernels and disables cuDNN
autotuning, which costs throughput. That trade is deliberate: a result that
cannot be reproduced is worth less than the time it saves.

Inference is deterministic regardless. Ties in the run file are broken by
document id rather than left to dictionary order, so equal-scoring documents
always rank the same way.