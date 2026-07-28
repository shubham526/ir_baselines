# Quickstart

## Install

```bash
git clone https://github.com/<user>/ir_baselines.git
cd ir_baselines
pip install -e .
```

Everything runs as a module, so it works from any directory once installed:

```bash
python -m ir_baselines.train --list-models
```

Two extras, optional because most of the package does not need them:

```bash
pip install -e ".[data]"        # ir_datasets, for --dataset
pip install -e ".[retrieval]"   # pyserini, for index and search
```

Pyserini needs a Java runtime — 21 for recent versions, 11 for older ones. A
version mismatch gives class-file errors rather than anything obvious, so check
both:

```bash
java -version
python -c "import pyserini; print(pyserini.__version__)"
```

---

## The pipeline

Five steps. Each writes what the next reads, and you can start at whichever one
matches what you already have.

### 1. Index a corpus

```bash
python -m ir_baselines.retrieve index \
  --docs corpus.jsonl --index ./lucene --threads 16
```

`corpus.jsonl` is one object per line with `doc_id` and a text field. Pyserini
wants a different shape, so the conversion is done for you and streamed, since
a collection corpus is usually larger than memory.

`--dataset <ir_datasets id>` indexes a collection from ir_datasets instead.

The index is built with `--storeRaw`, which is what lets step 2 read document
text back out. Skip it and `--save-corpus` produces nothing.

### 2. Retrieve a candidate run

```bash
python -m ir_baselines.retrieve search \
  --index ./lucene --queries queries.tsv \
  --method bm25+rm3 --k 1000 \
  --out bm25_rm3.run --save-corpus corpus.subset.jsonl
```

| `--method` | |
|---|---|
| `bm25` | BM25 alone |
| `bm25+rm3` | with RM3 pseudo-relevance feedback; the usual candidate ranking here |
| `bm25+rocchio` | with Rocchio feedback |
| `bm25+rerank` | BM25 then re-ranked by a trained checkpoint from this package |

**`--save-corpus` is worth using.** It writes `{doc_id, text}` for exactly the
documents in the run, from the same index in the same command. Step 3 needs
that text, and a corpus assembled separately can be missing documents the run
refers to — those candidates are then dropped without comment. It also avoids
reading a whole collection to use a small part of it: CODEC's corpus is 4.3 GB
for a candidate set of about forty thousand documents.

Tune `--k1` and `--b` if you have values for the collection. The defaults are
Pyserini's, not tuned, and a re-ranker inherits whatever its first stage gives
it.

### 3. Build the data

```bash
python -m ir_baselines.build_data \
  --run bm25_rm3.run --queries queries.tsv --docs corpus.subset.jsonl \
  --qrels qrels.txt --folds folds.json --out data/
```

writes, per fold:

```
data/fold-0/train.jsonl
data/fold-0/dev.jsonl     data/fold-0/dev.qrels
data/fold-0/test.jsonl    data/fold-0/test.qrels
```

**The per-split qrels matter.** Validation scores the dev run; passing the full
collection qrels means every topic in the other folds counts as unretrieved,
which training refuses.

**Sampling is two decisions, and they are two flags.**
`--negatives-per-positive` sets the ratio, defaulting to 1 because that is what
the published experiments used — a convention, not a validated choice.
`--negative-sampling` chooses which negatives, defaulting to `top`, which takes
a prefix of the run and so trains against the highest-ranked non-relevant
documents. That second choice does more work than the ratio.

Test data is never sampled: it carries the whole candidate list, because a run
covering fewer documents than the systems it is compared against is not
comparable to them.

**Field names differ between collections.** BEIR exposes `text`, native TREC
exposes `title` and `body`, CORD-19 and nfcorpus expose `abstract`. The field is
resolved per dataset, and where the choice is ambiguous it is refused rather
than guessed — on Robust04, `--query-field title` and `--query-field
description` are two different published query sets, and defaulting either way
would silently produce the wrong experiment.

### 4. Train

```bash
python -m ir_baselines.train --model rankt5 \
  --train data/fold-0/train.jsonl --dev data/fold-0/dev.jsonl \
  --qrels data/fold-0/dev.qrels \
  --save-dir out/rankt5/fold-0 \
  --epoch 10 --batch-size 16 --learning-rate 1e-5 --use-cuda
```

`--model` names the system; `--list-models` prints what is available with each
one's encoding and objective. `--pretrain` overrides the encoder — a short name,
a path to a local model directory, or a hub id. `--loss` overrides the objective
where the model supports more than one.

| Encoding | Length flags | Objective |
|---|---|---|
| `pair` (cross-encoders) | `--max-len` | `cross-entropy` |
| `dual` (multi-vector) | `--max-query-len`, `--max-doc-len` | `bce` or `ce-inbatch` |

A flag that does not apply prints a note; an objective the model cannot consume
is refused.

### 5. Evaluate

```bash
python -m ir_baselines.test --model rankt5 \
  --test data/fold-0/test.jsonl --checkpoint out/rankt5/fold-0/model.bin \
  --save-dir runs/rankt5 --run fold-0.run \
  --qrels data/fold-0/test.qrels --use-cuda
```

Five folds at once:

```bash
MODEL=rankt5 DATA=./data QRELS=./qrels.txt OUT=./runs bash scripts/run_5fold.sh
```

---

## Scoring conventions

Scoring is through ir_measures, which averages over the topics in the **qrels**
rather than the topics present in the run — `trec_eval -c`. Under the other
convention a run that loses its hard topics reports a *better* number.

`--judged-only` switches to `trec_eval -J`: unjudged documents are removed from
the ranking rather than counted as non-relevant. Where the judgment pool is
shallow the two are far apart — on one CODEC fold the same run reads AP 0.083
under `-c` and 0.322 under `-Jc`.

Use whichever the collection is reported under, and use it for **both**
validation and evaluation, or the checkpoint is selected under one convention
and reported under another. The choice is recorded in the checkpoint, since it
changes what the metric means.

`--metric` takes ir_measures names — `AP`, `nDCG@20`, `P@20`, `RR` — with the
trec_eval spellings (`map`, `ndcg_cut_20`, `P_20`, `recip_rank`) as aliases.

---

## What a checkpoint carries

| | |
|---|---|
| `config` | architecture settings, **verified on load** |
| `provenance` | git commit and dirty flag, python/torch/CUDA/transformers versions, GPU, hostname, command line, timestamp |
| `provenance['data']` | size, line count and **SHA-256** of every input file |
| `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` | training state, for `--resume` |
| `epoch`, `best_metric`, `history` | where the run got to |
| `rng_state` | python, numpy, torch and CUDA generators |

```bash
python -m ir_baselines.inspect out/rankt5/fold-0/model.bin
python -m ir_baselines.inspect runs/rankt5/fold-0.run
python -m ir_baselines.inspect runs/rankt5/fold-0.run --json
```

For a run this prints the whole chain — run, what produced it, the checkpoint,
and what trained that checkpoint — and warns if the run has changed since its
provenance sibling was written.

### The configuration is verified, and that matters

Several mismatches load with every key matched and no warning:

- **Tower topology.** With `--shared-encoder true` the query and document
  encoders are the same module, so `state_dict()` still emits both key names. A
  separate-tower checkpoint loads into a tied model with every key matched,
  silently overwriting one tower with the other.
- **T5 pooling.** Not a parameter at all, so nothing in the state dict reflects
  it. A checkpoint trained with `mean-all` and evaluated with `masked-mean`
  produces different scores throughout.
- **Sequence lengths.** A model trained at 512 tokens and evaluated at 250 sees
  truncated documents and reports a lower figure with no indication why.

A checkpoint carrying no configuration is refused unless
`--allow-unverified-checkpoint` is passed.

### Why the data digest

Regenerating training data with a different negative sample gives a file of the
same size and line count and different contents. Only the digest differs, and
`--resume` compares them before continuing an optimiser trajectory fitted to
different data.

---

## What a run file carries

A run file cannot carry its own provenance: the TREC format is six
whitespace-separated fields per line, and every parser rejects a comment. So
`test.py` writes a sibling:

```
runs/rankt5/fold-0.run
runs/rankt5/fold-0.run.provenance.json
```

holding the run's own digest, the inference-time environment, and the
checkpoint's path, digest, configuration and training-time provenance. The run
digest is what lets a reader tell whether the sibling belongs to the file in
front of them.

`--tag-commit` puts the short commit in field 6 — `bert.53ccb90` — for a run
separated from its sibling. It is skipped when tracked files are modified,
since a hash that does not identify the code is worse than no hash.

---

## Resuming an interrupted run

| | |
|---|---|
| `model.bin` | the **best** model, written only when the validation metric improves. Use for inference. |
| `last.bin` | the **latest** state, written at every evaluation. Use for `--resume`. |

```bash
python -m ir_baselines.train ... --epoch 20 --resume out/rankt5/fold-0/last.bin
```

Resuming restores the optimiser, scheduler, scaler, epoch counter, best metric,
history and random state, so the continued run sees the data order the original
would have seen. Restoring the seed alone would reproduce step zero, not the
step training stopped at.

Resuming from `model.bin` works but discards every epoch since the best one,
and says so. `--init-checkpoint` is a different thing: it starts a **new** run
from existing weights.

---

## Output worth reading

```
Run written to runs/rankt5/fold-0.run
  examples in test file : 8995
  pairs written         : 8995 across 9 topics
```

A run that silently loses pairs still scores and reports a plausible figure.
`--expected-topics` makes a topic-count mismatch fatal, which is worth using
when concatenating folds; `--qrels` additionally compares the topic *sets*,
since the right count with the wrong ids passes a count check and fails this
one.

---

## Reproducibility

Training is seeded: two runs with the same `--seed` produce byte-identical run
files. `utils.set_seed` also requests deterministic kernels, which costs
throughput — a deliberate trade.

Some GPU attention implementations are non-deterministic in the backward pass
and torch warns rather than failing. Inference is unaffected; if bit-exact
*training* matters, verify it on your hardware:

```bash
for i in 1 2; do
  python -m ir_baselines.train --model bert --seed 42 ... --save-dir /tmp/det$i
done
python -c "
import json
for i in (1,2): print(json.load(open(f'/tmp/det{i}/training_history.json'))['train_loss'])"
```

Inference is deterministic regardless. Ties in the run file break by document
id rather than dictionary order, so equal-scoring documents always rank the
same way.

---

## Tests

```bash
pip install -e ".[test]"
pytest
pytest -m "not slow"        # skip the ones that train a model
```

Nothing downloads: the encoder tests build a small BERT from a config, so the
suite runs offline in seconds.

If you add a model, `test_dispatch.py` will check that its `forward` signature
matches the batch key order the trainer uses. That check exists because the
trainer passes tensors positionally, so a mismatch feeds them into the wrong
parameters — possibly without a shape error.