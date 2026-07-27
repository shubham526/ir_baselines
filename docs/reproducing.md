# Reproducing the published baseline runs

The run files themselves are released, so the published numbers can be checked
with `trec_eval` alone — see *Checking without re-running* at the end. This
page is for regenerating them from the code.

Use the `v1.0-as-published` tag. Later tags change training behaviour; the
differences are in [`known-issues.md`](known-issues.md).

---

## Install

```bash
git clone https://github.com/shubham526/ir-baselines.git
cd ir-baselines
git checkout v1.0-as-published
pip install -e .
```

Everything is then run as a module:

```bash
python -m ir_baselines.train ...
python -m ir_baselines.test ...
python -m ir_baselines.entity.exact_match ...
python -m ir_baselines.entity.pairwise_sim ...
```

---

## Training configuration

Every baseline on every collection was trained with:

```json
{"Model Type": "pointwise", "Max Input": 512, "Model": "<encoder>",
 "Metric": "map", "Epochs": 4, "Batch Size": 20,
 "Learning Rate": 2e-05, "Warmup Steps": 1000}
```

As arguments:

```
--max-len 512 --epoch 4 --batch-size 20 --learning-rate 2e-5
--n-warmup-steps 1000 --metric map
```

Optimiser Adam, linear schedule with warmup. Five-fold cross-validation at the
query level; the checkpoint kept for each fold is the one with the best
validation MAP.

`Model Type: pointwise` refers to the objective: each (query, document) pair
carries a binary label. The cross-encoder computes cross-entropy over a
two-way classifier; the multi-vector models use binary cross-entropy over a
single score. Both are the same objective in a different parameterisation, and
the model declares which it needs.

---

## Data format

One JSON object per line.

```
train   {"query": "...", "doc": "...", "label": 0}
test    {"query_id": "301", "doc_id": "FBIS3-10082",
         "query": "...", "doc": "...", "label": 0}
```

The candidate set is a tuned BM25+RM3 run at depth 1000 per topic. The exact
run files used are on each paper's data page and should be used directly:
regenerating BM25 with a different toolkit or different parameters produces a
different candidate set and therefore different final numbers.

Positives and negatives are balanced for training and unbalanced for testing.

---

## Five folds

For each collection, for each fold:

```bash
python -m ir_baselines.train \
  --model cross-encoder --pretrain t5 \
  --train  data/fold-$k/train.jsonl \
  --dev    data/fold-$k/dev.jsonl \
  --qrels  data/fold-$k/dev.qrels \
  --save-dir out/t5/fold-$k \
  --max-len 512 --epoch 4 --batch-size 20 \
  --learning-rate 2e-5 --n-warmup-steps 1000 \
  --use-cuda --cuda 0

python -m ir_baselines.test \
  --model cross-encoder --pretrain t5 \
  --test data/fold-$k/test.jsonl \
  --checkpoint out/t5/fold-$k/model.bin \
  --save-dir runs/t5 --run fold-$k.run \
  --use-cuda --cuda 0
```

`--model` selects the architecture and `--pretrain` the encoder it wraps. The
seven cross-encoder rows in the published tables differ only in `--pretrain`.

Then concatenate:

```bash
cat runs/t5/fold-*.run > runs/t5.run
```

**Check the concatenation.** A run that is short a fold still scores without
error and reports a plausible figure. Two published cells were computed on
such a run before this was caught, so it is worth one command:

```bash
awk '{c[$1]++} END {n=0; s=0
  for (t in c) {n++; s+=c[t]}
  printf "%d topics, mean depth %.1f\n", n, s/n}' runs/t5.run
```

The topic count should match the qrels, and the mean depth should be close to
1000. `python -m ir_baselines.test` also prints the number of test examples
against the number of pairs written and warns if they differ.

---

## Entity baselines

ExactMatch and MaxSimCos are scoring scripts rather than trained models — no
training step, no checkpoint.

```bash
python -m ir_baselines.entity.exact_match \
  --doc-run    bm25+rm3.run \
  --docs       corpus.entities.jsonl \
  --entity-run entity_stage/master.entity.run \
  --save runs/exact_match.run --k 20

python -m ir_baselines.entity.pairwise_sim \
  --run         bm25+rm3.run \
  --docs        corpus.entities.jsonl \
  --annotations query_annotations.tsv \
  --embeddings  mmead_entities.wikipedia2vec.jsonl.gz \
  --metric cos --method max \
  --save runs/maxsimcos.run
```

`--metric cos --method max` is the published MaxSimCos row. ExactMatch writes
a much shallower run than its input, since a document sharing no entity with
the top-*k* is never ranked; that is the method, and
`src/ir_baselines/entity/README.md` gives the depths per collection.

---

## Scoring

```bash
trec_eval -c -m map -m ndcg_cut.20 -m P.20 -m recip_rank <qrels> runs/t5.run
```

**The flag differs by collection.**

| Collection | Flag |
|---|---|
| TREC Robust 2004 | `-c` |
| TREC Core 2018 | `-c` |
| TREC News 2021 | `-c` |
| TREC CAR | `-c` |
| CODEC | **`-Jc`** |

CODEC is scored `-Jc` in both papers, for every system including the
baselines. The flag is applied uniformly within each collection so no
comparison inside a table is affected, but using `-c` on CODEC will not
reproduce the published figures.

---

## Expected variance

Nothing is seeded in this tag, so two identically configured **training** runs
give different numbers. Shuffle order, dropout and initialisation all vary,
and GPU non-determinism contributes on top of that.

Treat a difference of 0.01–0.03 MAP between single runs as within the noise of
the pipeline rather than as a result. A run at or below the BM25+RM3 baseline
indicates a configuration problem rather than variance.

**Inference is deterministic.** Given the same checkpoint and the same test
file, `python -m ir_baselines.test` produces the same run every time. That is
what makes the check below meaningful.

If you need deterministic training, use `v2.0`, where the seeding,
tie-breaking and checkpoint-selection issues in
[`known-issues.md`](known-issues.md) are fixed. Those runs will not match the
published numbers exactly, for the same reason.

---

## Checking without re-running

Regenerating the runs is not necessary to check the published numbers. Each
paper's data page has one artifact package per collection, containing the run
file behind every published row and a `verify.sh` that scores each against the
value printed in the paper and reports the two side by side:

```bash
tar xzf robust04.tar.gz && cd robust04
./verify.sh /path/to/docs.graded.qrels
```

That path is deterministic and needs only `trec_eval`. It is also the only way
to check a row whose system is not in this repository — KNRM, ConvKNRM and
EDRM came from OpenMatch, and PARADE, CEDR, ColBERT v2, SPLADE, ANCE-MaxP and
EQFE from elsewhere.

- [DREQ data page](https://github.com/shubham526/ECIR2024-DREQ/wiki/Data)
- [QDER data page](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data)

---

## Verifying this code against the original

This repository merges two codebases that were previously separate — one for
the cross-encoders, one for the multi-vector models. The merge was checked by
running inference from the same checkpoint under both and comparing the
output:

```bash
python -m ir_baselines.test --model cross-encoder --pretrain t5 \
  --test <fold-0>/test.jsonl --checkpoint <fold-0>/model.bin \
  --save-dir /tmp/new --run f.run --run-tag DREQ \
  --max-len 512 --batch-size 8 --use-cuda
cmp /tmp/new/f.run <the run the original code produced>
```

On a CODEC T5 fold this is byte-identical: 8,995 pairs with the same topic,
document, rank and score on every line. T5 is the case worth checking, since
it exercises both the pooling described below and the checkpoint key remapping
described in [`known-issues.md`](known-issues.md).

`--run-tag` sets field 6 of the run file, which does not affect scoring but
does affect a byte comparison; the default is the model name.

---

## If the numbers do not match

In rough order of likelihood:

1. **Wrong evaluation flag.** CODEC needs `-Jc`.
2. **Short concatenation.** Check the topic count against the qrels.
3. **Regenerated candidate set.** Use the released BM25+RM3 runs rather than
   rebuilding them.
4. **Wrong tag.** `v2.0` changes training behaviour.
5. **Variance.** See above; repeat the run before concluding anything from a
   difference under 0.03 MAP.
6. **T5 pooling.** `--t5-pooling` must be `mean-all` for the published
   figures. `python -m ir_baselines.test` refuses a checkpoint whose stored
   setting disagrees, so this cannot happen silently.