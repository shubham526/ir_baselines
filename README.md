# ir_baselines

Neural re-ranking baselines used in:

- **DREQ** — Chatterjee, Mackie & Dalton. *DREQ: Document Re-ranking Using
  Entity-based Query Understanding.* ECIR 2024.
- **QDER** — Chatterjee & Dalton. *Query-Specific Document and Entity
  Representations for Multi-Vector Document Re-Ranking.* SIGIR 2025.

Both papers compare against the same baselines, trained by this code under one
protocol. The baseline run files are byte-identical between them on every
collection, so this repository is the single source for those rows rather than
a copy in each paper's repository.

**This tag is `v1.0-as-published`.** It is the code that produced the released
runs, with only those fixes that cannot change a score. Anything that would
change a number is listed in [`docs/known-issues.md`](docs/known-issues.md)
and fixed in `v2.0`. Cite whichever tag matches what you are doing:

| If you want to | Use |
|---|---|
| reproduce the published rows | `v1.0-as-published` |
| build on this code | `v2.0` |

---

## What is here

| `--model` | Class | Published rows |
|---|---|---|
| `cross-encoder` | `CrossEncoder` | BERT, RoBERTa, DeBERTa, ELECTRA, ConvBERT, RankT5, ERNIE |
| `me-bert` | `MEBERT` | ME-BERT (QDER Table 1) |
| `poly-encoder` | `PolyEncoder` | Poly-encoder (QDER Table 1) |

The seven cross-encoder rows are one class over different encoders, selected
with `--pretrain`.

**From OpenMatch.** KNRM, ConvKNRM and EDRM were run with
[OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1) (`master`
branch, `v1/` directory), not with code in this repository. Only the `task`
argument was dropped, since these papers use ranking alone. The methods
themselves are Xiong et al. (SIGIR 2017), Dai et al. (WSDM 2018) and Liu et
al. (ACL 2018).

**Not here.** PARADE, CEDR, ColBERT v2, SPLADE, ANCE-MaxP, EQFE and MaxSimCos
also appear in those papers' tables. They came from public run files, from
their authors' released implementations, or from separate code. The artifact
packages on each paper's data page hold the runs for every row, including
those.

**ME-BERT and Poly-encoder are our own implementations**, written from the
descriptions in their papers, since neither has a released implementation for
ad hoc document ranking. They are trained here under the pointwise objective
shared with the other baselines rather than the softmax cross-entropy over
sampled negatives their papers use, so the comparison is controlled but the
figures are not what the original authors would report. Their scoring
functions are also defective in this tag — see
[`docs/known-issues.md`](docs/known-issues.md).

---

## Structure

```
src/
  models/
    base.py            the contract: ENCODING, LOSS, score()
    cross_encoder.py   CrossEncoder
    multi_vector.py    MEBERT, PolyEncoder
    __init__.py        registry and shared arguments
  data/
    dataset.py         one dataset, 'pair' and 'dual' encoding
    dataloader.py
  train.py  test.py  trainer.py  evaluate.py  metrics.py  utils.py
tests/
  test_scoring.py      einsums against the published expressions
  test_encoding.py     both encoding modes, token_type_ids fallback
  test_checkpoints.py  all three checkpoint layouts
```

The two families differ only in how a pair reaches the model and in what the
score means. Each model declares both — `ENCODING` is `pair` or `dual`, `LOSS`
is `cross_entropy` or `bce`, and `score()` returns the value written to the
run file — so the dataset, trainer and evaluator do not branch on the model.

---

## Usage

Input is one JSON object per line:

```
train   {"query": ..., "doc": ..., "label": 0|1}
test    {"query_id": ..., "doc_id": ..., "query": ..., "doc": ..., "label": 0|1}
```

Train:

```bash
python src/train.py \
  --model cross-encoder --pretrain t5 \
  --train fold-0/train.jsonl --dev fold-0/dev.jsonl \
  --qrels qrels.txt --save-dir out/t5/fold-0 \
  --epoch 4 --batch-size 10 --learning-rate 2e-5 --use-cuda
```

Test:

```bash
python src/test.py \
  --model cross-encoder --pretrain t5 \
  --test fold-0/test.jsonl \
  --checkpoint out/t5/fold-0/model.bin \
  --save-dir runs/ --run fold-0.run --use-cuda
```

Then concatenate the five folds and score with `trec_eval`. Which flag to use
differs by collection — CODEC is `-Jc`, the others are `-c` — and the expected
numbers are on each paper's wiki.

### T5 pooling

T5 has no sequence-start representation, so its hidden states are averaged.
`--t5-pooling mean-all` averages over every position including padding and is
what produced the published runs; `--t5-pooling masked-mean` excludes padding.
The setting is stored in the checkpoint and checked at inference, because a
mismatch loads cleanly and changes every score.

---

## Tests

```bash
python tests/test_encoding.py
python tests/test_checkpoints.py
python tests/test_scoring.py     # fails in this tag, by design
```

`test_scoring.py` compares the multi-vector scoring functions against explicit
implementations of the expressions in their papers. In this tag both fail;
that is the record of the defect described in
[`docs/known-issues.md`](docs/known-issues.md), not a broken test. The first
two run without model weights and without network access.

---

## Runs and checkpoints

Not in this repository. The run files behind every published row, together
with a script that scores them against the paper, are on each paper's data
page:

- [DREQ](https://github.com/shubham526/ECIR2024-DREQ/wiki/Data)
- [QDER](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data)

---

## Licence

MIT. See [`LICENSE`](LICENSE).