# ir_baselines

Neural re-ranking baselines for ad hoc document retrieval, trained and
evaluated under one protocol so that comparisons between them are controlled.

```bash
pip install -e .
python -m ir_baselines.train --list-models
```

```
model         encoding  loss           in-batch  summary
------------  --------  -------------  --------  ----------------------------------------------------
bert          pair      cross-entropy  False     Cross-encoder over BERT-base
conv-bert     pair      cross-entropy  False     Cross-encoder over ConvBERT-base
deberta       pair      cross-entropy  False     Cross-encoder over DeBERTa-base
electra       pair      cross-entropy  False     Cross-encoder over ELECTRA-small
ernie         pair      cross-entropy  False     Cross-encoder over ERNIE 2.0 base
me-bert       dual      bce            True      One query vector, m document vectors, max inner product
              Luan et al., TACL 2021. Our implementation; see docs/models.md for the reproduction choices.
poly-encoder  dual      bce            True      m learned codes attend over the query
              Humeau et al., ICLR 2020. Our implementation; see docs/models.md for the reproduction choices.
rankt5        pair      cross-entropy  False     Cross-encoder over the T5 encoder
              T5 has no sequence-start representation, so hidden states are pooled by mean. See --t5-pooling.
roberta       pair      cross-entropy  False     Cross-encoder over RoBERTa-base

encoding  how the pair is tokenized: pair = one sequence, dual = query and document separately
in-batch  whether --loss ce-inbatch is available
```

## Usage

```bash
python -m ir_baselines.train \
  --model rankt5 \
  --train fold-0/train.jsonl --dev fold-0/dev.jsonl --qrels fold-0/dev.qrels \
  --save-dir out/rankt5/fold-0 \
  --epoch 10 --batch-size 16 --use-cuda

python -m ir_baselines.test \
  --model rankt5 \
  --test fold-0/test.jsonl --checkpoint out/rankt5/fold-0/model.bin \
  --save-dir runs/rankt5 --run fold-0.run \
  --qrels fold-0/test.qrels --use-cuda
```

Five folds, train through scoring:

```bash
MODEL=rankt5 DATA=./fold_data QRELS=./qrels.txt OUT=./runs \
  bash scripts/run_5fold.sh
```

Input is one JSON object per line — `{"query", "doc", "label"}` for training,
plus `query_id` and `doc_id` for evaluation. See
[docs/quickstart.md](docs/quickstart.md).

## What is here

| | |
|---|---|
| **cross-encoder** | one encoder over the concatenated pair, with a classifier on the pooled representation. Seven of the nine entries above. |
| **ME-BERT** | Luan et al., TACL 2021. One query vector against *m* document vectors, scored by maximum inner product. |
| **Poly-encoder** | Humeau et al., ICLR 2020. *m* learned codes attend over the query; the document vector then attends over those *m*. |
| **entity baselines** | `ir_baselines.entity.exact_match` and `ir_baselines.entity.pairwise_sim`, which score by entity overlap and entity-embedding similarity. |

Where a paper leaves a detail unspecified, the choice is a constructor argument
marked `REPRODUCTION CHOICE` in the source and documented in
[docs/models.md](docs/models.md), so it can be reported alongside a result.

**Not here:** KNRM, ConvKNRM and EDRM — use
[OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1), which
implements all three faithfully. ColBERT, whose query augmentation depends on an
attention implementation this package does not pin. Both explained in
[docs/models.md](docs/models.md).

## Reproducibility

Every checkpoint records what produced it:

```python
import torch
ck = torch.load('out/model.bin', weights_only=False)
ck['provenance']['git']                # commit, branch, and whether dirty
ck['provenance']['environment']        # torch, CUDA, transformers, GPU, host
ck['provenance']['data']               # size, lines and SHA-256 per input file
ck['config']                           # architecture, verified on load
ck['epoch'], ck['best_metric']         # where the run got to
```

The data digest is the part that earns its place: regenerating training data
with a different negative sample gives a file of the same size and line count
and different contents, and nothing else would notice.

**Training is seeded.** Two runs with the same `--seed` produce byte-identical
run files, including identical per-epoch losses.

**Interrupted runs resume exactly.** `--resume` restores the optimiser,
scheduler, scaler, epoch counter, best metric, history and random state, so the
continued run sees the data order the original would have seen. Restoring the
seed alone would reproduce step zero, not the step training stopped at.
`last.bin` is the latest state and is what `--resume` should point at;
`model.bin` is the best model and is what inference should use.

**Run files carry provenance too.** `test.py` writes a sibling
`<run>.provenance.json` recording the run's own digest, what produced it, and
what trained the checkpoint behind it — a run file cannot hold its own record,
since every TREC parser rejects a comment line. `--tag-commit` additionally
writes the short commit into field 6, for a run that gets separated from its
sibling. Read either with:

```bash
python -m ir_baselines.inspect out/model.bin
python -m ir_baselines.inspect runs/fold-0.run
```

**Mismatches are refused rather than warned about.** A checkpoint whose
configuration disagrees with the current settings, a resume whose input files
have changed, an objective the model cannot consume: each of these otherwise
produces a number that looks fine and is not.

## Design

The two model families differ in how a pair reaches the model and in what the
score means, and in nothing else. Each model declares both, so one dataset, one
trainer and one evaluator serve all of them:

```python
ENCODING = 'pair'            # or 'dual'
LOSS = 'cross-entropy'       # or 'bce'
SUPPORTS_INBATCH = False
```

Most of the rest of the code exists to catch failures that are otherwise
silent — a run missing topics scores *higher*, not lower; a checkpoint can load
into the wrong architecture with every key matched. Those are set out in
[docs/design.md](docs/design.md).

## Tests

```bash
pip install -e ".[test]"
pytest                    # everything, about four minutes
pytest -m "not slow"      # skip the ones that train, about seven seconds
```

| | |
|---|---|
| `test_scoring.py` | both models' scoring against explicit loop implementations of the paper equations, plus dtype safety and the input guards |
| `test_dispatch.py` | each `forward` signature against the batch key order the trainer uses, and every objective/encoding combination |
| `test_encoding.py` | both encodings, the `token_type_ids` fallback, and the load-time validations |
| `test_checkpoints.py` | every checkpoint layout, including the legacy prefix |
| `test_provenance.py` | git parsing, data digests, RNG round-trip, run siblings |
| `test_end_to_end.py` | the entry points as a user runs them, against a real tokenizer and a small real model. Marked `slow`. |

Nothing downloads. The end-to-end tests build a two-layer BERT from a config
and register it through `IR_BASELINES_ENCODERS`, so the suite runs offline.

The end-to-end assertions are about exit status and written artifacts rather
than about metrics: a few dozen synthetic examples say nothing about quality,
and the point is that the pipeline is wired correctly and the guards fire.

## Documentation

| | |
|---|---|
| [quickstart.md](docs/quickstart.md) | install, data format, the flags that matter |
| [models.md](docs/models.md) | each model, its reproduction choices, and the objectives |
| [design.md](docs/design.md) | why the code is shaped this way |
| [reproducing-papers.md](docs/reproducing-papers.md) | for figures from papers that used earlier versions |
| [CHANGELOG.md](CHANGELOG.md) | what changed between releases, and why |

## Licence

MIT.