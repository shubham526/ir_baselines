# ir_baselines

Neural re-ranking baselines for ad hoc document retrieval, with the pipeline
around them: build an index, retrieve a candidate run, turn it into training
data, train, evaluate.

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

## The pipeline

Each step writes what the next one reads, so nothing has to be glued together
by hand.

```bash
# 1. index a corpus                                    (needs Java; see below)
python -m ir_baselines.retrieve index \
  --docs corpus.jsonl --index ./lucene

# 2. retrieve a candidate run, and the corpus subset that goes with it
python -m ir_baselines.retrieve search \
  --index ./lucene --queries queries.tsv \
  --method bm25+rm3 --out bm25_rm3.run --save-corpus corpus.subset.jsonl

# 3. build training, validation and test data, per fold
python -m ir_baselines.build_data \
  --run bm25_rm3.run --queries queries.tsv --docs corpus.subset.jsonl \
  --qrels qrels.txt --folds folds.json --out data/

# 4. train
python -m ir_baselines.train --model rankt5 \
  --train data/fold-0/train.jsonl --dev data/fold-0/dev.jsonl \
  --qrels data/fold-0/dev.qrels \
  --save-dir out/rankt5/fold-0 --use-cuda

# 5. evaluate
python -m ir_baselines.test --model rankt5 \
  --test data/fold-0/test.jsonl --checkpoint out/rankt5/fold-0/model.bin \
  --save-dir runs/rankt5 --run fold-0.run \
  --qrels data/fold-0/test.qrels --use-cuda
```

Or start at step 3 with a candidate run you already have. Or at step 4 with
your own JSONL — one object per line, `{"query", "doc", "label"}` for training
plus `query_id` and `doc_id` for evaluation.

Five folds at once:

```bash
MODEL=rankt5 DATA=./data QRELS=./qrels.txt OUT=./runs bash scripts/run_5fold.sh
```

Queries and documents can come from [ir_datasets](https://ir-datasets.com)
instead of local files — pass `--dataset` in place of `--queries` and `--docs`.

## What is here

| | |
|---|---|
| **cross-encoder** | one encoder over the concatenated pair, with a classifier on the pooled representation. Seven of the nine models above. |
| **ME-BERT** | Luan et al., TACL 2021. One query vector against *m* document vectors, scored by maximum inner product. |
| **Poly-encoder** | Humeau et al., ICLR 2020. *m* learned codes attend over the query; the document vector then attends over those *m*. |
| **entity baselines** | `ir_baselines.entity.exact_match` and `ir_baselines.entity.pairwise_sim`, scoring by entity overlap and entity-embedding similarity. |

Where a paper leaves a detail unspecified, the choice is a constructor argument
marked `REPRODUCTION CHOICE` in the source and documented in
[docs/models.md](docs/models.md), so it can be reported alongside a result.

**Not here:** KNRM, ConvKNRM and EDRM — use
[OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1), which
implements all three faithfully. ColBERT, whose query augmentation depends on
an attention implementation this package does not pin. Both explained in
[docs/models.md](docs/models.md).

## Evaluation

Scoring is through [ir_measures](https://ir-measur.es), which averages over the
topics in the **qrels** rather than the topics present in the run. That is
`trec_eval -c`, and it matters: under the other convention a run that loses its
hard topics reports a *better* number, and where that number selects a
checkpoint, the model that lost the most topics wins.

`--judged-only` switches to `trec_eval -J`, removing unjudged documents from
the ranking rather than counting them as non-relevant. Some collections are
reported that way, and where the judgment pool is shallow the two conventions
are far apart — on one CODEC fold the same run reads AP 0.083 under `-c` and
0.322 under `-Jc`. Use whichever the collection is reported under, for both
validation and evaluation.

Measures are ir_measures names — `AP`, `nDCG@20`, `P@20`, `RR` — with the
`trec_eval` spellings accepted as aliases.

## Reproducibility

Every checkpoint records what produced it, and so does every run:

```bash
python -m ir_baselines.inspect out/rankt5/fold-0/model.bin
python -m ir_baselines.inspect runs/rankt5/fold-0.run
```

```python
ck = torch.load('out/model.bin', weights_only=False)
ck['provenance']['git']            # commit, branch, and whether tracked files were dirty
ck['provenance']['environment']    # torch, CUDA, transformers, GPU, host
ck['provenance']['data']           # size, lines and SHA-256 per input file
ck['config']                       # architecture, verified on load
ck['epoch'], ck['best_metric']
```

A run file cannot carry its own record — every TREC parser rejects a comment
line — so `test.py` writes a sibling `<run>.provenance.json` holding the run's
own digest, what produced it, and what trained the checkpoint behind it.
`--tag-commit` additionally writes the short commit into field 6, for a run
that gets separated from its sibling.

**Training is seeded.** Two runs with the same `--seed` produce byte-identical
run files. **Interrupted runs resume exactly**: `--resume` restores the
optimiser, scheduler, scaler, epoch counter, best metric, history and random
state. `last.bin` is the latest state and is what `--resume` should point at;
`model.bin` is the best model and is what inference should use.

**Mismatches are refused, not warned about** — a checkpoint whose configuration
disagrees with the current settings, a resume whose input files have changed,
an objective the model cannot consume. Each otherwise produces a number that
looks fine and is not.

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

## Requirements

`pip install -e .` covers training and evaluation. Two extras are optional
because most of the package does not need them:

```bash
pip install -e ".[data]"        # ir_datasets, for --dataset
pip install -e ".[retrieval]"   # pyserini, for the index and search commands
pip install -e ".[test]"        # pytest
```

Pyserini needs a Java runtime — 21 for recent versions, 11 for older ones.
Check with `java -version` and `python -c "import pyserini; print(pyserini.__version__)"`.

## Tests

```bash
pip install -e ".[test]"
pytest                      # everything
pytest -m "not slow"        # skip the ones that train a model: ~10 seconds
```

125 tests. Nothing downloads: the encoder tests build a small BERT from a
config rather than fetching one, so the suite runs offline.

| | |
|---|---|
| `test_scoring.py` | both models' scoring against explicit loop implementations of the paper equations, plus dtype safety and the input guards |
| `test_dispatch.py` | each `forward` signature against the batch key order the trainer uses — it passes tensors positionally, so a mismatch would feed them into the wrong parameters |
| `test_encoding.py` | both encodings, the `token_type_ids` fallback, and every input validation |
| `test_checkpoints.py` | every checkpoint layout, including the legacy ones |
| `test_provenance.py` | git parsing, data digests, RNG round-trip, run siblings |
| `test_end_to_end.py` | the entry points as subprocesses: training, inference, resume, and every guard, asserted on exit status |

## Documentation

| | |
|---|---|
| [quickstart.md](docs/quickstart.md) | the pipeline end to end, and the flags that matter |
| [models.md](docs/models.md) | each model, its reproduction choices, and the objectives |
| [design.md](docs/design.md) | why the code is shaped this way |
| [reproducing-papers.md](docs/reproducing-papers.md) | for figures from papers that used earlier versions |
| [CHANGELOG.md](CHANGELOG.md) | what changed between releases, and why |

## Licence

MIT.