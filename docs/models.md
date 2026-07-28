# Models

`python -m ir_baselines.train --list-models` prints this table at the command
line. What follows is the detail behind it.

| `--model` | Class | Encoding | Objective | In-batch |
|---|---|---|---|---|
| `bert` | `CrossEncoder` | pair | cross-entropy | no |
| `roberta` | `CrossEncoder` | pair | cross-entropy | no |
| `deberta` | `CrossEncoder` | pair | cross-entropy | no |
| `electra` | `CrossEncoder` | pair | cross-entropy | no |
| `conv-bert` | `CrossEncoder` | pair | cross-entropy | no |
| `ernie` | `CrossEncoder` | pair | cross-entropy | no |
| `rankt5` | `CrossEncoder` | pair | cross-entropy | no |
| `me-bert` | `MEBERT` | dual | bce or ce-inbatch | yes |
| `poly-encoder` | `PolyEncoder` | dual | bce or ce-inbatch | yes |

Seven entries are the same class over different pretrained encoders. They are
listed separately because those are the systems as usually named, and because
`--model rankt5` is a clearer thing to type than a class name plus an encoder
flag.

`--pretrain` takes a short name from the table, a path to a local model
directory, or a hub id. Further short names can be registered through the
`IR_BASELINES_ENCODERS` environment variable, as a JSON object mapping name to
path — useful for a locally fine-tuned encoder, and it survives into the
subprocesses the end-to-end tests spawn.

---

## The contract

Each model declares three things, and the dataset, trainer and evaluator read
them rather than branching on the model name:

```python
ENCODING          'pair' | 'dual'
LOSS              the objective its forward() output expects
SUPPORTS_INBATCH  whether encode_query / encode_doc / score_matrix exist
```

and implements three methods:

```python
forward(...)          training-time output, matching LOSS
score(batch, device)  the value written to the run file
config_dict()         settings that must match between training and inference
```

`score()` being separate from `forward()` is what lets one evaluator serve both
families: a two-way classifier writes the softmax probability of the relevant
class, a single-score model writes its score, and neither the evaluator nor
the run writer needs to know which it has.

Adding a model means subclassing `BaselineRanker`, declaring those three
attributes, implementing those three methods, and adding one line to
`REGISTRY`. `tests/test_dispatch.py` will then check that the `forward`
signature matches the key order the trainer uses, which is a real hazard: the
trainer passes tensors positionally, so a mismatch feeds them into the wrong
parameters.

---

## CrossEncoder

`src/ir_baselines/models/cross_encoder.py`

One encoder over the concatenated `(query, document)` sequence, with a two-way
linear classifier on the pooled representation. The run file records the
softmax probability of the relevant class.

Pooling depends on the encoder family:

| Encoder | Pooled representation |
|---|---|
| BERT and family | the sequence-start position of the last hidden state |
| DistilBERT | the same; the model takes no `token_type_ids` |
| T5 | the mean of the last hidden state — see below |

Because the query and document are encoded together there is no separate query
representation to cache, so in-batch negatives are not available and the
objective is always cross-entropy.

### T5 pooling

T5 has no sequence-start representation to pool, so the hidden states are
averaged. Two behaviours are available:

    --t5-pooling mean-all      average over every position, padding included
    --t5-pooling masked-mean   average over non-padding positions only

`mean-all` is the default. The difference is not cosmetic: on a 185-token
document padded to 512, roughly two thirds of the averaged positions are
padding, so the pooled vector is dominated by whatever the encoder emits
there.

The setting is recorded in the checkpoint and verified at inference, because it
is not a parameter — nothing in the state dict reflects it, and a mismatch
loads cleanly and changes every score.

---

## MEBERT

`src/ir_baselines/models/multi_vector.py`

Luan, Eisenstein, Toutanova & Collins. *Sparse, Dense, and Attentional
Representations for Text Retrieval.* TACL 2021.

The query is one vector and the document is *m*:

```
f^(1)(x)   = h_1(x)                        query, the sequence-start embedding
f^(m)(y)   = [h_1(y), ..., h_m(y)]         document, first m token embeddings
score(x,y) = max_j <f^(1)(x), f^(m)_j(y)>
```

The asymmetry is the point. Because the score is a maximum of inner products,
retrieval can be done with standard MIPS by adding *m* entries per document to
the index. Making the query multi-vector as well, or summing over document
vectors instead of taking the maximum, destroys that property and changes the
model.

`--me-bert-m` sets *m*, default 8. The paper uses 8 in Section 6 and 3 or 4 in
Section 7. `--me-bert-proj-dim` selects the ME-BERT-k variants, which apply a
768×k feed-forward down-projection.

**Worth knowing when reading results.** The *m* document vectors are the first
*m* **token positions**, so on a long document they cover only its opening.
That is faithful to the paper, whose experiments are on MS MARCO passages, and
it is a genuine reason the method can underperform on longer documents. Report
it rather than treating a low score as a bug.

### Reproduction choices

The paper is silent on these, so they are constructor arguments and should be
reported alongside any result:

| | Default | Note |
|---|---|---|
| `shared_encoder` | `True` | The paper says only "we encode queries and documents using BERT-base" and does not state whether the towers are tied. |
| `tie_projections` | follows `shared_encoder` | So a tied encoder does not silently get two different projections. |
| projection block | `Linear(bias=False)` + `LayerNorm` | The layer normalisation is described for DE-BERT-k; whether it carries over to ME-BERT-k is not restated. The bias term is unspecified. Applies only when `proj_dim` is set. |
| `logit_scale` | `False` | Not in the paper. See below. |

---

## PolyEncoder

`src/ir_baselines/models/multi_vector.py`

Humeau, Shuster, Lachaux & Weston. *Poly-encoders: Architectures and
Pre-training Strategies for Fast and Accurate Multi-sentence Scoring.*
ICLR 2020.

*m* learned codes each attend over the query tokens to produce *m* context
vectors; the document vector then attends over those *m*, and the score is a
dot product:

```
y_ctxt^i = sum_j w_j h_j,   (w_1..w_N) = softmax(c_i . h_1, ..., c_i . h_N)
y_ctxt   = sum_i w_i y_ctxt^i,
                            (w_1..w_m) = softmax(y_cand . y_ctxt^1, ...)
score    = y_ctxt . y_cand
```

**Role assignment.** In the paper's dialogue setting the *context* is the long
input and the *candidate* is the short item being retrieved. For document
ranking the roles follow the computational constraint rather than length: the
candidate must be a single cacheable vector, so the document is the candidate
and the query is the context. That is also how Luan et al. describe this
architecture — "computes a fixed number of vectors per query, and aggregates
them by softmax attention against document vectors".

`--poly-m` sets *m*, default 16. The paper uses 16, 64 and 360.

**Padding masking** is not discussed in the paper but is required here: without
it the first softmax distributes attention over padding positions, and with
padding to a fixed length that is most of the sequence.

### Reproduction choices

| | Default | Note |
|---|---|---|
| `shared_encoder` | `False` | Section 4.2, which 4.4 inherits: "two separate transformers ... initially start with the same weights, but are allowed to update separately during fine-tuning". |
| `code_init_std` | `hidden ** -0.5` | The paper says only "randomly initialized" and gives no distribution. |
| `logit_scale` | `False` | Not in the paper. See below. |

---

## Objectives

    --loss cross-entropy   pointwise softmax CE over a two-way classifier.
                           The cross-encoder objective, and its only one.

    --loss bce             pointwise BCEWithLogitsLoss on the pair score.
                           A controlled protocol: the same objective for every
                           model, so rows stay on the same footing.

    --loss ce-inbatch      softmax CE over the other documents in the batch.
                           Multi-vector models only. Requires
                           --positives-only.

`--loss` defaults to whatever the model expects, so it can usually be left
alone. An objective the model cannot consume is refused at argument parsing
rather than producing a shape error mid-training.

**`ce-inbatch` is not a reproduction of either paper's negative sampling.**
ME-BERT combines sampled candidates from a precomputed list with in-batch
negatives and optional hard-negative mining; Poly-encoder inherits the
Bi-encoder setup. This is a simple in-batch variant inspired by them.

Where `query_id` is present, off-diagonal entries for the same query are masked
out, so a second positive for the same query in the batch is not scored as a
negative. Without `query_id` the trainer warns once and proceeds.

### `--logit-scale`

A learnable affine calibration `a*s + b` on the score. **Not in either
multi-vector paper**, and unnecessary under their own objective, which is
scale-invariant in `b` and only mildly sensitive to `a`.

It matters under `--loss bce`. Unnormalised inner products of 768-dimensional
BERT states have a standard deviation around 10–15, so roughly half of all
pairs land in the saturated region of the sigmoid and receive gradients on the
order of 1e-4. The calibration lets the model recover a usable operating
point. Enable it with `bce`; report it when you do.

It has no effect on the cross-encoders, which score through a classifier
rather than an inner product.

---

## Models not in this package

**KNRM, ConvKNRM and EDRM.** Use
[OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1) (`master`
branch, `v1/` directory), which implements all three. They take word-level
input from a separate vocabulary rather than a transformers tokenizer, and EDRM
additionally takes entity ids and entity description ids, so they fit neither
encoding here — and a faithful upstream implementation already exists.

Their papers: Xiong et al. (SIGIR 2017), Dai et al. (WSDM 2018), Liu et al.
(ACL 2018).

**ColBERT.** Its query augmentation depends on the appended `[MASK]` positions
having their attention mask set to zero while still producing output
embeddings. Under fused attention kernels those positions return zero vectors,
so the augmentation contributes nothing to MaxSim and the model silently
degrades to a truncated bi-encoder — no error, plausible scores. Implementing
that safely means pinning an attention implementation, which is a constraint
the rest of this package does not need. Use the reference implementation.