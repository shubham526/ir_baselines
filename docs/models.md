# Models

Three classes cover the eleven baseline rows this repository produced. The
cross-encoder family is one class over nine encoders; the two multi-vector
models are one class each.

| `--model` | `--pretrain` | Published as | Appears in |
|---|---|---|---|
| `cross-encoder` | `bert` | BERT | QDER Table 1, 2, 3 |
| `cross-encoder` | `roberta` | RoBERTa | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `cross-encoder` | `deberta` | DeBERTa | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `cross-encoder` | `electra` | ELECTRA | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `cross-encoder` | `conv-bert` | ConvBERT | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `cross-encoder` | `ernie` | ERNIE | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `cross-encoder` | `t5` | RankT5 (Enc) | DREQ Table 1, 2 · QDER Table 1, 2, 3 |
| `me-bert` | `bert` | ME-BERT | QDER Table 1 |
| `poly-encoder` | `bert` | Poly-encoder | QDER Table 1 |

`--pretrain distilbert` is supported but no published row uses it.

KNRM, ConvKNRM and EDRM also appear in those tables and are **not** in this
repository — see *From OpenMatch* below.

---

## CrossEncoder

`src/models/cross_encoder.py`

One encoder over the concatenated `(query, document)` sequence, with a
two-way linear classifier on the pooled representation. The run file records
the softmax probability of the relevant class.

Pooling depends on the encoder family:

| Encoder | Pooled representation |
|---|---|
| BERT and family | the sequence-start position of the last hidden state |
| DistilBERT | the same; the model takes no `token_type_ids` |
| T5 | the mean of the last hidden state — see below |

**T5 pooling.** T5 has no sequence-start representation to pool, so the hidden
states are averaged. The published runs average over *every* position,
padding included: on a 185-token document padded to 512, roughly two thirds
of the averaged positions are padding. Both behaviours are available:

    --t5-pooling mean-all      average over all positions (default; published)
    --t5-pooling masked-mean   average over non-padding positions only

The setting is written into the checkpoint and checked at inference. A
checkpoint trained with one pooling and evaluated with the other loads with
every key matched and produces different scores throughout, so `test.py`
refuses the mismatch rather than warning about it.

---

## MEBERT

`src/models/multi_vector.py`

Luan, Eisenstein, Toutanova & Collins. *Sparse, Dense, and Attentional
Representations for Text Retrieval.* TACL 2021.

The method represents the query by one vector and the document by *m*, and
scores a pair by the maximum inner product over the document's *m* vectors.
The asymmetry is the point: because the score is a maximum of inner products,
retrieval can be done with standard MIPS by adding *m* entries per document to
the index.

**This is our own implementation.** No implementation for ad hoc document
ranking has been released, so it was written from the equations in the paper.
See *Attribution* below.

**The implementation in this tag does not compute that expression.** The query
is represented by *m* vectors rather than one, and the maximum is taken over
query positions rather than over the *m* document vectors, so the multi-vector
maximum never happens. It is kept as it is here because it produced the
published row; the revised version is in `v2.0`. See
[`known-issues.md`](known-issues.md).

`--me-bert-m` selects *m*, default 8. The paper uses 8 in Section 6 and 3 or 4
in Section 7.

One property worth knowing when reading the numbers: the *m* document vectors
are the first *m* **token positions**, so on a newswire collection they cover
only the opening of the article. That is faithful to the paper, whose
experiments are on MS MARCO passages, and it is a plausible reason the method
underperforms on these collections independently of the defect above.

---

## PolyEncoder

`src/models/multi_vector.py`

Humeau, Shuster, Lachaux & Weston. *Poly-encoders: Architectures and
Pre-training Strategies for Fast and Accurate Multi-sentence Scoring.*
ICLR 2020.

*m* learned context codes each attend over the input sequence to produce *m*
context vectors; the candidate vector then attends over those *m* to produce a
single representation, and the score is a dot product.

**This is our own implementation**, for the same reason as ME-BERT.

**The implementation in this tag does not compute that expression.** The
attention contracts both the code index and the sequence index in one
`einsum`, so the softmax sums to one across the whole tensor and the codes
have no effect on the score — the document representation is an unweighted sum
over all positions, padding included. The learned codes, the mechanism the
method is named for, are inert. Kept as it is here for the same reason; see
[`known-issues.md`](known-issues.md).

`--poly-m` selects *m*, default 16. The paper uses 16, 64 and 360.

---

## From OpenMatch

KNRM, ConvKNRM and EDRM were run with
[OpenMatch](https://github.com/thunlp/OpenMatch/tree/master/v1), `master`
branch, `v1/` directory, rather than with code in this repository. The only
change was to drop the `task` argument, since these papers use ranking alone
and never the classification variant; the forward passes were not modified.

The specific commit was not recorded, and OpenMatch's `master` may have moved
since. The relevant files are `v1/OpenMatch/models/knrm.py`, `conv_knrm.py`
and `edrm.py`.

Their methods:

| Model | Paper                                                                                                                                                              |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| KNRM | Xiong, Dai, Callan, Liu & Power. *[End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://dl.acm.org/doi/10.1145/3077136.3080809).* SIGIR 2017.             |
| ConvKNRM | Dai, Xiong, Callan & Liu. *[Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](https://dl.acm.org/doi/10.1145/3159652.3159659).* WSDM 2018. |
| EDRM | Liu, Xiong, Sun, & Liu. *[Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](https://aclanthology.org/P18-1223/).* ACL 2018.           |

These take word-level input from a separate vocabulary rather than a
transformers tokenizer, and EDRM additionally takes entity IDs and entity
description IDs, so they do not fit either encoding mode used here. That is
the practical reason they were not brought into this repository, beyond the
fact that a faithful upstream implementation already exists.

---

## Attribution

Where a public run file or a released implementation existed for a system on a
collection, that is what the papers used, and it is not in this repository.
Where none existed, the system was implemented here and trained under the
shared protocol.

**ME-BERT and Poly-encoder are ours.** Two things differ from their papers
beyond the code itself, and both apply to the published rows:

- **Training objective.** Both papers train with softmax cross-entropy over
  sampled and in-batch negatives. These rows use the pointwise binary
  cross-entropy objective shared with every other baseline in the table, so
  that the comparison is controlled. A figure in these rows therefore reflects
  the architecture under this protocol, not what the original authors would
  report.
- **Score calibration.** Unnormalised inner products of 768-dimensional BERT
  states have a standard deviation near 14, which places roughly half of all
  pairs in the saturated region of the sigmoid under a pointwise objective. An
  affine calibration of the score was added to compensate. It appears in
  neither paper and is unnecessary under their own objective, which is
  scale-tolerant.

**Not in this repository.** KNRM, ConvKNRM and EDRM came from OpenMatch, as
above. PARADE, CEDR, ColBERT v2, SPLADE, ANCE-MaxP, AND EQFE also
appear in the published tables and came from public runs, from their authors'
implementations, or from separate code. The artifact packages on each paper's
data page hold the run files for every row, including all of these, together
with a note of where each came from.

---

## Adding a model

Subclass `BaselineRanker` and declare three things:

```python
class MyRanker(BaselineRanker):
    ENCODING = 'pair'          # or 'dual'
    LOSS = 'cross_entropy'     # or 'bce'

    def forward(self, **batch): ...
    def score(self, batch, device): ...      # -> [B], what goes in the run
    def config_dict(self): ...               # stored in the checkpoint
```

Then register it in `src/models/__init__.py`. The dataset, trainer and
evaluator read `ENCODING` and `LOSS` and do not branch on the model name, so
nothing else needs changing.

`score()` returning the run-file value rather than a raw logit is deliberate:
the cross-encoder writes a softmax probability and the multi-vector models
write a raw score, and keeping that inside the model is what lets one
evaluator serve both.