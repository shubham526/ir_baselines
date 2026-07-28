# Reproducing figures from papers that used this code

This package was extracted from the code behind two papers:

- **DREQ** — Chatterjee, Mackie & Dalton. *DREQ: Document Re-ranking Using
  Entity-based Query Understanding.* ECIR 2024.
- **QDER** — Chatterjee & Dalton. *Query-Specific Document and Entity
  Representations for Multi-Vector Document Re-Ranking.* SIGIR 2025.

**This release does not reproduce those figures.** Use the
`v1.0-as-published` tag, which is the code that produced them.

Nothing else in this repository depends on either paper; if you are using the
package for your own work you can stop reading here.

---

## Which tag to use

| If you want to | Use |
|---|---|
| reproduce a published row | `v1.0-as-published` |
| build on the code | the current release |

`v1.0-as-published` contains only those fixes that cannot change a score. Its
`docs/known-issues.md` lists what was deliberately left broken and why. This
release fixes all of it, which is why it does not reproduce.

The differences that change a number:

| | v1.0-as-published | now |
|---|---|---|
| ME-BERT scoring | maximum over the wrong axis | as the paper defines it |
| Poly-encoder scoring | learned codes inert | as the paper defines it |
| seeding | none | seeded; identical seeds give identical runs |
| run-file ties | dictionary order | broken by document id |
| validation scoring | topics present only, so an incomplete run scored higher | refuses an incomplete run |
| final epoch | not evaluated unless it landed on `--eval-every` | always evaluated |
| warmup | fixed count, could exceed total steps | fraction of total steps |
| checkpoint selection | `>=`, so a tie kept the later one | `>`, keeps the first to reach it |
| optimiser | Adam | AdamW with weight decay |

The ME-BERT and Poly-encoder rows move most. Both remain far below every other
baseline under either implementation, so no ranking or significance annotation
in either paper changes.

---

## Checking the published numbers without running anything

Regenerating runs is not necessary and not the easiest path. Each paper's data
page has one artifact package per collection, containing the run file behind
every published row and a `verify.sh` that scores each against the value
printed in the paper:

```bash
tar xzf robust04.tar.gz && cd robust04
./verify.sh /path/to/docs.graded.qrels
```

That needs only `trec_eval`, is deterministic, and covers rows whose systems
are not in this package at all — KNRM, ConvKNRM and EDRM came from OpenMatch,
and PARADE, CEDR, ColBERT v2, SPLADE, ANCE-MaxP and EQFE from elsewhere.

---

## Model names

`--model` names changed when the package was generalised. The published tables
print the left column:

| Printed as | `--model` |
|---|---|
| BERT | `bert` |
| RoBERTa | `roberta` |
| DeBERTa | `deberta` |
| ELECTRA | `electra` |
| ConvBERT | `conv-bert` |
| ERNIE | `ernie` |
| RankT5 (Enc) | `rankt5` |
| ME-BERT | `me-bert` |
| Poly-encoder | `poly-encoder` |
| ExactMatch | `python -m ir_baselines.entity.exact_match` |
| MaxSimCos | `python -m ir_baselines.entity.pairwise_sim` |

---

## Training configuration used for the published runs

Every baseline on every collection:

```
--max-len 512 --epoch 4 --batch-size 20 --learning-rate 2e-5
--n-warmup-steps 1000 --metric map
```

Optimiser Adam — not AdamW, which is the default here. Five-fold
cross-validation at the query level; the checkpoint kept per fold is the one
with the best validation MAP.

The candidate set is a tuned BM25+RM3 run at depth 1000 per topic. Use the
released run files rather than regenerating them: a different toolkit or
different parameters produces a different candidate set and therefore different
final numbers.

---

## Scoring flags

**The flag differs by collection**, and using the wrong one will not reproduce
the published figure:

| Collection | Flag |
|---|---|
| TREC Robust 2004 | `-c` |
| TREC Core 2018 | `-c` |
| TREC News 2021 | `-c` |
| TREC CAR | `-c` |
| CODEC | **`-Jc`** |

CODEC is scored `-Jc` in both papers, for every system including the
baselines. It is applied uniformly within the collection, so no comparison
inside a table is affected — but the flag must be used.

In this package that is `--judged-only`, on both `train` and `test`. Without
it the figures are computed under `-c` and are not comparable to the published
ones: on one CODEC fold the same run reads AP 0.083 under `-c` and 0.322 under
`-Jc`.

---

## Errata

Individual cells in both papers' tables do not match the runs that produced
them, and two reported rows were computed over a smaller candidate pool than
the baselines they are compared against. Those are listed with corrected
values on each paper's Errata page, not here: they are properties of the
tables, not of this code.