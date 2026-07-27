# Entity baselines

Two scoring scripts, not trained models. Each reads a candidate run and writes
a re-ranked one; there is nothing to fit and no checkpoint.

| Script | Published as | Appears in |
|---|---|---|
| `exact_match.py` | ExactMatch | QDER Table 1, 2, 3 |
| `pairwise_sim.py` | MaxSimCos | DREQ Table 1, 2 |

Both were written for these papers rather than taken from a published
implementation.

---

## Where the query entities come from

This is the difference that matters when reading the two rows, and it is not
apparent from the tables.

**ExactMatch takes its query entities from the entity ranking** — the output of
the first stage that DREQ and QDER also consume. It therefore depends on that
stage in the same way they do, including on how the entity judgments behind it
were derived. See the `entity_stage/README.txt` in any collection's artifact
package.

**MaxSimCos takes its query entities from the query's own entity links**, read
from `query_annotations.tsv`. It does not use the entity ranker, the derived
entity judgments, or anything downstream of them. Of the entity-aware rows in
these tables it is the only one independent of that stage.

---

## ExactMatch

For each query, take the top-*k* entities of its entity ranking. A document
scores

    sum over those entities appearing in the document of
        (1 / rank of the entity) * (score of the entity)

and documents scoring zero are not written to the run.

```bash
python exact_match.py \
  --doc-run    bm25+rm3.run \
  --docs       corpus.entities.jsonl \
  --entity-run entity_stage/master.entity.run \
  --save       runs/exact_match.run \
  --k 20
```

**The run is much shallower than the candidate set**, because a document
sharing no entity with the top-20 never appears in it:

| Collection | Mean depth | Candidate set |
|---|---|---|
| Robust04 (title) | 16.8 | 1000 |
| Robust04 (desc) | 14.3 | 1000 |
| TREC Core 2018 | 43.6 | 1000 |
| TREC News 2021 | 53.1 | 1000 |
| CODEC | 100.8 | 999.6 |

That is the method rather than a fault. Under `trec_eval -c` the run is
charged for the relevant documents it does not rank, which is why its MAP is
low while its MRR is high — when it ranks anything at all, what it ranks tends
to be relevant.

---

## MaxSimCos

Look up each query entity and each document entity in a Wikipedia2Vec table,
form the similarity matrix between the two sets, and reduce it to a score.

```bash
python pairwise_sim.py \
  --run         bm25+rm3.run \
  --docs        corpus.entities.jsonl \
  --annotations query_annotations.tsv \
  --embeddings  mmead_entities.wikipedia2vec.jsonl.gz \
  --metric cos --method max \
  --save   runs/maxsimcos.run
```

`--metric cos --method max` is the published MaxSimCos row: cosine similarity,
the maximum over document entities for each query entity, summed. The other
three combinations were produced during development and are not in either
paper.

The confidence score attached to each query entity by the entity linker is
read but does not enter the score.

---

## Inputs

| File | Where |
|---|---|
| `bm25+rm3.run` | each collection's entry on the data page |
| entity-linked corpus | the entity-linked corpora on the data page |
| `query_annotations.tsv` | each collection's entry on the data page |
| Wikipedia2Vec embeddings | the entity resources on the data page |
| `master.entity.run` | `entity_stage/` in each collection's artifact package |

---

## TREC CAR

The CAR variants of both scripts are identical in algorithm and differ only in
input handling: paragraphs are read from a CBOR file rather than JSONL, and
entity ids are mapped from Wikipedia page ids to CAR page ids through
`id2name.tsv` and `wiki2car.tsv`.

---

## Output

Both scripts write a TREC run and then report what they produced:

```
Topics in the document run   : 250
Topics with an entity ranking: 248
Candidates scored            : 249,700
Pairs written                : 4,193
Dropped for scoring zero     : 245,507 (98.3%)
```

The last three lines are worth reading. A large drop is expected here and is
the method working as intended; a topic count below the qrels is not, and
means the run will be scored as if those topics returned nothing.

The output file is opened once in write mode, so re-running overwrites rather
than appending. An earlier version appended, and running twice without
deleting the output silently doubled the run file.