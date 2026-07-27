"""
ir_baselines -- neural re-ranking baselines for ad hoc document retrieval.

Implementations of published re-ranking models, trained and evaluated under a
single protocol so that comparisons between them are controlled.

    cross-encoder     one encoder over the concatenated (query, document)
                      sequence, with a classifier on the pooled
                      representation. BERT, RoBERTa, DeBERTa, ELECTRA,
                      ConvBERT, ERNIE and RankT5 are this model over different
                      pretrained encoders.

    ME-BERT           Luan, Eisenstein, Toutanova & Collins, TACL 2021.
                      One query vector against m document vectors, scored by
                      maximum inner product.

    Poly-encoder      Humeau, Shuster, Lachaux & Weston, ICLR 2020.
                      m learned context codes attend over the query; the
                      document vector then attends over those m.

Entry points:

    python -m ir_baselines.train --model rankt5 ...
    python -m ir_baselines.test  --model rankt5 ...
    python -m ir_baselines.entity.exact_match ...
    python -m ir_baselines.entity.pairwise_sim ...

`--model` names the system; `python -m ir_baselines.train --list-models`
prints what is available. Where a paper leaves a detail unspecified, the
choice is exposed as a constructor argument and marked REPRODUCTION CHOICE in
the source, so it can be reported alongside any result.

If you are reproducing figures from a paper that used this code, see
docs/reproducing-papers.md: this release is not the one those figures were
produced with.
"""

__version__ = '2.0.0'