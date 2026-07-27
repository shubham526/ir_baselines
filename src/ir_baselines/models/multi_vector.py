"""
Multi-vector baselines for document re-ranking.

MEBERT      Luan, Eisenstein, Toutanova & Collins.
            "Sparse, Dense, and Attentional Representations for Text Retrieval."
            TACL 2021.

PolyEncoder Humeau, Shuster, Lachaux & Weston.
            "Poly-encoders: Architectures and Pre-training Strategies for Fast
            and Accurate Multi-sentence Scoring." ICLR 2020.

Both are implemented from the equations in the papers. Each class separates
encoding from scoring, so documents can be encoded once and reused, and so a
batch of queries can be scored against a batch of documents for in-batch
negative training:

    encode_query(...)   -> query representation
    encode_doc(...)     -> document representation
    score_pairs(...)    -> [B]      aligned pairs, query i against document i
    score_matrix(...)   -> [Q, N]   every query against every document
    forward(...)        -> [B]      aligned pairs, for pointwise/pairwise losses

Points where the papers are silent are marked REPRODUCTION CHOICE and exposed
as constructor arguments. Those must be reported alongside any results.

The scoring functions are checked against explicit implementations of the
paper equations by tests/test_scoring.py.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .base import (
    BaselineRanker,
    _LogitScale,
    _make_projection,
    _mask_fill,
    _masked_softmax,
    _validate_attention_mask,
)


# ---------------------------------------------------------------------------
# ME-BERT
# ---------------------------------------------------------------------------

class MEBERT(BaselineRanker):
    r"""
    Multi-Vector Encoding from BERT (Luan et al., TACL 2021, Section 3).

    The query is represented by ONE vector and the document by m vectors:

        f^(1)(x)   = h_1(x)                        query, the [CLS] embedding
        f^(m)(y)   = [h_1(y), ..., h_m(y)]         document, first m token embeddings
        score(x,y) = max_{j=1..m} <f^(1)(x), f^(m)_j(y)>

    The asymmetry is the point of the method: because the score is a maximum of
    inner products, retrieval can be done with standard MIPS by adding m entries
    per document to the index. Making the query multi-vector as well, or summing
    over document vectors instead of taking the maximum, destroys that property
    and changes the model.

    Section 6 uses m = 8; Section 7 reports m = 3 (passage) and m = 4 (document)
    on MS MARCO. ME-BERT-k applies a 768 x k feed-forward down-projection
    followed by layer normalisation; set `proj_dim` for those variants.

    Note on long documents: the m document vectors are the first m *token*
    positions, so on a newswire collection they cover only the opening of the
    article. That is faithful to the paper, but the paper's experiments are on
    MS MARCO passages, and it is a genuine reason the method can underperform
    here. Report it rather than treating a low score as a bug.

    REPRODUCTION CHOICES exposed here:
      shared_encoder   the paper says only "we encode queries and documents
                       using BERT-base" and does not state whether the towers
                       are tied. Default True.
      tie_projections  defaults to follow `shared_encoder`, so a tied encoder
                       does not silently get two different projections.
      logit_scale      see _LogitScale. Default False.
    """

    ENCODING = 'dual'
    LOSS = 'bce'
    SUPPORTS_INBATCH = True

    def __init__(
            self,
            pretrained: str,
            m: int = 8,
            proj_dim: Optional[int] = None,
            shared_encoder: bool = True,
            tie_projections: Optional[bool] = None,
            logit_scale: bool = False,
            validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(m, int) or m < 1:
            raise ValueError(f"m must be a positive integer, received {m!r}")
        if proj_dim is not None and (not isinstance(proj_dim, int) or proj_dim < 1):
            raise ValueError(f"proj_dim must be a positive integer or None, received {proj_dim!r}")

        self.pretrained = pretrained
        self.m = m
        self.shared_encoder = shared_encoder
        self.validate_inputs = validate_inputs
        self.tie_projections = shared_encoder if tie_projections is None else tie_projections

        self.config = AutoConfig.from_pretrained(pretrained)
        self.query_encoder = AutoModel.from_pretrained(pretrained, config=self.config)
        self.doc_encoder = (
            self.query_encoder if shared_encoder
            else AutoModel.from_pretrained(pretrained, config=self.config)
        )

        hidden = self.config.hidden_size
        if proj_dim is not None:
            self.query_proj = _make_projection(hidden, proj_dim)
            self.doc_proj = self.query_proj if self.tie_projections else _make_projection(hidden, proj_dim)
        else:
            self.query_proj = None
            self.doc_proj = None

        self.scale = _LogitScale() if logit_scale else None

    # -- encoding ----------------------------------------------------------

    def encode_query(
            self,
            query_input_ids: torch.Tensor,
            query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One vector per query, the sequence-start representation. -> [b, d]"""
        if self.validate_inputs:
            _validate_attention_mask(query_attention_mask, 'query_attention_mask')
        h = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        ).last_hidden_state
        q = h[:, 0, :]
        return self.query_proj(q) if self.query_proj is not None else q

    def encode_doc(
            self,
            doc_input_ids: torch.Tensor,
            doc_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The first m token representations, with the matching mask so that a
        padding slot in a short document cannot win the maximum.

        -> ([b, m', d], [b, m'])  where m' = min(m, sequence length)
        """
        if self.validate_inputs:
            _validate_attention_mask(doc_attention_mask, 'doc_attention_mask')
        h = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        ).last_hidden_state
        d = h[:, :self.m, :]
        mask = doc_attention_mask[:, :self.m]
        return (self.doc_proj(d) if self.doc_proj is not None else d), mask

    # -- scoring -----------------------------------------------------------

    def score_pairs(
            self,
            q_vec: torch.Tensor,
            d_vecs: torch.Tensor,
            d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_vec  : [b, d]
        d_vecs : [b, m, d]
        d_mask : [b, m]
        -> [b]

        'bd,bmd->bm' contracts ONLY the embedding dimension, leaving one score
        per document vector. Contracting m as well would sum the document
        vectors together and remove the multi-vector behaviour entirely.
        """
        scores = torch.einsum('bd,bmd->bm', q_vec, d_vecs)
        scores = _mask_fill(scores, d_mask.bool())
        out = scores.max(dim=1).values
        return self.scale(out) if self.scale is not None else out

    def score_matrix(
            self,
            q_vec: torch.Tensor,
            d_vecs: torch.Tensor,
            d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Every query against every document, for in-batch negatives.

        q_vec  : [Q, d]
        d_vecs : [N, m, d]
        d_mask : [N, m]
        -> [Q, N]
        """
        scores = torch.einsum('qd,nmd->qnm', q_vec, d_vecs)
        scores = _mask_fill(scores, d_mask.bool().unsqueeze(0).expand_as(scores))
        out = scores.max(dim=-1).values
        return self.scale(out) if self.scale is not None else out

    def forward(
            self,
            query_input_ids: torch.Tensor,
            query_attention_mask: torch.Tensor,
            doc_input_ids: torch.Tensor,
            doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_vec = self.encode_query(query_input_ids, query_attention_mask)
        d_vecs, d_mask = self.encode_doc(doc_input_ids, doc_attention_mask)
        return self.score_pairs(q_vec, d_vecs, d_mask)

    # -- contract ----------------------------------------------------------

    def score(self, batch: Dict[str, Any], device) -> torch.Tensor:
        """Inference score. Identical to forward for this model."""
        return self.forward(*self._batch_to(
            batch, device,
            'query_input_ids', 'query_attention_mask',
            'doc_input_ids', 'doc_attention_mask'))

    def config_dict(self) -> Dict[str, Any]:
        return {
            'model': 'me-bert',
            'pretrained': self.pretrained,
            'me_bert_m': self.m,
            'me_bert_proj_dim': (None if self.query_proj is None
                                 else self.query_proj[0].out_features),
            'shared_encoder': self.shared_encoder,
            'tie_projections': self.tie_projections,
            'logit_scale': self.scale is not None,
        }


# ---------------------------------------------------------------------------
# Poly-encoder
# ---------------------------------------------------------------------------

class PolyEncoder(BaselineRanker):
    r"""
    Poly-encoder (Humeau et al., ICLR 2020, Section 4.4).

    In the paper's dialogue setting the *context* is the long input and the
    *candidate* is the short item being retrieved. For document ranking the
    roles follow the computational constraint rather than length: the candidate
    must be a single cacheable vector, so

        candidate  ->  document   (one vector)
        context    ->  query      (m vectors)

    which is also how Luan et al. describe this architecture -- "computes a
    fixed number of vectors per query, and aggregates them by softmax attention
    against document vectors".

    Given query token representations h_1..h_N and m learned codes c_1..c_m:

        y_ctxt^i = sum_j w_j h_j,  (w_1..w_N) = softmax(c_i . h_1, ..., c_i . h_N)

    The document vector then attends over those m vectors:

        y_ctxt = sum_i w_i y_ctxt^i,
                 (w_1..w_m) = softmax(y_cand . y_ctxt^1, ..., y_cand . y_ctxt^m)

    and the score is y_ctxt . y_cand.

    Two details from Section 4.2, which Section 4.4 inherits:
      - two separate transformers, "initially start with the same weights, but
        are allowed to update separately during fine-tuning";
      - the candidate is reduced with red(.) = first output, i.e. [CLS].

    Padding masking is not discussed in the paper but is required: without it
    the first softmax distributes attention over padding positions, and with
    padding to a fixed length that is most of the sequence.

    m is 16, 64 or 360 in the paper's experiments.

    REPRODUCTION CHOICES: code initialisation std (the paper says only
    "randomly initialized"), shared_encoder, logit_scale.
    """

    ENCODING = 'dual'
    LOSS = 'bce'
    SUPPORTS_INBATCH = True

    def __init__(
            self,
            pretrained: str,
            poly_m: int = 16,
            shared_encoder: bool = False,
            code_init_std: Optional[float] = None,
            logit_scale: bool = False,
            validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(poly_m, int) or poly_m < 1:
            raise ValueError(f"poly_m must be a positive integer, received {poly_m!r}")

        self.pretrained = pretrained
        self.poly_m = poly_m
        self.shared_encoder = shared_encoder
        self.validate_inputs = validate_inputs

        self.config = AutoConfig.from_pretrained(pretrained)
        self.query_encoder = AutoModel.from_pretrained(pretrained, config=self.config)
        self.doc_encoder = (
            self.query_encoder if shared_encoder
            # "two separate transformers ... initially start with the same weights"
            else AutoModel.from_pretrained(pretrained, config=self.config)
        )

        hidden = self.config.hidden_size
        self.poly_code_embeddings = nn.Embedding(poly_m, hidden)
        # "The m context codes are randomly initialized, and learnt during finetuning."
        # The paper gives no distribution; hidden ** -0.5 is a reproduction choice.
        self.code_init_std = hidden ** -0.5 if code_init_std is None else code_init_std
        nn.init.normal_(self.poly_code_embeddings.weight, std=self.code_init_std)

        self.scale = _LogitScale() if logit_scale else None

    # -- encoding ----------------------------------------------------------

    def encode_query(
            self,
            query_input_ids: torch.Tensor,
            query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The m context vectors, one per learned code. -> [b, m, d]"""
        if self.validate_inputs:
            _validate_attention_mask(query_attention_mask, 'query_attention_mask')
        h = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        ).last_hidden_state                                          # [b, Lq, d]

        code_ids = torch.arange(self.poly_m, device=h.device)
        codes = self.poly_code_embeddings(code_ids)                  # [m, d]

        code_scores = torch.einsum('md,bnd->bmn', codes, h)          # [b, m, Lq]
        q_mask = query_attention_mask.unsqueeze(1).expand(-1, self.poly_m, -1)
        code_probs = _masked_softmax(code_scores, q_mask, dim=-1)

        return torch.einsum('bmn,bnd->bmd', code_probs, h)           # [b, m, d]

    def encode_doc(
            self,
            doc_input_ids: torch.Tensor,
            doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The candidate vector, red(.) = first output. -> [b, d]"""
        if self.validate_inputs:
            _validate_attention_mask(doc_attention_mask, 'doc_attention_mask')
        h = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        ).last_hidden_state
        return h[:, 0, :]

    # -- scoring -----------------------------------------------------------

    def score_pairs(self, y_ctxt: torch.Tensor, y_cand: torch.Tensor) -> torch.Tensor:
        """
        y_ctxt : [b, m, d]
        y_cand : [b, d]
        -> [b]
        """
        cand_scores = torch.einsum('bd,bmd->bm', y_cand, y_ctxt)     # [b, m]
        cand_probs = torch.softmax(cand_scores, dim=-1)
        out = (cand_probs * cand_scores).sum(dim=-1)
        return self.scale(out) if self.scale is not None else out

    def score_matrix(self, y_ctxt: torch.Tensor, y_cand: torch.Tensor) -> torch.Tensor:
        """
        Every query against every document, for in-batch negatives.

        y_ctxt : [Q, m, d]
        y_cand : [N, d]
        -> [Q, N]

        Uses score = sum_i softmax_i(y_cand . y_ctxt^i) * (y_cand . y_ctxt^i),
        which is algebraically identical to forming the candidate-conditioned
        context vector and then taking the final dot product, but avoids
        materialising a [Q, N, d] tensor.
        """
        inter = torch.einsum('nd,qmd->qnm', y_cand, y_ctxt)          # [Q, N, m]
        probs = torch.softmax(inter, dim=-1)
        out = (probs * inter).sum(dim=-1)                            # [Q, N]
        return self.scale(out) if self.scale is not None else out

    def forward(
            self,
            query_input_ids: torch.Tensor,
            query_attention_mask: torch.Tensor,
            doc_input_ids: torch.Tensor,
            doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        y_ctxt = self.encode_query(query_input_ids, query_attention_mask)
        y_cand = self.encode_doc(doc_input_ids, doc_attention_mask)
        return self.score_pairs(y_ctxt, y_cand)

    # -- contract ----------------------------------------------------------

    def score(self, batch: Dict[str, Any], device) -> torch.Tensor:
        """Inference score. Identical to forward for this model."""
        return self.forward(*self._batch_to(
            batch, device,
            'query_input_ids', 'query_attention_mask',
            'doc_input_ids', 'doc_attention_mask'))

    def config_dict(self) -> Dict[str, Any]:
        return {
            'model': 'poly-encoder',
            'pretrained': self.pretrained,
            'poly_m': self.poly_m,
            'shared_encoder': self.shared_encoder,
            'code_init_std': self.code_init_std,
            'logit_scale': self.scale is not None,
        }