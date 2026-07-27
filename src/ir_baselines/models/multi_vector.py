"""
Multi-vector baselines.

ME-BERT   Luan, Eisenstein, Toutanova & Collins. TACL 2021.
PolyEnc   Humeau, Shuster, Lachaux & Weston. ICLR 2020.
ColBERT   Khattab & Zaharia. SIGIR 2020.

The query and the document are encoded separately, so these use ENCODING
'dual'. Each produces a single score per pair and is trained with binary
cross-entropy.

NOTE ON THIS VERSION. The ME-BERT and Poly-encoder scoring functions below do
not compute the expressions given in their papers:

  - PolyEncoder contracts both the code index and the sequence index in the
    same einsum, so the softmax sums to one across the whole tensor and the
    learned context codes have no effect on the score.
  - MEBERT represents the query with m vectors rather than one, and takes the
    maximum over query positions rather than over the m document vectors, so
    the multi-vector maximum never happens.

They are kept as they are in this tag because they are what produced the
published rows. tests/test_scoring.py checks both against explicit
implementations of the published expressions and fails here by design. The
revised implementations are in the next release; see docs/known-issues.md.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .base import BaselineRanker


class PolyEncoder(BaselineRanker):
    ENCODING = 'dual'
    LOSS = 'bce'

    def __init__(self, pretrained, poly_m=16):
        super().__init__()
        self.pretrained = pretrained
        self.poly_m = poly_m
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)
        self.poly_code_embeddings = nn.Embedding(self.poly_m, self.config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, self.config.hidden_size ** -0.5)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        doc_outputs = self.encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)[0]

        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=doc_input_ids.device)
        poly_codes = self.poly_code_embeddings(poly_code_ids)          # [m, d]

        query_outputs = self.encoder(input_ids=query_input_ids,
                                     attention_mask=query_attention_mask)[0]
        query_rep = query_outputs[:, 0, :]                             # [b, d]

        attn_scores = torch.einsum('nd,bd->bn', poly_codes, query_rep)  # [b, m]
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=1)

        # 'bn,bmd->bd' contracts n and m together, so this is an unweighted
        # sum over document positions. See the note at the top of this file.
        doc_embs = torch.einsum('bn,bmd->bd', attn_probs, doc_outputs)  # [b, d]

        return torch.einsum('bd,bd->b', doc_embs, query_rep)

    def score(self, batch, device):
        return self.forward(*self._to(
            batch, device,
            'query_input_ids', 'query_attention_mask',
            'doc_input_ids', 'doc_attention_mask'))

    def config_dict(self):
        return {'pretrained': self.pretrained, 'poly_m': self.poly_m}


class MEBERT(BaselineRanker):
    ENCODING = 'dual'
    LOSS = 'bce'

    def __init__(self, pretrained, m=8):
        super().__init__()
        self.pretrained = pretrained
        self.m = m
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        query_outputs = self.encoder(query_input_ids, attention_mask=query_attention_mask)
        doc_outputs = self.encoder(doc_input_ids, attention_mask=doc_attention_mask)

        # The paper gives the query one vector and the document m; this takes
        # the first m of both. See the note at the top of this file.
        query_repr = query_outputs.last_hidden_state[:, :self.m]
        doc_repr = doc_outputs.last_hidden_state[:, :self.m]

        scores = torch.einsum('bmd,bnd->bm', query_repr, doc_repr)
        return scores.max(dim=1).values

    def score(self, batch, device):
        return self.forward(*self._to(
            batch, device,
            'query_input_ids', 'query_attention_mask',
            'doc_input_ids', 'doc_attention_mask'))

    def config_dict(self):
        return {'pretrained': self.pretrained, 'me_bert_m': self.m}
