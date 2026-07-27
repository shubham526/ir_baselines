"""
Cross-encoder baselines.

One BERT-family encoder over the concatenated (query, document) sequence,
with a two-way classifier on the pooled representation. The published rows for
BERT, RoBERTa, DeBERTa, ELECTRA, ConvBERT, RankT5, KNRM, EDRM and ERNIE were
produced by this class; the encoder is chosen with --pretrain.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel

from .base import BaselineRanker, mean_all, masked_mean


class CrossEncoder(BaselineRanker):
    ENCODING = 'pair'
    LOSS = 'cross_entropy'

    def __init__(self, pretrained: str, t5_pooling: str = 'mean-all'):
        """
        t5_pooling
            'mean-all'     mean over all positions including padding. This is
                           what produced the published runs and is the
                           default.
            'masked-mean'  mean over non-padding positions only.

        The choice changes every score a T5 model produces, so it is stored in
        the checkpoint and checked at inference time. It has no effect on any
        other encoder, which pool by taking the sequence-start position.
        """
        super().__init__()
        if t5_pooling not in ('mean-all', 'masked-mean'):
            raise ValueError(f"t5_pooling must be 'mean-all' or 'masked-mean', got {t5_pooling!r}")
        self.pretrained = pretrained
        self.t5_pooling = t5_pooling
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if isinstance(self.encoder, DistilBertModel):
            # DistilBERT takes no token_type_ids.
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = output.last_hidden_state[:, 0, :]
        elif isinstance(self.encoder, T5EncoderModel):
            # T5 has no sequence-start representation to pool, so the hidden
            # states are averaged. See t5_pooling above.
            last_hidden_state = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.t5_pooling == 'masked-mean':
                pooled = masked_mean(last_hidden_state, attention_mask)
            else:
                pooled = mean_all(last_hidden_state)
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
            pooled = output.last_hidden_state[:, 0, :]

        score = self.classifier(pooled).squeeze(-1)
        return score, pooled

    def score(self, batch, device):
        logits, _ = self.forward(
            *self._to(batch, device, 'input_ids', 'attention_mask', 'token_type_ids'))
        # The run file records the probability of the relevant class, not the
        # raw logit.
        return logits.softmax(dim=-1)[:, 1].squeeze(-1)

    def config_dict(self):
        return {'pretrained': self.pretrained, 't5_pooling': self.t5_pooling}
