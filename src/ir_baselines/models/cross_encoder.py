"""
Cross-encoder re-ranking baseline.

One encoder over the concatenated (query, document) sequence, with a two-way
classifier on the pooled representation. This single class covers seven rows
of the published tables -- BERT, RoBERTa, DeBERTa, ELECTRA, ConvBERT, RankT5
and ERNIE -- which differ only in which pretrained encoder they wrap.

Because the query and the document are encoded together there is no separate
query representation to cache, so in-batch negative training is not available:
SUPPORTS_INBATCH is False and the objective is always cross-entropy over the
two-way classifier.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel

from .base import BaselineRanker, masked_mean, mean_all

T5_POOLING_CHOICES = ('mean-all', 'masked-mean')


class CrossEncoder(BaselineRanker):
    """
    Pooling depends on the encoder family:

        BERT and family   the sequence-start position of the last hidden state
        DistilBERT        the same; the model takes no token_type_ids
        T5                the mean of the last hidden state, see below

    T5 POOLING. T5 has no sequence-start representation to pool, so the hidden
    states are averaged. Two behaviours are available:

        'mean-all'     average over every position, padding included. This is
                       what produced the published T5 runs, and the default.
        'masked-mean'  average over non-padding positions only.

    The difference is not cosmetic: on a 185-token document padded to 512,
    roughly two thirds of the averaged positions are padding. The setting is
    recorded in config_dict(), stored in the checkpoint, and verified at
    inference, because a checkpoint trained under one pooling and evaluated
    under the other loads with every key matched and produces different scores
    throughout.
    """

    ENCODING = 'pair'
    LOSS = 'cross-entropy'
    SUPPORTS_INBATCH = False

    def __init__(
            self,
            pretrained: str,
            t5_pooling: str = 'mean-all',
    ) -> None:
        super().__init__()
        if t5_pooling not in T5_POOLING_CHOICES:
            raise ValueError(
                f't5_pooling must be one of {T5_POOLING_CHOICES}, received {t5_pooling!r}'
            )

        self.pretrained = pretrained
        self.t5_pooling = t5_pooling
        self.config = AutoConfig.from_pretrained(pretrained)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        # T5EncoderModel rather than AutoModel: AutoModel would build the full
        # encoder-decoder and the decoder weights would be unused.
        #
        # Dispatch on config.model_type rather than on the checkpoint name. A
        # substring test for 't5' would misfire on any unrelated model whose
        # name happens to contain it, and would miss a T5 checkpoint under a
        # name that does not.
        self.is_t5 = self.config.model_type == 't5'
        if self.is_t5:
            self.encoder = T5EncoderModel.from_pretrained(pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(pretrained, config=self.config)

    # -- pooling -----------------------------------------------------------

    def _pool(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        if isinstance(self.encoder, DistilBertModel):
            # DistilBERT takes no token_type_ids.
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :]

        if isinstance(self.encoder, T5EncoderModel):
            h = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            if self.t5_pooling == 'masked-mean':
                return masked_mean(h, attention_mask)
            return mean_all(h)

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.last_hidden_state[:, 0, :]

    # -- contract ----------------------------------------------------------

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """-> [B, 2] logits, for cross-entropy against a class index."""
        pooled = self._pool(input_ids, attention_mask, token_type_ids)
        return self.classifier(pooled)

    def score(self, batch: Dict[str, Any], device) -> torch.Tensor:
        """
        -> [B], the probability of the relevant class.

        The run file records the softmax probability rather than the raw
        logit. Ranking is unchanged by the transform for a single query, but
        the written scores are what a reader sees, and they were produced this
        way.
        """
        logits = self.forward(*self._batch_to(
            batch, device, 'input_ids', 'attention_mask', 'token_type_ids'))
        return logits.softmax(dim=-1)[:, 1]

    def config_dict(self) -> Dict[str, Any]:
        return {
            'model': 'cross-encoder',
            'pretrained': self.pretrained,
            't5_pooling': self.t5_pooling,
        }