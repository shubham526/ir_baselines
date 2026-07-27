"""
The contract every baseline ranker implements.

The two families in this repository differ in how a (query, document) pair
reaches the model and in what the score means, and nothing else. Rather than
branch on the model name in the dataset, the trainer and the evaluator, each
model declares those two things and the rest of the pipeline reads them.

    ENCODING   'pair'  the query and document are tokenized together as one
                       sequence, giving input_ids / attention_mask /
                       token_type_ids.
               'dual'  they are tokenized separately, giving
                       query_input_ids / query_attention_mask and
                       doc_input_ids / doc_attention_mask.

    LOSS       'cross_entropy'  the model emits [B, 2] logits and the label is
                                a class index.
               'bce'            the model emits [B] logits and the label is a
                                float in {0, 1}.

    score(batch, device) -> [B]
               The value written to the run file. For a two-way classifier
               that is the softmax probability of the relevant class, not the
               raw logit; for a single-score model it is the score itself.
               Keeping this in the model is what lets one evaluator serve
               both families.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaselineRanker(nn.Module, ABC):
    ENCODING: str = 'pair'
    LOSS: str = 'cross_entropy'

    @abstractmethod
    def forward(self, **batch):
        """Training-time forward. Returns whatever LOSS expects."""

    @abstractmethod
    def score(self, batch, device):
        """Inference-time scoring. Returns a [B] tensor for ranking."""

    # -- helpers shared by both families ------------------------------------

    @staticmethod
    def _to(batch, device, *keys):
        return [batch[k].to(device) for k in keys]


def masked_mean(last_hidden_state, attention_mask):
    """
    Mean over non-padding positions.

    Not the pooling used for the published runs -- see CrossEncoder's
    t5_pooling argument -- but available for comparison.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def mean_all(last_hidden_state, attention_mask=None):
    """
    Mean over every position, padding included.

    This is the pooling that produced the published T5 runs. On a document of
    185 tokens padded to 512, roughly two thirds of the averaged positions are
    padding, so the two poolings are not interchangeable and the choice must
    travel with the checkpoint.
    """
    return torch.mean(last_hidden_state, dim=1)
