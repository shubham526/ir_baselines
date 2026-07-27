"""
The contract every ranker implements, and the tensor helpers they share.

Two families of model live in this package and they differ in exactly two
ways: how a (query, document) pair reaches the model, and what the score
means. Rather than branch on the model name in the dataset, the trainer and
the evaluator, each model declares both.

    ENCODING          'pair'  query and document tokenized together as one
                              sequence -> input_ids, attention_mask,
                              token_type_ids
                      'dual'  tokenized separately -> query_input_ids,
                              query_attention_mask, doc_input_ids,
                              doc_attention_mask

    LOSS              the objective the model's forward() output expects.
                      'cross-entropy'  forward returns [B, 2] logits, the
                                       label is a class index
                      'bce'            forward returns [B] logits, the label
                                       is a float in {0, 1}

    SUPPORTS_INBATCH  whether encode_query / encode_doc / score_matrix exist,
                      i.e. whether in-batch negative training is possible. A
                      cross-encoder has no separate query and document
                      representations, so it cannot do this.

    forward(...)      training-time output, matching LOSS
    score(batch, ...) inference-time value written to the run file. For a
                      two-way classifier that is the probability of the
                      relevant class, not the raw logit; for a single-score
                      model it is the score itself. Keeping the transform in
                      the model is what lets one evaluator serve both
                      families.
    config_dict()     the settings that must match between training and
                      inference, stored in the checkpoint and verified on load
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# tensor helpers
# ---------------------------------------------------------------------------

def _mask_fill(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """
    Set masked positions to the most negative value representable in the
    tensor's own dtype.

    torch.finfo(torch.float32).min overflows float16 and bfloat16 and raises at
    masked_fill time, which breaks the mixed-precision training both papers use.
    Selecting the value from `scores.dtype` is safe in every dtype and, unlike
    -inf, cannot produce NaN if a row turns out to be fully masked.
    """
    return scores.masked_fill(~keep, torch.finfo(scores.dtype).min)


def _validate_attention_mask(attention_mask: torch.Tensor, name: str) -> None:
    """
    Reject fully padded sequences.

    A correctly tokenised input always contains at least the sequence-start
    token, so this should never fire. It matters because the failure is silent
    otherwise: an all-padding ME-BERT document makes every slot ineligible for
    the maximum, and `max` then returns the sentinel itself -- a large negative
    score that ranks last and looks like a confident prediction rather than a
    malformed batch.

    Note that this forces a host-device synchronisation, so it serialises the
    step boundary during training. Pass `validate_inputs=False` to the model
    once a configuration has been smoke-tested if that cost shows up in
    profiling.
    """
    keep = attention_mask.bool()
    if keep.ndim != 2:
        raise ValueError(
            f'{name} must have shape [batch, length], received {tuple(keep.shape)}'
        )
    empty = ~keep.any(dim=-1)
    if bool(empty.any()):
        idx = empty.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f'{name} contains fully masked sequences at batch indices {idx}.')


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax that ignores padding positions.

    scores : [..., N]
    mask   : [..., N], 1 for real tokens, 0 for padding
    """
    keep = mask.bool()
    if (~keep).all(dim=dim).any():
        raise ValueError(
            "masked_softmax received a fully masked row. A transformer input "
            "always contains at least the sequence-start token, so this "
            "indicates malformed batching rather than a short sequence."
        )
    return torch.softmax(_mask_fill(scores, keep), dim=dim)


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding positions."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def mean_all(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Mean over every position, padding included.

    This is the pooling that produced the published T5 runs. On a document of
    185 tokens padded to 512, roughly two thirds of the averaged positions are
    padding, so the two poolings are not interchangeable and the choice must
    travel with the checkpoint.
    """
    return torch.mean(last_hidden_state, dim=1)


class _LogitScale(nn.Module):
    """
    Affine calibration of a raw inner-product score, y = a * s + b.

    Both multi-vector papers train with softmax cross-entropy over competing
    candidates, which is invariant to b and only mildly sensitive to a. A
    pointwise objective such as BCEWithLogitsLoss is not: unnormalised dot
    products of 768-dimensional BERT states have standard deviation of roughly
    10-15, so close to half of all pairs land in the saturated region of the
    sigmoid and receive gradients on the order of 1e-4. If a pointwise or
    pairwise loss is used, this layer lets the model recover a usable operating
    point.

    REPRODUCTION CHOICE. Not in either paper. Disabled by default, since with
    the papers' own objective it is unnecessary.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.a * scores + self.b


def _make_projection(hidden: int, projected: int) -> nn.Sequential:
    """
    REPRODUCTION CHOICE.

    Luan et al. describe the feed-forward-plus-layer-normalisation construction
    while introducing the lower-dimensional DE-BERT-k variant. The ME-BERT-k
    paragraph specifies a 768 x k feed-forward projection but does not restate
    whether the layer normalisation carries over. The same block is used here
    for ME-BERT-k on the assumption that it does.

    The bias term is also unspecified; `bias=False` is a choice, not the paper.

    Neither applies to ME-BERT-768, i.e. `proj_dim=None`, which is the default
    and is unaffected by this decision.
    """
    return nn.Sequential(
        nn.Linear(hidden, projected, bias=False),
        nn.LayerNorm(projected),
    )


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------

class BaselineRanker(nn.Module, ABC):
    ENCODING: str = 'pair'
    LOSS: str = 'cross-entropy'
    SUPPORTS_INBATCH: bool = False

    @abstractmethod
    def forward(self, **batch) -> torch.Tensor:
        """Training-time output, matching LOSS."""

    @abstractmethod
    def score(self, batch: Dict[str, Any], device) -> torch.Tensor:
        """Inference-time score, [B], as written to the run file."""

    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        """
        Settings that must match between training and inference.

        Stored inside the checkpoint. A mismatch in any of these means the
        weights do not mean what the architecture expects, and
        load_state_dict will not say so.
        """

    # -- helpers -----------------------------------------------------------

    def _batch_to(self, batch: Dict[str, Any], device, *keys):
        return [batch[k].to(device, non_blocking=True) for k in keys]

    def resize_if_needed(self, n_tokens: int) -> None:
        """
        Grow every encoder's embedding matrix if tokens were added to the
        vocabulary. Without this, a new token id indexes out of range at the
        first forward pass.
        """
        for attr in ('query_encoder', 'doc_encoder', 'encoder'):
            enc = getattr(self, attr, None)
            if enc is None or not hasattr(enc, 'resize_token_embeddings'):
                continue
            if enc.get_input_embeddings().weight.size(0) != n_tokens:
                enc.resize_token_embeddings(n_tokens)