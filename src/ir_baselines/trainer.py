from typing import Optional

import torch
import torch.nn.functional as F
import tqdm

from .data.dataset import TENSOR_KEYS

LOSS_CHOICES = ('cross-entropy', 'bce', 'ce-inbatch')


def inbatch_scores(model, batch, device) -> torch.Tensor:
    """
    Score every query in the batch against every document in the batch.

    Works for any model exposing the encode/score interface; MEBERT.encode_doc
    returns (vectors, mask) while PolyEncoder.encode_doc returns a single
    tensor, so the tuple is unpacked when present.

    -> [B, B], with the diagonal holding the aligned (query i, document i) pairs.
    """
    if not model.SUPPORTS_INBATCH:
        raise TypeError(
            f'{type(model).__name__} encodes the query and document together, '
            f'so there are no separate representations to cross-score and '
            f'in-batch negatives are not available. Use --loss bce or the '
            f"model's own default."
        )
    q = model.encode_query(
        batch['query_input_ids'].to(device, non_blocking=True),
        batch['query_attention_mask'].to(device, non_blocking=True),
    )
    d = model.encode_doc(
        batch['doc_input_ids'].to(device, non_blocking=True),
        batch['doc_attention_mask'].to(device, non_blocking=True),
    )
    return model.score_matrix(q, *d) if isinstance(d, tuple) else model.score_matrix(q, d)


class Trainer:
    """
    loss = 'cross-entropy'
        Pointwise softmax cross-entropy over a two-way classifier. The
        objective for cross-encoders, which have no separate query and
        document representations.

    loss = 'bce'
        Pointwise BCEWithLogitsLoss on the aligned pair score. This is the
        controlled protocol used by the other re-ranking baselines in the
        comparison, so it keeps the rows on the same footing. Raw inner
        products of 768-dimensional BERT states have a standard deviation
        around 14, which puts roughly half of all pairs in the saturated region
        of the sigmoid; enable the model's `logit_scale` with this objective.

    loss = 'ce-inbatch'
        Softmax cross-entropy over the other documents in the batch. This is
        inspired by the multi-vector papers' objectives but is NOT a
        reproduction of either negative-sampling recipe: ME-BERT combines
        sampled candidates from a precomputed list with in-batch negatives and
        optional hard-negative mining, and Poly-encoder inherits the
        Bi-encoder setup. Requires positives-only training data.

        Where `query_id` is present in the batch, off-diagonal entries for the
        same query are masked out, so a second positive for the same query in
        the batch is not scored as a negative.
    """

    def __init__(self, model, optimizer, criterion, scheduler, metric, data_loader, device,
                 loss: str = 'bce',
                 amp_dtype: Optional[torch.dtype] = None,
                 max_grad_norm: Optional[float] = 1.0,
                 log_every: int = 0):
        if loss not in LOSS_CHOICES:
            raise ValueError(f'loss must be one of {LOSS_CHOICES}, got {loss!r}')
        if loss == 'cross-entropy' and model.ENCODING != 'pair':
            raise ValueError(
                f"loss 'cross-entropy' expects a two-way classifier over a single "
                f'sequence; {type(model).__name__} encodes the query and document '
                f"separately. Use 'bce' or 'ce-inbatch'."
            )
        if loss in ('bce', 'ce-inbatch') and model.ENCODING == 'pair':
            raise ValueError(
                f'{type(model).__name__} emits [B, 2] logits, which {loss!r} cannot '
                f"consume. Use 'cross-entropy'."
            )
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._metric = metric
        self._data_loader = data_loader
        self._device = device
        self._loss = loss
        self._amp_dtype = amp_dtype
        self._max_grad_norm = max_grad_norm
        self._log_every = log_every
        self._warned_no_qid = False
        # GradScaler is needed for fp16 only; bf16 has fp32's exponent range.
        self._scaler = torch.amp.GradScaler(
            'cuda', enabled=(amp_dtype == torch.float16 and device.type == 'cuda')
        )

    def _forward(self, batch) -> torch.Tensor:
        """Call the model with whatever keys its encoding produces."""
        keys = TENSOR_KEYS[self._model.ENCODING]
        return self._model(*[batch[k].to(self._device, non_blocking=True) for k in keys])

    def _pointwise_loss(self, batch) -> torch.Tensor:
        out = self._forward(batch)
        label = batch['label'].to(self._device, non_blocking=True)
        if self._loss == 'cross-entropy':
            # CrossEntropyLoss takes class indices, so the label is an integer.
            return self._criterion(out.float(), label.long())
        return self._criterion(out.float(), label.float())

    def _ce_loss(self, batch) -> torch.Tensor:
        scores = inbatch_scores(self._model, batch, self._device)
        qids = batch.get('query_id')
        if qids is not None:
            same = torch.tensor(
                [[a == b for b in qids] for a in qids], device=scores.device
            )
            same.fill_diagonal_(False)
            if same.any():
                scores = scores.masked_fill(same, torch.finfo(scores.dtype).min)
        elif not self._warned_no_qid:
            print('WARNING  training data carries no query_id, so a second positive '
                  'for the same query inside a batch is scored as a negative.')
            self._warned_no_qid = True
        target = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores.float(), target)

    def _step(self, batch) -> Optional[float]:
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        # Cross-entropy over in-batch negatives needs at least one negative.
        # A trailing singleton batch is skipped rather than handled by
        # drop_last, which would discard up to batch_size-1 positives per epoch.
        if self._loss == 'ce-inbatch' and batch['label'].size(0) < 2:
            return None

        autocast = torch.amp.autocast(
            device_type=self._device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_dtype is not None and self._device.type == 'cuda',
        )
        with autocast:
            if self._loss == 'ce-inbatch':
                loss = self._ce_loss(batch)
            else:
                loss = self._pointwise_loss(batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'Non-finite training loss ({loss.item()}). Continuing would '
                f'propagate NaN through every parameter on the next backward pass.'
            )

        self._scaler.scale(loss).backward()
        if self._max_grad_norm is not None:
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        return loss.item()

    def train(self) -> float:
        """
        Returns the MEAN batch loss, which is comparable across epochs and
        across folds of different sizes. A summed loss is not.
        """
        total, n, skipped = 0.0, 0, 0
        bar = tqdm.tqdm(self._data_loader, total=len(self._data_loader), desc='train')
        for i, batch in enumerate(bar):
            batch_loss = self._step(batch)
            if batch_loss is None:
                skipped += 1
                continue
            total += batch_loss
            n += 1
            if self._log_every and (i + 1) % self._log_every == 0:
                bar.set_postfix(loss=f'{total / n:.4f}')
        if skipped:
            print(f'NOTE  skipped {skipped} batch(es) too small for in-batch negatives.')
        return total / max(n, 1)