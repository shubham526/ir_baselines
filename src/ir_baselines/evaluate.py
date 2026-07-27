from typing import Optional

import torch
import tqdm


def evaluate(model, data_loader, device, amp_dtype: Optional[torch.dtype] = None):
    """
    Returns {query_id: {doc_id: [score, label]}}.

    The score transform belongs to the model: a two-way classifier writes the
    probability of the relevant class, a single-score model writes its score.
    This function only calls `model.score(batch, device)`, so it serves both
    families without knowing which one it has.

    `amp_dtype` defaults to None, i.e. full precision. Casting the result to
    float32 afterwards does NOT recover precision lost inside the dot products,
    the softmax or the maximum -- it only stores a quantised value in a wider
    container. Reduced-precision inference is available for quick checks, but
    any run that goes in a table should be produced in float32.
    """
    if amp_dtype is not None:
        print('WARNING  reduced-precision inference. Scores are quantised at '
              'computation time and casting to float32 afterwards does not '
              'recover that precision. Use full precision for reported runs.')

    rst_dict = {}
    model.eval()

    autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None and device.type == 'cuda',
    )

    with torch.no_grad():
        for dev_batch in tqdm.tqdm(data_loader, total=len(data_loader), desc='eval'):
            query_id, doc_id, label = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label']
            with autocast:
                batch_score = model.score(dev_batch, device)

            if not torch.isfinite(batch_score).all():
                bad = (~torch.isfinite(batch_score)).nonzero().flatten().tolist()
                raise FloatingPointError(
                    f'Non-finite scores at batch positions {bad[:8]} '
                    f'(query ids {[query_id[i] for i in bad[:8]]}). A run written '
                    f'from these would rank the affected documents arbitrarily.'
                )

            batch_score = batch_score.float().detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, int(l)]

    return rst_dict