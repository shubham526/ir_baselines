import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every source of randomness in the training path.

    `deterministic=True` also disables cuDNN autotuning and requests
    deterministic kernels. That costs throughput, but a re-run that cannot be
    reproduced is worth less than the time it saves.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:                       # older torch
            torch.use_deterministic_algorithms(True)


def save_trec(
        rst_file: str,
        rst_dict: Dict[str, Dict[str, List[float]]],
        run_tag: str = 'Baseline',
        top_k: Optional[int] = None,
) -> None:
    """
    Write a run in TREC six-field format.

    Ties are broken by document id rather than left to dictionary order, so the
    same scores always produce the same ranking. Float scores are written at
    full repr precision so that re-scoring a written run reproduces the metric.
    """
    os.makedirs(os.path.dirname(os.path.abspath(rst_file)) or '.', exist_ok=True)
    with open(rst_file, 'w') as writer:
        for q_id in sorted(rst_dict):
            scores = rst_dict[q_id]
            res = sorted(scores.items(), key=lambda x: (-x[1][0], x[0]))
            if top_k is not None:
                res = res[:top_k]
            for rank, (d_id, value) in enumerate(res, start=1):
                writer.write(f'{q_id} Q0 {d_id} {rank} {value[0]!r} {run_tag}\n')


def run_topic_set(run_file: str) -> Set[str]:
    """The set of topic ids in a run file."""
    topics = set()
    with open(run_file) as f:
        for line in f:
            if line.strip():
                topics.add(line.split()[0])
    return topics


def qrels_topic_set(qrels_file: str) -> Set[str]:
    """The set of topic ids in a qrels file."""
    topics = set()
    with open(qrels_file) as f:
        for line in f:
            if line.strip():
                topics.add(line.split()[0])
    return topics


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(save_path: Optional[str], model: torch.nn.Module,
                    config: Optional[Dict[str, Any]] = None,
                    optimizer=None, scheduler=None, scaler=None,
                    epoch: Optional[int] = None,
                    best_metric: Optional[float] = None,
                    provenance: Optional[Dict[str, Any]] = None,
                    rng: Optional[Dict[str, Any]] = None,
                    history: Optional[Dict[str, Any]] = None) -> None:
    """
    Saves the weights together with everything needed to trace or continue the
    run that produced them.

        config       architecture settings, verified on load. A mismatch here
                     loads with every key matched and silently means something
                     different.
        provenance   git commit, environment versions, data digests, command
                     line. Answers "what produced this file" without which the
                     answer is guesswork.
        optimizer,   the training state. Without these, --resume restarts the
        scheduler,   optimiser moments and the learning-rate schedule, which
        scaler       is a different run from the one that was interrupted.
        rng          generator states, so a resumed run sees the data order the
                     original would have seen. The seed alone reproduces step
                     zero, not the step training stopped at.
        epoch,       where to pick up, and what to beat.
        best_metric
        history      per-epoch losses and metrics so far.

    All of it travels inside the checkpoint rather than in sibling files, so a
    checkpoint that is copied or renamed keeps its provenance.
    """
    if save_path is None:
        return
    obj: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
    }
    if provenance is not None:
        obj['provenance'] = provenance
    if optimizer is not None:
        obj['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        obj['scheduler_state_dict'] = scheduler.state_dict()
    # A disabled GradScaler still has a state dict; storing it keeps resume
    # symmetric whether or not fp16 was in use.
    if scaler is not None:
        obj['scaler_state_dict'] = scaler.state_dict()
    if epoch is not None:
        obj['epoch'] = epoch
    if best_metric is not None:
        obj['best_metric'] = best_metric
    if rng is not None:
        obj['rng_state'] = rng
    if history is not None:
        obj['history'] = history
    torch.save(obj, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path: Optional[str], model: torch.nn.Module, device,
                    strict: bool = True) -> Dict[str, Any]:
    """
    Loads the weights and returns everything else the checkpoint carries:
    config, provenance, optimizer/scheduler/scaler state, epoch, best_metric,
    rng_state and history. An older bare state_dict returns {}.

    `strict=True` by default. Note that strictness alone does NOT catch a
    tower-topology mismatch: when `shared_encoder=True` the query and document
    encoders are the same module object, so `state_dict()` still emits both key
    names and a separate-tower checkpoint loads with every key matched and no
    warning -- silently overwriting the query tower with the document tower.
    Only the configuration check catches that, which is why the config is
    stored here and verified by the caller.
    """
    if load_path is None:
        return {}
    obj = torch.load(load_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        state_dict, extras = obj['model_state_dict'], obj
    else:                                       # bare state_dict, older format
        state_dict, extras = obj, {}
        print('NOTE  checkpoint carries no config; provenance cannot be verified.')

    # Checkpoints from earlier versions of the cross-encoder name the encoder
    # attribute 't5.' where it is now 'encoder.'. The tensors and their shapes
    # are unchanged, so the keys are remapped rather than the weights
    # regenerated. Without this a strict load fails; with strict=False it would
    # leave the encoder randomly initialised and say nothing.
    if any(k.startswith('t5.') for k in state_dict):
        state_dict = {('encoder.' + k[3:] if k.startswith('t5.') else k): v
                      for k, v in state_dict.items()}
        print('NOTE  remapped legacy "t5." parameter names to "encoder.".')

    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing, unexpected = result
        if missing or unexpected:
            print(f'WARNING  missing keys: {list(missing)[:8]}')
            print(f'WARNING  unexpected keys: {list(unexpected)[:8]}')
    print(f'Model loaded from <== {load_path}')
    return extras


def check_config(stored: Dict[str, Any], current: Dict[str, Any],
                 keys: Tuple[str, ...]) -> List[Tuple[str, Any, Any]]:
    """Returns [(key, stored_value, current_value)] for every disagreement."""
    out = []
    for k in keys:
        if k not in stored or stored[k] is None or current.get(k) is None:
            continue
        if stored[k] != current[k]:
            out.append((k, stored[k], current[k]))
    return out