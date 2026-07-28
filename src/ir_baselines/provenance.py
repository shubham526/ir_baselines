"""
Provenance: what produced this artifact.

Everything here answers a question that is expensive to answer later and
cheap to record now.

    Which code?          the git commit, and whether the tree was dirty
    Which environment?   python, torch, CUDA, transformers, the GPU
    Which data?          size, line count and digest of every input file
    Which invocation?    the full command line
    When?                a UTC timestamp

A checkpoint without these is a file of weights that cannot be traced. Two
years later, "was this run before or after we changed the tokenizer" is
answerable in seconds with them and not at all without.

The RNG helpers exist for a narrower reason: resuming training without
restoring the random state gives a different data order from the run that
would have continued, so the resumed run is not the run that was
interrupted.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import torch


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git_info(path: Optional[str] = None) -> Dict[str, Any]:
    """
    The commit that produced this run, and whether the tree was modified.

    `dirty` matters more than the commit: a clean commit hash identifies the
    code exactly, a dirty one means the hash is a lower bound on what was
    actually running. Recorded rather than warned about, because working from
    a dirty tree during development is normal.

    Returns {'available': False, ...} outside a repository rather than
    raising, since the package is usable from a wheel.
    """
    path = path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(*cmd, strip: bool = True) -> Optional[str]:
        try:
            out = subprocess.run(cmd, cwd=path, capture_output=True, text=True,
                                 timeout=10, check=False)
            if out.returncode != 0:
                return None
            return out.stdout.strip() if strip else out.stdout
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run('git', 'rev-parse', 'HEAD')
    if commit is None:
        return {'available': False}

    # Not stripped: porcelain format is two status characters, a space, then
    # the path, and a modified-but-unstaged file starts with a space. Stripping
    # would remove it and shift every subsequent slice by one character.
    status = run('git', 'status', '--porcelain', strip=False)

    # 'dirty' means TRACKED files have uncommitted changes, which is what makes
    # the commit hash an unreliable identifier of what ran. Untracked files are
    # listed separately: a repository almost always has some -- __pycache__,
    # scratch outputs -- and treating those as dirty would mean the hash is
    # never usable.
    modified, untracked = [], []
    for line in (status or '').split('\n'):
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        (untracked if code == '??' else modified).append(path)

    return {
        'available': True,
        'commit': commit,
        'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
        'describe': run('git', 'describe', '--tags', '--always', '--dirty'),
        'dirty': bool(modified),
        'dirty_files': modified[:20],
        'untracked': untracked[:20],
    }


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def environment() -> Dict[str, Any]:
    """
    Versions of everything that can change a number.

    transformers is here because it has: `encode_plus` was removed in v5, and
    T5 tokenisation and attention defaults have both changed within v4. A run
    that cannot be reproduced under a different version is a run whose
    transformers version needs to be known.
    """
    env: Dict[str, Any] = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'hostname': socket.gethostname(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    try:
        import transformers
        env['transformers'] = transformers.__version__
    except Exception:                                   # pragma: no cover
        env['transformers'] = None
    try:
        import numpy
        env['numpy'] = numpy.__version__
    except Exception:                                   # pragma: no cover
        env['numpy'] = None
    if torch.cuda.is_available():
        try:
            env['gpu'] = [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
        except Exception:                               # pragma: no cover
            env['gpu'] = None
    return env


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def file_fingerprint(path: str, digest: bool = True,
                     max_digest_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Size, line count and SHA-256 of one input file.

    The digest is what distinguishes "the same file" from "a file at the same
    path". Regenerating training data with a different negative sample gives a
    file of the same size and line count and different contents, and nothing
    downstream would notice.

    `max_digest_bytes` hashes only a prefix, for files large enough that a full
    pass is not worth it. A prefix digest is recorded as such, since it cannot
    detect a change beyond that point.
    """
    if not os.path.exists(path):
        return {'path': path, 'exists': False}

    size = os.path.getsize(path)
    info: Dict[str, Any] = {
        'path': os.path.abspath(path),
        'exists': True,
        'bytes': size,
        'modified': datetime.fromtimestamp(
            os.path.getmtime(path), timezone.utc).isoformat(),
    }

    lines = 0
    h = hashlib.sha256() if digest else None
    read = 0
    # A training file runs to gigabytes and hashing it is not instant, so it
    # reports progress rather than appearing to hang at startup. Below a few
    # hundred megabytes the bar is more noise than help.
    show = size > (256 << 20)
    bar = None
    if show:
        from tqdm import tqdm as _tqdm
        bar = _tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024,
                    desc=f'hashing {os.path.basename(path)}', leave=False)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            if bar is not None:
                bar.update(len(chunk))
            lines += chunk.count(b'\n')
            if h is not None:
                if max_digest_bytes is None:
                    h.update(chunk)
                elif read < max_digest_bytes:
                    h.update(chunk[:max_digest_bytes - read])
            read += len(chunk)
    if bar is not None:
        bar.close()
    info['lines'] = lines
    if h is not None:
        info['sha256'] = h.hexdigest()
        if max_digest_bytes is not None and size > max_digest_bytes:
            info['sha256_covers_bytes'] = max_digest_bytes
    return info


def data_fingerprints(paths: Iterable[Optional[str]], digest: bool = True,
                      max_digest_bytes: Optional[int] = None) -> Dict[str, Any]:
    out = {}
    for p in paths:
        if p:
            out[p] = file_fingerprint(p, digest=digest,
                                      max_digest_bytes=max_digest_bytes)
    return out


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------

def rng_state() -> Dict[str, Any]:
    """
    Every generator that affects training order or initialisation.

    Restoring the seed alone is not enough to resume: the seed reproduces the
    state at step zero, not the state at the step where training stopped.
    """
    import random

    import numpy as np
    state: Dict[str, Any] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore what rng_state() captured. Missing entries are skipped."""
    import random

    import numpy as np
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'].cpu()
                            if hasattr(state['torch'], 'cpu') else state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        except (RuntimeError, ValueError) as e:
            # A checkpoint from a machine with a different GPU count.
            print(f'NOTE  could not restore CUDA RNG state ({e}); '
                  f'CUDA-side randomness will differ from the original run.')


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def collect(data_files: Iterable[Optional[str]] = (),
            digest: bool = True,
            max_digest_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Everything, in one dictionary, for embedding in a checkpoint."""
    from . import __version__
    return {
        'ir_baselines_version': __version__,
        'created': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'working_dir': os.getcwd(),
        'git': git_info(),
        'environment': environment(),
        'data': data_fingerprints(data_files, digest=digest,
                                  max_digest_bytes=max_digest_bytes),
    }


def summarise(prov: Dict[str, Any]) -> str:
    """A few lines for the console, rather than the whole dictionary."""
    if not prov:
        return 'provenance: none recorded'
    lines = []
    g = prov.get('git', {})
    if g.get('available'):
        mark = ' (DIRTY)' if g.get('dirty') else ''
        lines.append(f"  git          {g.get('describe') or g.get('commit', '')[:12]}{mark}")
        if g.get('dirty'):
            n = len(g.get('dirty_files', []))
            lines.append(f"               {n} modified file(s); the commit does not "
                         f"identify what ran")
    else:
        lines.append('  git          not a repository')
    e = prov.get('environment', {})
    lines.append(f"  python       {e.get('python')}  torch {e.get('torch')} "
                 f"(cuda {e.get('cuda')})  transformers {e.get('transformers')}")
    if e.get('gpu'):
        lines.append(f"  gpu          {', '.join(e['gpu'])}")
    for path, d in prov.get('data', {}).items():
        if d.get('exists'):
            short = d.get('sha256', '')[:12]
            lines.append(f"  data         {os.path.basename(path)}  "
                         f"{d['lines']:,} lines  sha256:{short}")
        else:
            lines.append(f"  data         {path}  MISSING")
    return 'provenance:\n' + '\n'.join(lines)


def compare_data(stored: Dict[str, Any], current: Dict[str, Any]):
    """
    Differences between two data fingerprint sets.

    Returns [(path, field, stored, current)]. An empty list means the inputs
    are the same files by digest, not merely the same paths.
    """
    out = []
    for path, was in (stored or {}).items():
        now = (current or {}).get(path)
        if now is None:
            continue
        for field in ('sha256', 'lines', 'bytes'):
            if field in was and field in now and was[field] != now[field]:
                out.append((path, field, was[field], now[field]))
    return out

# ---------------------------------------------------------------------------
# run files
# ---------------------------------------------------------------------------

def run_stats(run_path: str) -> Dict[str, Any]:
    """Shape of a written run: topics, pairs, and the file's own digest."""
    topics = set()
    pairs = 0
    with open(run_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                topics.add(parts[0])
                pairs += 1
    out = file_fingerprint(run_path)
    out['topics'] = len(topics)
    out['pairs'] = pairs
    return out


def write_run_provenance(run_path: str,
                         inference: Dict[str, Any],
                         checkpoint_path: Optional[str] = None,
                         checkpoint_config: Optional[Dict[str, Any]] = None,
                         checkpoint_provenance: Optional[Dict[str, Any]] = None,
                         digest_checkpoint: bool = True) -> str:
    """
    Write `<run>.provenance.json` beside a run file.

    A run file cannot carry its own provenance: the TREC format is six
    whitespace-separated fields per line and every parser, `trec_eval` and
    `pytrec_eval` included, rejects a comment line. So the record goes in a
    sibling, and the run file's own SHA-256 goes inside it — which is what
    lets a reader confirm the sibling belongs to the run rather than to an
    earlier version of it.

    Training-time and inference-time provenance are kept apart because they
    can differ, and the difference is often the answer: the same checkpoint
    scored on a different machine, or under a different transformers version,
    does not always produce the same run.
    """
    record: Dict[str, Any] = {
        'ir_baselines_version': inference.get('ir_baselines_version'),
        'run': run_stats(run_path),
        'produced_by': inference,
    }
    if checkpoint_path:
        ck: Dict[str, Any] = {'path': os.path.abspath(checkpoint_path)}
        if digest_checkpoint and os.path.exists(checkpoint_path):
            ck.update({k: v for k, v in file_fingerprint(checkpoint_path).items()
                       if k in ('bytes', 'sha256', 'modified')})
        if checkpoint_config:
            ck['config'] = checkpoint_config
        if checkpoint_provenance:
            ck['provenance'] = checkpoint_provenance
        record['checkpoint'] = ck

    out_path = run_path + '.provenance.json'
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return out_path


def short_commit(n: int = 7) -> Optional[str]:
    """
    A short commit hash for the run tag, or None outside a repository.

    A dirty tree returns None rather than a hash: a hash that does not
    identify the code is worse than no hash, because it looks like it does.
    """
    g = git_info()
    if not g.get('available') or g.get('dirty'):
        return None
    return g['commit'][:n]

def write_data_provenance(output_path: str,
                          inputs: Iterable[Optional[str]] = (),
                          parameters: Optional[Dict[str, Any]] = None,
                          script: Optional[str] = None,
                          digest_output: bool = True) -> str:
    """
    Write `<output>.provenance.json` beside a generated data file.

    The chain is otherwise broken at its first link. A checkpoint records the
    digest of the training file it read, so you can tell whether two runs used
    the same data -- but nothing records how that file was made, from what, or
    under which parameters. Derived entity judgments are the sharpest case:
    the file is the supervision, and the rule that produced it is the thing
    most worth being able to check later.

    `parameters` should carry the settings that change the output: sampling
    ratios, thresholds, the rule variant. They are recorded, not verified;
    nothing downstream can check them, which is why writing them down is the
    only protection.
    """
    record: Dict[str, Any] = {
        'ir_baselines_version': __import__('ir_baselines').__version__,
        'script': script or (sys.argv[0] if sys.argv else None),
        'output': (file_fingerprint(output_path) if digest_output
                   else {'path': os.path.abspath(output_path)}),
        'parameters': parameters or {},
        'produced_by': collect(data_files=inputs),
    }
    out_path = output_path + '.provenance.json'
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return out_path