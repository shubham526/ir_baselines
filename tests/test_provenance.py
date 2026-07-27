"""
Checks the provenance record: git parsing, data fingerprints, RNG round-trip,
and the enriched checkpoint.
"""

import json
import os
import subprocess
import sys
import tempfile

import torch

from ir_baselines import provenance as P
from ir_baselines import utils

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# --------------------------------------------------------------- git
print('\ngit')
repo = tempfile.mkdtemp()
subprocess.run(['git', 'init', '-q', repo], check=True)
for cmd in (['git', 'config', 'user.email', 't@t'], ['git', 'config', 'user.name', 't']):
    subprocess.run(cmd, cwd=repo, check=True)
open(os.path.join(repo, 'a.txt'), 'w').write('one\n')
subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
subprocess.run(['git', 'commit', '-qm', 'first'], cwd=repo, check=True)

g = P.git_info(repo)
check('clean tree reports a commit', g['available'] and len(g['commit']) == 40)
check('clean tree is not dirty', g['dirty'] is False, f"dirty={g['dirty']}")

# A modified tracked file: git status prints ' M path', with a LEADING SPACE.
# Stripping the output before slicing off the status characters eats the first
# character of every path, so this asserts the path survives intact.
open(os.path.join(repo, 'a.txt'), 'w').write('two\n')
g = P.git_info(repo)
check('modified tree is dirty', g['dirty'] is True)
check('dirty path is intact', g['dirty_files'] == ['a.txt'],
      f"got {g['dirty_files']}")

check('outside a repository reports unavailable',
      P.git_info(tempfile.mkdtemp())['available'] is False)

# --------------------------------------------------------- environment
print('\nenvironment')
e = P.environment()
for key in ('python', 'platform', 'torch', 'transformers'):
    check(f'{key} recorded', e.get(key) is not None, str(e.get(key)))

# ---------------------------------------------------------------- data
print('\ndata fingerprints')
fd, p1 = tempfile.mkstemp(suffix='.jsonl')
os.write(fd, b'{"a":1}\n{"a":2}\n'); os.close(fd)
f1 = P.file_fingerprint(p1)
check('line count', f1['lines'] == 2, str(f1['lines']))
check('digest recorded', len(f1['sha256']) == 64)

# same length, different content -- the case a size or line check misses
fd, p2 = tempfile.mkstemp(suffix='.jsonl')
os.write(fd, b'{"a":1}\n{"a":3}\n'); os.close(fd)
f2 = P.file_fingerprint(p2)
check('same size and lines, different digest',
      f1['bytes'] == f2['bytes'] and f1['lines'] == f2['lines']
      and f1['sha256'] != f2['sha256'])

check('missing file reported, not raised',
      P.file_fingerprint('/nonexistent/x')['exists'] is False)

changed = P.compare_data({p1: f1}, {p1: f2})
check('compare_data detects a content change',
      any(c[1] == 'sha256' for c in changed), str([c[1] for c in changed]))
check('compare_data is quiet on identical files',
      P.compare_data({p1: f1}, {p1: f1}) == [])

# ----------------------------------------------------------------- rng
print('\nRNG round-trip')
import random

import numpy as np
random.seed(1); np.random.seed(1); torch.manual_seed(1)
state = P.rng_state()
before = (random.random(), float(np.random.rand()), float(torch.rand(1)))
# advance all three
random.random(); np.random.rand(); torch.rand(1)
P.set_rng_state(state)
after = (random.random(), float(np.random.rand()), float(torch.rand(1)))
check('python RNG restored', before[0] == after[0])
check('numpy RNG restored', before[1] == after[1])
check('torch RNG restored', before[2] == after[2])
check('empty state is a no-op', P.set_rng_state({}) is None)

# ---------------------------------------------------------- checkpoint
print('\ncheckpoint round-trip')


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)


m = Tiny()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
opt.step()
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
scaler = torch.amp.GradScaler('cuda', enabled=False)
prov = P.collect([p1])

fd, ck = tempfile.mkstemp(suffix='.bin'); os.close(fd)
utils.save_checkpoint(ck, m, {'model': 'tiny'}, optimizer=opt, scheduler=sched,
                      scaler=scaler, epoch=3, best_metric=0.42, provenance=prov,
                      rng=P.rng_state(), history={'epoch': [1, 2, 3]})

m2 = Tiny()
extras = utils.load_checkpoint(ck, m2, 'cpu')
for key in ('config', 'provenance', 'optimizer_state_dict', 'scheduler_state_dict',
            'scaler_state_dict', 'epoch', 'best_metric', 'rng_state', 'history'):
    check(f'{key} survives the round trip', key in extras)
check('epoch value', extras.get('epoch') == 3)
check('best_metric value', extras.get('best_metric') == 0.42)
check('provenance carries the git record', 'git' in extras.get('provenance', {}))
check('optimizer state is loadable',
      opt.load_state_dict(extras['optimizer_state_dict']) is None)

# a bare state_dict still loads, and reports nothing extra
fd, bare = tempfile.mkstemp(suffix='.bin'); os.close(fd)
torch.save(m.state_dict(), bare)
check('bare state_dict returns no extras',
      utils.load_checkpoint(bare, Tiny(), 'cpu') == {})

for f in (p1, p2, bare):
    os.unlink(f)



# ------------------------------------------------------------ run files
print('\nrun provenance')
fd, run_path = tempfile.mkstemp(suffix='.run'); os.close(fd)
with open(run_path, 'w') as f:
    for t in range(3):
        for d in range(4):
            f.write(f't{t} Q0 t{t}_d{d} {d + 1} {1.0 - d / 10:.4f} tag\n')

stats = P.run_stats(run_path)
check('topics counted', stats['topics'] == 3, str(stats['topics']))
check('pairs counted', stats['pairs'] == 12, str(stats['pairs']))
check('run digest recorded', len(stats['sha256']) == 64)

inference = P.collect([run_path])
out = P.write_run_provenance(run_path, inference=inference,
                             checkpoint_path=ck if os.path.exists(ck) else None,
                             checkpoint_config={'model': 'tiny'},
                             checkpoint_provenance=prov)
check('sibling written beside the run', out == run_path + '.provenance.json'
      and os.path.exists(out))

with open(out) as f:
    rec = json.load(f)
check('sibling records the run digest',
      rec['run']['sha256'] == stats['sha256'])
check('sibling records inference provenance', 'produced_by' in rec)
check('sibling records the checkpoint', 'checkpoint' in rec)
check('training and inference provenance are kept apart',
      rec['checkpoint'].get('provenance') is not None
      and rec['checkpoint']['provenance'] is not rec['produced_by'])

# the digest is what lets a reader tell the sibling from a stale one
with open(run_path, 'a') as f:
    f.write('t9 Q0 t9_d0 1 0.5 tag\n')
check('a changed run no longer matches its sibling digest',
      P.run_stats(run_path)['sha256'] != rec['run']['sha256'])

os.unlink(run_path); os.unlink(out)

# --------------------------------------------------------- short_commit
print('\nshort_commit')
subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
subprocess.run(['git', 'commit', '-qm', 'second'], cwd=repo, check=True)
# git_info() defaults to the package directory, so point short_commit at a
# known repo by checking git_info directly for the semantics it relies on.
g = P.git_info(repo)
check('clean tree yields a usable hash', g['dirty'] is False and len(g['commit']) == 40)
open(os.path.join(repo, 'a.txt'), 'w').write('three\n')
check('a modified tracked file makes it unusable', P.git_info(repo)['dirty'] is True)
open(os.path.join(repo, 'untracked.txt'), 'w').write('x\n')
subprocess.run(['git', 'checkout', '--', 'a.txt'], cwd=repo, check=True)
g = P.git_info(repo)
check('an untracked file does NOT make it unusable', g['dirty'] is False,
      f"untracked={g['untracked']}")

os.unlink(ck)

print()
print('ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
sys.exit(1 if FAILED else 0)