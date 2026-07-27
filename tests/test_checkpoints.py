"""
Checks that all three checkpoint layouts load, and that the T5 pooling stored
in a checkpoint is what test.py compares against.

The prefix case is not hypothetical: checkpoints from the original experiments
name the encoder attribute 't5.' while the released model calls it 'encoder.'.
Same 102 tensors, same shapes -- so a strict load succeeds on the wrong keys
if the rename is not applied, and silently produces a randomly initialised
encoder.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

from ir_baselines import utils

FAILED = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILED.append(name)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)


def tmp():
    fd, path = tempfile.mkstemp(suffix='.bin')
    os.close(fd)
    return path


ref = Tiny()
ref_sd = {k: v.clone() for k, v in ref.state_dict().items()}

print('\nCheckpoint layouts')

# 1. written by save_checkpoint
p1 = tmp()
utils.save_checkpoint(p1, ref, {'t5_pooling': 'mean-all', 'model': 'cross-encoder'})
m = Tiny()
cfg = utils.load_checkpoint(p1, m, 'cpu')
check('wrapped layout loads', all(
    torch.equal(m.state_dict()[k], ref_sd[k]) for k in ref_sd))
check('config comes back from the checkpoint', cfg.get('t5_pooling') == 'mean-all')

# 2. bare state_dict, as written by earlier versions
p2 = tmp()
torch.save(ref.state_dict(), p2)
m = Tiny()
cfg = utils.load_checkpoint(p2, m, 'cpu')
check('bare state_dict loads', all(
    torch.equal(m.state_dict()[k], ref_sd[k]) for k in ref_sd))
check('bare state_dict reports no config', cfg == {})

# 3. 't5.'-prefixed keys
p3 = tmp()
torch.save({('t5.' + k[len('encoder.'):] if k.startswith('encoder.') else k): v
            for k, v in ref.state_dict().items()}, p3)
m = Tiny()
utils.load_checkpoint(p3, m, 'cpu')
check("'t5.' prefix is remapped to 'encoder.'", all(
    torch.equal(m.state_dict()[k], ref_sd[k]) for k in ref_sd))

print('\nWhy the remap matters')
raw = torch.load(p3, map_location='cpu', weights_only=False)
m = Tiny()
try:
    m.load_state_dict(raw)
    check('a raw load of prefixed keys is rejected by strict loading', False)
except RuntimeError:
    check('a raw load of prefixed keys is rejected by strict loading', True)

for p in (p1, p2, p3):
    os.unlink(p)

print()
print('ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} check(s) failed')
sys.exit(1 if FAILED else 0)