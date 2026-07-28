"""
Provenance: git state, environment, data fingerprints, RNG round-trip, and the
sibling record written beside a run file.
"""

import json
import os
import subprocess

import numpy as np
import pytest
import torch

from ir_baselines import provenance as P
from ir_baselines import utils


def _git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


# ===============================================================  git

def test_clean_tree_reports_a_commit(git_repo):
    info = P.git_info(git_repo)
    assert info['available'] is True
    assert len(info['commit']) == 40
    assert info['dirty'] is False


def test_modified_tracked_file_makes_the_tree_dirty(git_repo):
    (open(os.path.join(git_repo, 'a.txt'), 'w')).write('two\n')
    info = P.git_info(git_repo)
    assert info['dirty'] is True


def test_dirty_path_is_not_truncated(git_repo):
    """
    `git status --porcelain` prints ' M path' with a LEADING SPACE. Stripping
    the output before slicing off the two status characters eats the first
    character of every path, which corrupts the record silently.
    """
    open(os.path.join(git_repo, 'a.txt'), 'w').write('two\n')
    info = P.git_info(git_repo)
    assert info['dirty_files'] == ['a.txt']
    assert all(os.path.exists(os.path.join(git_repo, f))
               for f in info['dirty_files'])


def test_untracked_file_does_not_make_the_tree_dirty(git_repo):
    """
    A repository nearly always has untracked files -- __pycache__, scratch
    output. Treating those as dirty would mean the commit hash was never
    usable for provenance.
    """
    open(os.path.join(git_repo, 'scratch.txt'), 'w').write('x\n')
    info = P.git_info(git_repo)
    assert info['dirty'] is False
    assert info['untracked'] == ['scratch.txt']


def test_outside_a_repository_reports_unavailable(tmp_path):
    assert P.git_info(str(tmp_path))['available'] is False


# ======================================================  environment

@pytest.mark.parametrize('key', ['python', 'platform', 'hostname', 'torch',
                                 'transformers', 'numpy'])
def test_environment_records(key):
    assert P.environment().get(key) is not None


# =====================================================  fingerprints

@pytest.fixture
def two_files(tmp_path):
    """Same size and line count, different contents."""
    a = tmp_path / 'a.jsonl'
    b = tmp_path / 'b.jsonl'
    a.write_text('{"a":1}\n{"a":2}\n')
    b.write_text('{"a":1}\n{"a":3}\n')
    return str(a), str(b)


def test_fingerprint_records_lines_and_digest(two_files):
    a, _ = two_files
    f = P.file_fingerprint(a)
    assert f['lines'] == 2
    assert len(f['sha256']) == 64
    assert f['exists'] is True


def test_digest_distinguishes_files_a_size_check_cannot(two_files):
    """
    Regenerating training data with a different negative sample gives the same
    size and line count. Only the digest differs.
    """
    fa, fb = (P.file_fingerprint(p) for p in two_files)
    assert fa['bytes'] == fb['bytes']
    assert fa['lines'] == fb['lines']
    assert fa['sha256'] != fb['sha256']


def test_missing_file_is_reported_not_raised():
    assert P.file_fingerprint('/nonexistent/file')['exists'] is False


def test_compare_data_detects_a_content_change(two_files):
    a, b = two_files
    changed = P.compare_data({a: P.file_fingerprint(a)}, {a: P.file_fingerprint(b)})
    assert any(field == 'sha256' for _, field, _, _ in changed)


def test_compare_data_is_quiet_on_identical_files(two_files):
    a, _ = two_files
    f = P.file_fingerprint(a)
    assert P.compare_data({a: f}, {a: f}) == []


# =============================================================  rng

def test_rng_state_round_trips():
    import random
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    state = P.rng_state()
    before = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.random(); np.random.rand(); torch.rand(1)     # advance all three
    P.set_rng_state(state)
    after = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert before == after


def test_empty_rng_state_is_a_no_op():
    assert P.set_rng_state({}) is None


# =======================================================  run files

@pytest.fixture
def run_file(tmp_path):
    path = tmp_path / 'fold-0.run'
    with open(path, 'w') as f:
        for t in range(3):
            for d in range(4):
                f.write(f't{t} Q0 t{t}_d{d} {d + 1} {1.0 - d / 10:.4f} tag\n')
    return str(path)


def test_run_stats(run_file):
    stats = P.run_stats(run_file)
    assert stats['topics'] == 3
    assert stats['pairs'] == 12
    assert len(stats['sha256']) == 64


def test_sibling_is_written_beside_the_run(run_file, tmp_path):
    out = P.write_run_provenance(run_file, inference=P.collect([run_file]))
    assert out == run_file + '.provenance.json'
    assert os.path.exists(out)


def test_sibling_records_the_runs_own_digest(run_file):
    """Which is what lets a reader tell the sibling from a stale one."""
    out = P.write_run_provenance(run_file, inference=P.collect([run_file]))
    with open(out) as f:
        rec = json.load(f)
    assert rec['run']['sha256'] == P.run_stats(run_file)['sha256']


def test_a_changed_run_no_longer_matches_its_sibling(run_file):
    out = P.write_run_provenance(run_file, inference=P.collect([run_file]))
    with open(out) as f:
        recorded = json.load(f)['run']['sha256']
    with open(run_file, 'a') as f:
        f.write('t9 Q0 t9_d0 1 0.5 tag\n')
    assert P.run_stats(run_file)['sha256'] != recorded


def test_training_and_inference_provenance_are_kept_apart(run_file, tmp_path):
    """
    The same checkpoint scored on a different machine does not always produce
    the same run, so the two records must not be collapsed.
    """
    ck = tmp_path / 'm.bin'
    torch.save({'model_state_dict': {}, 'config': {}}, ck)
    training = P.collect([])
    out = P.write_run_provenance(
        run_file, inference=P.collect([run_file]),
        checkpoint_path=str(ck), checkpoint_config={'model': 'tiny'},
        checkpoint_provenance=training)
    with open(out) as f:
        rec = json.load(f)
    assert 'produced_by' in rec
    assert rec['checkpoint']['provenance']['created'] == training['created']
    assert rec['checkpoint']['sha256']


# ======================================================  collect()

def test_collect_carries_every_section(two_files):
    a, _ = two_files
    prov = P.collect([a])
    for key in ('ir_baselines_version', 'created', 'command', 'git',
                'environment', 'data'):
        assert key in prov
    assert a in prov['data']


def test_summarise_handles_an_empty_record():
    assert 'none recorded' in P.summarise({})


# ==============================  the checkpoint carries all of it

def test_checkpoint_carries_provenance(tmp_path, two_files):
    a, _ = two_files

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)

    path = tmp_path / 'ck.bin'
    utils.save_checkpoint(str(path), Tiny(), {'model': 'tiny'},
                          provenance=P.collect([a]), rng=P.rng_state())
    extras = utils.load_checkpoint(str(path), Tiny(), 'cpu')
    assert 'git' in extras['provenance']
    assert a in extras['provenance']['data']
    assert 'rng_state' in extras
