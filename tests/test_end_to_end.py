"""
End-to-end: the entry points, against a real tokenizer and a real (if tiny)
model.

These are slower than the rest of the suite because they train. Skip them with
`pytest -m "not slow"`.

Every assertion here is about exit status or a written artifact rather than
about a metric: a few dozen synthetic examples say nothing about quality, and
the point is that the pipeline is wired correctly and the guards fire.
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def run_cli(module, *args, expect_ok=True):
    """Invoke an entry point the way a user would, and check its exit status."""
    proc = subprocess.run([sys.executable, '-m', f'ir_baselines.{module}', *args],
                          capture_output=True, text=True)
    if expect_ok and proc.returncode != 0:
        pytest.fail(f'{module} exited {proc.returncode}\n'
                    f'--- stdout\n{proc.stdout[-2000:]}\n'
                    f'--- stderr\n{proc.stderr[-2000:]}')
    # Diagnostics are printed as they are found, but the final refusal is
    # raised, and SystemExit goes to stderr. An assertion about what the tool
    # said should not depend on which of the two it was.
    proc.output = proc.stdout + proc.stderr
    return proc


@pytest.fixture
def trained(tmp_path, corpus, tiny_encoder):
    """A trained cross-encoder, its checkpoint directory returned."""
    out = tmp_path / 'out'
    run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
            '--train', corpus['train'], '--dev', corpus['dev'],
            '--qrels', corpus['qrels'], '--save-dir', str(out),
            '--epoch', '2', '--batch-size', '4', '--max-len', '32', '--seed', '42')
    return out


# =========================================================  training

@pytest.mark.parametrize('model,extra', [
    ('bert', []),
    ('me-bert', ['--logit-scale', '--max-query-len', '8', '--max-doc-len', '32']),
    ('poly-encoder', ['--logit-scale', '--poly-m', '4',
                      '--max-query-len', '8', '--max-doc-len', '32']),
])
def test_train_every_model_family(tmp_path, corpus, tiny_encoder, model, extra):
    out = tmp_path / model
    run_cli('train', '--model', model, '--pretrain', 'tiny',
            '--train', corpus['train'], '--dev', corpus['dev'],
            '--qrels', corpus['qrels'], '--save-dir', str(out),
            '--epoch', '1', '--batch-size', '4', '--max-len', '32',
            '--seed', '42', *extra)
    assert (out / 'model.bin').exists()
    assert (out / 'config.json').exists()


def test_in_batch_negatives_train(tmp_path, corpus, tiny_encoder):
    out = tmp_path / 'ce'
    run_cli('train', '--model', 'me-bert', '--pretrain', 'tiny',
            '--loss', 'ce-inbatch', '--positives-only',
            '--train', corpus['train'], '--dev', corpus['dev'],
            '--qrels', corpus['qrels'], '--save-dir', str(out),
            '--epoch', '1', '--batch-size', '4',
            '--max-query-len', '8', '--max-doc-len', '32')
    assert (out / 'model.bin').exists()


def test_both_checkpoints_are_written(trained):
    """model.bin is the best; last.bin is where training got to."""
    assert (trained / 'model.bin').exists()
    assert (trained / 'last.bin').exists()


# ========================================================  inference

def test_inference_writes_a_run_and_its_provenance(tmp_path, corpus, tiny_encoder, trained):
    runs = tmp_path / 'runs'
    run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
            '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
            '--save-dir', str(runs), '--run', 'fold-0.run', '--max-len', '32',
            '--qrels', corpus['qrels'], '--expected-topics', str(corpus['topics']))

    run_file = runs / 'fold-0.run'
    sidecar = runs / 'fold-0.run.provenance.json'
    assert run_file.exists() and sidecar.exists()

    from ir_baselines import provenance as P
    rec = json.loads(sidecar.read_text())
    assert rec['run']['sha256'] == P.run_stats(str(run_file))['sha256']
    assert rec['checkpoint']['config']['model'] == 'cross-encoder'


def test_run_file_is_well_formed(tmp_path, corpus, tiny_encoder, trained):
    runs = tmp_path / 'runs'
    run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
            '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
            '--save-dir', str(runs), '--run', 'r.run', '--max-len', '32')
    for line in (runs / 'r.run').read_text().splitlines():
        fields = line.split()
        assert len(fields) == 6, f'not six fields: {line!r}'
        assert fields[1] == 'Q0'
        int(fields[3]); float(fields[4])


def test_tag_commit_is_skipped_outside_a_repository(tmp_path, corpus, tiny_encoder, trained):
    runs = tmp_path / 'runs'
    proc = run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
                   '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
                   '--save-dir', str(runs), '--run', 't.run', '--max-len', '32',
                   '--tag-commit')
    tag = (runs / 't.run').read_text().splitlines()[0].split()[5]
    # either a clean commit was available, or the tag is unchanged and said so
    assert tag == 'bert' or tag.startswith('bert.')
    if tag == 'bert':
        assert '--tag-commit' in proc.stdout


# =====================================================  determinism

def test_same_seed_gives_identical_runs(tmp_path, corpus, tiny_encoder):
    outputs = []
    for i in (1, 2):
        out = tmp_path / f'det{i}'
        run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
                '--train', corpus['train'], '--dev', corpus['dev'],
                '--qrels', corpus['qrels'], '--save-dir', str(out),
                '--epoch', '2', '--batch-size', '4', '--max-len', '32', '--seed', '42')
        run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
                '--test', corpus['test'], '--checkpoint', str(out / 'model.bin'),
                '--save-dir', str(out), '--run', 'o.run', '--max-len', '32')
        outputs.append((out / 'o.run').read_text())
    assert outputs[0] == outputs[1]


# =========================================================  resume

def test_resume_continues_rather_than_restarting(tmp_path, corpus, tiny_encoder, trained):
    run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
            '--train', corpus['train'], '--dev', corpus['dev'],
            '--qrels', corpus['qrels'], '--save-dir', str(trained),
            '--epoch', '4', '--batch-size', '4', '--max-len', '32', '--seed', '42',
            '--resume', str(trained / 'last.bin'))
    history = json.loads((trained / 'training_history.json').read_text())
    assert history['epoch'] == [1, 2, 3, 4]


def test_resume_refuses_changed_data(tmp_path, corpus, tiny_encoder, trained):
    rows = [json.loads(l) for l in open(corpus['train'])]
    rows[0]['doc'] = 'entirely different text'
    with open(corpus['train'], 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    proc = run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
                   '--train', corpus['train'], '--dev', corpus['dev'],
                   '--qrels', corpus['qrels'], '--save-dir', str(tmp_path / 'x'),
                   '--epoch', '6', '--batch-size', '4', '--max-len', '32',
                   '--resume', str(trained / 'last.bin'), expect_ok=False)
    assert proc.returncode != 0
    assert 'differ from those the checkpoint was trained on' in proc.output


# ==========================================================  inspect

def test_inspect_reads_a_checkpoint(trained):
    proc = run_cli('inspect', str(trained / 'model.bin'))
    assert 'configuration' in proc.stdout
    assert 'provenance' in proc.stdout


def test_inspect_reads_a_run(tmp_path, corpus, tiny_encoder, trained):
    runs = tmp_path / 'runs'
    run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
            '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
            '--save-dir', str(runs), '--run', 'i.run', '--max-len', '32')
    proc = run_cli('inspect', str(runs / 'i.run'))
    assert 'produced by' in proc.stdout
    assert 'checkpoint' in proc.stdout


def test_inspect_warns_when_a_run_has_changed(tmp_path, corpus, tiny_encoder, trained):
    runs = tmp_path / 'runs'
    run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
            '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
            '--save-dir', str(runs), '--run', 'w.run', '--max-len', '32')
    path = runs / 'w.run'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    proc = run_cli('inspect', str(path))
    assert 'WARNING' in proc.stdout


# ===========================================================  guards

def test_inference_without_a_checkpoint_is_refused(tmp_path, corpus, tiny_encoder):
    proc = run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
                   '--test', corpus['test'], '--save-dir', str(tmp_path / 'x'),
                   '--run', 'x.run', expect_ok=False)
    assert proc.returncode != 0


def test_length_mismatch_is_refused(tmp_path, corpus, tiny_encoder, trained):
    proc = run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
                   '--test', corpus['test'], '--checkpoint', str(trained / 'model.bin'),
                   '--save-dir', str(tmp_path / 'x'), '--run', 'x.run',
                   '--max-len', '16', expect_ok=False)
    assert proc.returncode != 0
    assert 'max_len' in proc.stdout


def test_unverified_checkpoint_is_refused(tmp_path, corpus, tiny_encoder, trained):
    import torch
    bare = tmp_path / 'bare.bin'
    obj = torch.load(trained / 'model.bin', weights_only=False)
    torch.save(obj['model_state_dict'], bare)

    proc = run_cli('test', '--model', 'bert', '--pretrain', 'tiny',
                   '--test', corpus['test'], '--checkpoint', str(bare),
                   '--save-dir', str(tmp_path / 'x'), '--run', 'x.run',
                   '--max-len', '32', expect_ok=False)
    assert proc.returncode != 0


@pytest.mark.parametrize('args,fragment', [
    (['--loss', 'bce'], 'cross-entropy'),
    (['--loss', 'ce-inbatch'], 'not available'),
])
def test_incompatible_objective_is_refused(tmp_path, corpus, tiny_encoder, args, fragment):
    proc = run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
                   '--train', corpus['train'], '--dev', corpus['dev'],
                   '--qrels', corpus['qrels'], '--save-dir', str(tmp_path / 'x'),
                   *args, expect_ok=False)
    assert proc.returncode != 0
    assert fragment in proc.stderr


def test_positives_only_with_a_pointwise_objective_is_refused(tmp_path, corpus, tiny_encoder):
    proc = run_cli('train', '--model', 'me-bert', '--pretrain', 'tiny',
                   '--loss', 'bce', '--positives-only',
                   '--train', corpus['train'], '--dev', corpus['dev'],
                   '--qrels', corpus['qrels'], '--save-dir', str(tmp_path / 'x'),
                   expect_ok=False)
    assert proc.returncode != 0


def test_resume_with_init_checkpoint_is_refused(tmp_path, corpus, tiny_encoder, trained):
    proc = run_cli('train', '--model', 'bert', '--pretrain', 'tiny',
                   '--resume', str(trained / 'last.bin'),
                   '--init-checkpoint', str(trained / 'last.bin'),
                   '--train', corpus['train'], '--dev', corpus['dev'],
                   '--qrels', corpus['qrels'], '--save-dir', str(tmp_path / 'x'),
                   expect_ok=False)
    assert proc.returncode != 0


def test_list_models_needs_no_other_arguments():
    proc = run_cli('train', '--list-models')
    assert 'encoding' in proc.stdout
    assert 'me-bert' in proc.stdout