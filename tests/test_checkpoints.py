"""
Checkpoint loading and saving.

The layouts here are not hypothetical. Released checkpoints exist in all three
forms, and the `t5.` case is the one that silently does the wrong thing: the
tensors and shapes match, so without the remap a strict load fails, and a
non-strict one would leave the encoder randomly initialised and say nothing.
"""

import pytest
import torch

from ir_baselines import utils


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)


@pytest.fixture
def reference():
    torch.manual_seed(0)
    model = Tiny()
    return model, {k: v.clone() for k, v in model.state_dict().items()}


def _same_weights(model, expected):
    got = model.state_dict()
    return all(torch.equal(got[k], expected[k]) for k in expected)


# ===========================================================  layouts

def test_wrapped_layout_round_trips(tmp_path, reference):
    model, weights = reference
    path = tmp_path / 'wrapped.bin'
    utils.save_checkpoint(str(path), model, {'t5_pooling': 'mean-all'})

    loaded = Tiny()
    extras = utils.load_checkpoint(str(path), loaded, 'cpu')
    assert _same_weights(loaded, weights)
    assert extras['config']['t5_pooling'] == 'mean-all'


def test_bare_state_dict_still_loads(tmp_path, reference):
    """Written by earlier versions, which stored no configuration."""
    model, weights = reference
    path = tmp_path / 'bare.bin'
    torch.save(model.state_dict(), path)

    loaded = Tiny()
    extras = utils.load_checkpoint(str(path), loaded, 'cpu')
    assert _same_weights(loaded, weights)
    assert extras == {}


def test_legacy_t5_prefix_is_remapped(tmp_path, reference):
    model, weights = reference
    renamed = {('t5.' + k[len('encoder.'):] if k.startswith('encoder.') else k): v
               for k, v in model.state_dict().items()}
    path = tmp_path / 'legacy.bin'
    torch.save(renamed, path)

    loaded = Tiny()
    utils.load_checkpoint(str(path), loaded, 'cpu')
    assert _same_weights(loaded, weights)


def test_a_raw_load_of_legacy_keys_would_fail(tmp_path, reference):
    """Which is why the remap exists rather than being left to strict=False."""
    model, _ = reference
    renamed = {('t5.' + k[len('encoder.'):] if k.startswith('encoder.') else k): v
               for k, v in model.state_dict().items()}
    path = tmp_path / 'legacy.bin'
    torch.save(renamed, path)

    raw = torch.load(path, map_location='cpu', weights_only=False)
    with pytest.raises(RuntimeError):
        Tiny().load_state_dict(raw)


# ======================================================  training state

def test_training_state_round_trips(tmp_path, reference):
    model, _ = reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    path = tmp_path / 'full.bin'
    utils.save_checkpoint(str(path), model, {'model': 'tiny'},
                          optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                          epoch=3, best_metric=0.42,
                          history={'epoch': [1, 2, 3]})

    extras = utils.load_checkpoint(str(path), Tiny(), 'cpu')
    assert extras['epoch'] == 3
    assert extras['best_metric'] == pytest.approx(0.42)
    assert extras['history']['epoch'] == [1, 2, 3]
    for key in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'):
        assert key in extras

    # and the optimiser state is actually usable
    fresh = Tiny()
    reloaded = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    reloaded.load_state_dict(extras['optimizer_state_dict'])


def test_optional_fields_are_omitted_when_not_given(tmp_path, reference):
    model, _ = reference
    path = tmp_path / 'minimal.bin'
    utils.save_checkpoint(str(path), model, {'model': 'tiny'})

    extras = utils.load_checkpoint(str(path), Tiny(), 'cpu')
    for key in ('optimizer_state_dict', 'epoch', 'rng_state', 'provenance'):
        assert key not in extras


def test_save_to_none_is_a_no_op(reference):
    model, _ = reference
    assert utils.save_checkpoint(None, model, {}) is None


def test_load_from_none_returns_nothing(reference):
    model, _ = reference
    assert utils.load_checkpoint(None, model, 'cpu') == {}


# =====================================================  config checking

@pytest.mark.parametrize('stored,current,keys,expected', [
    ({'a': 1}, {'a': 1}, ('a',), []),
    ({'a': 1}, {'a': 2}, ('a',), [('a', 1, 2)]),
    ({}, {'a': 2}, ('a',), []),                    # absent keys are skipped
    ({'a': None}, {'a': 2}, ('a',), []),           # so are None values
    ({'a': 1}, {'a': None}, ('a',), []),
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 3}, ('a', 'b'), [('b', 2, 3)]),
])
def test_check_config(stored, current, keys, expected):
    assert utils.check_config(stored, current, keys) == expected


def test_check_config_skips_keys_it_was_not_asked_about():
    assert utils.check_config({'a': 1}, {'a': 2}, ('b',)) == []
