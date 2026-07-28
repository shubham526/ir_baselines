"""
The dispatch layer: each model declares how a batch reaches it and which
objective its output feeds, and the rest of the pipeline reads those.

The first test here guards a hazard that has no other detection: the trainer
calls `model(*[batch[k] for k in TENSOR_KEYS[ENCODING]])`, passing tensors
POSITIONALLY. If a forward signature and the key order ever disagree, tensors
land in the wrong parameters, and with compatible shapes that produces wrong
numbers rather than an error.
"""

import inspect

import pytest
import torch

from ir_baselines.data.dataset import TENSOR_KEYS
from ir_baselines.models import MODEL_CHOICES, REGISTRY, spec_for
from ir_baselines.trainer import LOSS_CHOICES, Trainer

MODEL_CLASSES = sorted({spec.cls for spec in REGISTRY.values()}, key=lambda c: c.__name__)


@pytest.mark.parametrize('cls', MODEL_CLASSES, ids=lambda c: c.__name__)
def test_forward_signature_matches_batch_key_order(cls):
    params = [p for p in inspect.signature(cls.forward).parameters if p != 'self']
    assert params == list(TENSOR_KEYS[cls.ENCODING])


@pytest.mark.parametrize('cls', MODEL_CLASSES, ids=lambda c: c.__name__)
def test_contract_attributes_are_coherent(cls):
    assert cls.ENCODING in TENSOR_KEYS
    assert cls.LOSS in LOSS_CHOICES
    # in-batch training needs separate representations to cross-score
    assert cls.SUPPORTS_INBATCH == hasattr(cls, 'score_matrix')


@pytest.mark.parametrize('cls', MODEL_CLASSES, ids=lambda c: c.__name__)
def test_every_model_implements_the_contract(cls):
    for method in ('forward', 'score', 'config_dict'):
        assert callable(getattr(cls, method, None)), f'{cls.__name__} lacks {method}'


@pytest.mark.parametrize('name', MODEL_CHOICES)
def test_registry_entries_are_well_formed(name):
    spec = spec_for(name)
    assert spec.cls in MODEL_CLASSES
    assert spec.summary
    from ir_baselines.models import ENCODER_MAP
    assert spec.encoder in ENCODER_MAP


def test_unknown_model_is_rejected_with_the_options():
    with pytest.raises(KeyError, match='unknown model'):
        spec_for('colbert')


# =======================================  objective / encoding compatibility

class _StubModel:
    """Only the attributes the Trainer dispatches on."""

    def __init__(self, encoding, supports_inbatch):
        self.ENCODING = encoding
        self.SUPPORTS_INBATCH = supports_inbatch

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])


def _build_trainer(model, loss):
    return Trainer(model=model, optimizer=None, criterion=None, scheduler=None,
                   metric='map', data_loader=[], device=torch.device('cpu'),
                   loss=loss)


@pytest.mark.parametrize('encoding,supports,loss,accepted', [
    ('pair', False, 'cross-entropy', True),
    ('pair', False, 'bce', False),
    ('pair', False, 'ce-inbatch', False),
    ('dual', True, 'bce', True),
    ('dual', True, 'ce-inbatch', True),
    ('dual', True, 'cross-entropy', False),
])
def test_objective_compatibility(encoding, supports, loss, accepted):
    model = _StubModel(encoding, supports)
    if accepted:
        assert _build_trainer(model, loss) is not None
    else:
        with pytest.raises(ValueError):
            _build_trainer(model, loss)


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match='loss must be one of'):
        _build_trainer(_StubModel('dual', True), 'nonsense')


def test_inbatch_refused_when_the_model_cannot_cross_score():
    """A cross-encoder has no separate query and document representations."""
    from ir_baselines.trainer import inbatch_scores
    with pytest.raises(TypeError, match='encodes the query and document together'):
        inbatch_scores(_StubModel('pair', False), {}, torch.device('cpu'))
