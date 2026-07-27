"""
The trainer calls model(*[batch[k] for k in TENSOR_KEYS[ENCODING]]), so the
key order must match each forward() signature exactly. Reordering either
silently feeds tensors into the wrong parameters.
"""
import sys, types, inspect, torch
stub = types.ModuleType('transformers')
class _Cfg:
    hidden_size = 8; model_type = 'bert'
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
class _M:
    @classmethod
    def from_pretrained(cls, *a, **k): return torch.nn.Identity()
stub.AutoConfig = _Cfg; stub.AutoModel = _M
stub.DistilBertModel = type('DistilBertModel', (), {})
stub.T5EncoderModel = type('T5EncoderModel', (), {})
sys.modules['transformers'] = stub

from ir_baselines.data.dataset import TENSOR_KEYS
from ir_baselines.models import REGISTRY

bad = 0
seen = set()
for name, spec in REGISTRY.items():
    if spec.cls in seen:
        continue
    seen.add(spec.cls)
    params = [p for p in inspect.signature(spec.cls.forward).parameters
              if p != 'self']
    keys = list(TENSOR_KEYS[spec.cls.ENCODING])
    ok = params == keys
    if not ok: bad += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {spec.cls.__name__:<14} "
          f"ENCODING={spec.cls.ENCODING}")
    print(f"        forward params : {params}")
    print(f"        TENSOR_KEYS    : {keys}")

print()

# --- the Trainer must refuse an objective the model cannot consume ---------
print('Objective compatibility')
from ir_baselines.trainer import Trainer, LOSS_CHOICES


class _Stub:
    def __init__(self, encoding, supports):
        self.ENCODING = encoding
        self.SUPPORTS_INBATCH = supports

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])


def _build(model, loss):
    return Trainer(model=model, optimizer=None, criterion=None, scheduler=None,
                   metric='map', data_loader=[], device=torch.device('cpu'),
                   loss=loss)


cases = [
    ('pair', False, 'cross-entropy', True),
    ('pair', False, 'bce', False),
    ('pair', False, 'ce-inbatch', False),
    ('dual', True, 'bce', True),
    ('dual', True, 'ce-inbatch', True),
    ('dual', True, 'cross-entropy', False),
]
for encoding, supports, loss, should_work in cases:
    try:
        _build(_Stub(encoding, supports), loss)
        got = True
    except ValueError:
        got = False
    ok = got == should_work
    if not ok:
        bad += 1
    verdict = 'accepted' if got else 'rejected'
    print(f"  {'PASS' if ok else 'FAIL'}  {encoding:<5} + {loss:<14} {verdict}")

try:
    _build(_Stub('dual', True), 'nonsense')
    print('  FAIL  unknown loss accepted'); bad += 1
except ValueError:
    print('  PASS  unknown loss rejected')

print()
print('ALL CHECKS PASSED' if not bad else f'{bad} FAILED')
sys.exit(1 if bad else 0)