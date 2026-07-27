"""
Model registry.

`--model` names a system; `--pretrain` overrides the encoder it wraps.

    python -m ir_baselines.train --model rankt5 ...
    python -m ir_baselines.train --model me-bert --pretrain roberta ...

Seven of the entries are the same CrossEncoder class over different pretrained
encoders, which is why `bert` and `rankt5` appear as separate models: they are
the systems as usually named, not separate architectures.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

from .base import BaselineRanker, masked_mean, mean_all
from .cross_encoder import T5_POOLING_CHOICES, CrossEncoder
from .multi_vector import MEBERT, PolyEncoder

__all__ = [
    'BaselineRanker', 'CrossEncoder', 'MEBERT', 'PolyEncoder',
    'REGISTRY', 'MODEL_CHOICES', 'ENCODER_MAP', 'Spec',
    'build_model', 'spec_for', 'describe', 'resolve_encoder',
    'masked_mean', 'mean_all', 'T5_POOLING_CHOICES',
]


@dataclass(frozen=True)
class Spec:
    """How to build one system."""
    cls: type
    encoder: str            # default --pretrain key
    summary: str            # one line, shown by --list-models
    note: str = ''


#: Short names for pretrained encoders, so --pretrain stays readable.
ENCODER_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-base',
    'ernie': 'nghuyong/ernie-2.0-base-en',
    'electra': 'google/electra-small-discriminator',
    'conv-bert': 'YituTech/conv-bert-base',
    't5': 't5-base',
}


REGISTRY: Dict[str, Spec] = {
    # -- cross-encoders: one class, seven encoders ------------------------
    'bert':      Spec(CrossEncoder, 'bert',      'Cross-encoder over BERT-base'),
    'roberta':   Spec(CrossEncoder, 'roberta',   'Cross-encoder over RoBERTa-base'),
    'deberta':   Spec(CrossEncoder, 'deberta',   'Cross-encoder over DeBERTa-base'),
    'electra':   Spec(CrossEncoder, 'electra',   'Cross-encoder over ELECTRA-small'),
    'conv-bert': Spec(CrossEncoder, 'conv-bert', 'Cross-encoder over ConvBERT-base'),
    'ernie':     Spec(CrossEncoder, 'ernie',     'Cross-encoder over ERNIE 2.0 base'),
    'rankt5':    Spec(CrossEncoder, 't5',        'Cross-encoder over the T5 encoder',
                      note='T5 has no sequence-start representation, so hidden states '
                           'are pooled by mean. See --t5-pooling.'),

    # -- multi-vector ------------------------------------------------------
    'me-bert':      Spec(MEBERT, 'bert',
                         'One query vector, m document vectors, max inner product',
                         note='Luan et al., TACL 2021. Our implementation; see '
                              'docs/models.md for the reproduction choices.'),
    'poly-encoder': Spec(PolyEncoder, 'bert',
                         'm learned codes attend over the query',
                         note='Humeau et al., ICLR 2020. Our implementation; see '
                              'docs/models.md for the reproduction choices.'),
}

MODEL_CHOICES: Tuple[str, ...] = tuple(sorted(REGISTRY))


def _extra_encoders() -> Dict[str, str]:
    """
    Encoders registered through the environment, as a JSON object mapping a
    short name to a path or hub id:

        IR_BASELINES_ENCODERS='{"my-bert": "/models/my-bert"}'

    This exists because the short names are a convenience, not a restriction.
    A locally fine-tuned or domain-specific encoder should be usable without
    editing the package, and it has to survive into a subprocess, so an
    environment variable rather than a runtime call.
    """
    raw = os.environ.get('IR_BASELINES_ENCODERS')
    if not raw:
        return {}
    try:
        extra = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f'IR_BASELINES_ENCODERS is not valid JSON: {e}') from None
    if not isinstance(extra, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra.items()):
        raise ValueError('IR_BASELINES_ENCODERS must be a JSON object of '
                         '{"name": "path or hub id"}')
    return extra


ENCODER_MAP.update(_extra_encoders())


def resolve_encoder(key: str) -> str:
    """
    A short name, a filesystem path, or a hub id.

    Short names are checked first, then anything that looks like a directory
    on disk. Anything else is passed through to transformers, which will
    report its own error if the id does not exist -- a better message than
    this function could give.
    """
    if key in ENCODER_MAP:
        return ENCODER_MAP[key]
    if os.path.isdir(key):
        return key
    if '/' in key or os.sep in key:
        return key                      # a hub id such as 'microsoft/deberta-base'
    raise KeyError(
        f'unknown --pretrain {key!r}. Known names: {", ".join(sorted(ENCODER_MAP))}. '
        f'A filesystem path or a hub id may also be given, or names added through '
        f'IR_BASELINES_ENCODERS.')


def spec_for(model: str) -> Spec:
    try:
        return REGISTRY[model]
    except KeyError:
        raise KeyError(
            f'unknown model {model!r}. Available: {", ".join(MODEL_CHOICES)}.'
        ) from None


def build_model(args) -> BaselineRanker:
    """
    Construct the model named by --model.

    `--pretrain` defaults to the registry entry's encoder, so the usual
    configuration is what you get without passing it.
    """
    spec = spec_for(args.model)
    pretrained = resolve_encoder(getattr(args, 'pretrain', None) or spec.encoder)

    if spec.cls is CrossEncoder:
        return CrossEncoder(
            pretrained=pretrained,
            t5_pooling=getattr(args, 't5_pooling', 'mean-all'),
        )

    common = dict(
        pretrained=pretrained,
        logit_scale=getattr(args, 'logit_scale', False),
        validate_inputs=not getattr(args, 'no_validate_inputs', False),
    )
    # shared_encoder is left at each class's own default unless explicitly
    # set, so ME-BERT keeps its tied towers and Poly-encoder its separate ones.
    if getattr(args, 'shared_encoder', None) is not None:
        common['shared_encoder'] = args.shared_encoder

    if spec.cls is MEBERT:
        return MEBERT(m=args.me_bert_m, proj_dim=args.me_bert_proj_dim, **common)
    if spec.cls is PolyEncoder:
        return PolyEncoder(poly_m=args.poly_m, **common)

    raise AssertionError(f'registry entry for {args.model!r} has no constructor branch')


def describe() -> str:
    """A table of what is available, for --list-models."""
    width = max(len(k) for k in REGISTRY)
    head = (f'{"model":<{width}}  {"encoding":<8}  {"loss":<13}  {"in-batch":<8}  summary\n'
            f'{"-" * width}  {"-" * 8}  {"-" * 13}  {"-" * 8}  {"-" * 52}')
    lines = [head]
    for name in MODEL_CHOICES:
        s = REGISTRY[name]
        lines.append(
            f'{name:<{width}}  {s.cls.ENCODING:<8}  {s.cls.LOSS:<13}  '
            f'{str(s.cls.SUPPORTS_INBATCH):<8}  {s.summary}')
        if s.note:
            lines.append(f'{"":<{width}}  {s.note}')
    lines.append('')
    lines.append('encoding  how the pair is tokenized: pair = one sequence, '
                 'dual = query and document separately')
    lines.append('in-batch  whether --loss ce-inbatch is available')
    return '\n'.join(lines)