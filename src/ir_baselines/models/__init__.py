"""
Model registry.

--model selects a class; --pretrain selects the encoder it wraps. The
cross-encoder family is one class over several encoders, so the nine
cross-encoder rows in the published tables all come from CrossEncoder with a
different --pretrain.
"""

from .base import BaselineRanker, mean_all, masked_mean
from .cross_encoder import CrossEncoder
from .multi_vector import MEBERT, PolyEncoder

MODEL_REGISTRY = {
    'cross-encoder': CrossEncoder,
    'me-bert': MEBERT,
    'poly-encoder': PolyEncoder,
}

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


def build_model(args):
    """Construct a model from parsed arguments."""
    pretrained = ENCODER_MAP[args.pretrain]
    if args.model == 'cross-encoder':
        return CrossEncoder(pretrained=pretrained, t5_pooling=args.t5_pooling)
    if args.model == 'me-bert':
        return MEBERT(pretrained=pretrained, m=args.me_bert_m)
    if args.model == 'poly-encoder':
        return PolyEncoder(pretrained=pretrained, poly_m=args.poly_m)
    raise ValueError(f'unknown model {args.model!r}')


def add_model_args(parser):
    """Arguments shared by train.py and test.py, defined once."""
    parser.add_argument('--model', required=True, choices=sorted(MODEL_REGISTRY),
                        help='Model family. The nine cross-encoder rows in the '
                             'published tables are cross-encoder with different '
                             '--pretrain.')
    parser.add_argument('--pretrain', default='bert', choices=sorted(ENCODER_MAP),
                        help='Pretrained encoder. Default: bert.')
    parser.add_argument('--t5-pooling', default='mean-all',
                        choices=('mean-all', 'masked-mean'),
                        help='How T5 hidden states are pooled. mean-all averages '
                             'over every position including padding and is what '
                             'produced the published runs; masked-mean excludes '
                             'padding. Applies to --pretrain t5 only. '
                             'Default: mean-all.')
    parser.add_argument('--poly-m', default=16, type=int,
                        help='Poly-encoder m. Default: 16')
    parser.add_argument('--me-bert-m', default=8, type=int,
                        help='ME-BERT m. Default: 8')
    return parser
