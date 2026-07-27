"""
Shared construction of tokenizer, dataset and model.

train.py and test.py both go through here, so a setting cannot drift between
them. That is not hypothetical: an earlier version of this code had a
--max-doc-len default of 512 in training and 250 in inference, which
truncated documents at inference time without any error.

The architecture configuration assembled here travels inside the checkpoint
and is verified when it is loaded, because several of these settings change
what the weights mean while still loading with every key matched.
"""

import argparse
import sys
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer

from .data import RankingDataset
from .models import (ENCODER_MAP, MODEL_CHOICES, T5_POOLING_CHOICES, build_model,
                     describe, resolve_encoder, spec_for)
from .trainer import LOSS_CHOICES

#: Settings that change what the weights mean. A checkpoint trained under one
#: set of these cannot be evaluated under another, and load_state_dict will not
#: say so: a tied-tower checkpoint loads into a separate-tower model with every
#: key matched, silently overwriting one tower with the other.
ARCHITECTURE_KEYS: Tuple[str, ...] = (
    'model', 'pretrained', 'encoding',
    't5_pooling',
    'shared_encoder', 'tie_projections', 'logit_scale',
    'poly_m', 'me_bert_m', 'me_bert_proj_dim',
    'max_len', 'max_query_len', 'max_doc_len',
)


def _positive_int(name):
    def check(value):
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{name} must be an integer, got {value!r}')
        if v < 1:
            raise argparse.ArgumentTypeError(f'{name} must be >= 1, got {v}')
        return v
    return check


def _bool_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ('1', 'true', 'yes', 'y', 't'):
        return True
    if v in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {value!r}')


def handle_list_models() -> None:
    """
    Print the model table and exit, before argparse can complain that --model
    is missing. Called at the top of main() in both scripts.
    """
    if '--list-models' in sys.argv:
        print(describe())
        raise SystemExit(0)


def add_common_args(parser):
    """Arguments shared by train.py and test.py, defined in exactly one place."""
    parser.add_argument('--list-models', action='store_true',
                        help='Print the available models and exit.')
    parser.add_argument('--model', required=True, type=str, choices=MODEL_CHOICES,
                        help=f'Model to run: {" | ".join(MODEL_CHOICES)}. '
                             f'Use --list-models for details.')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Pretrained encoder, overriding the model default. A '
                             f'short name ({" | ".join(sorted(ENCODER_MAP))}), a path '
                             'to a local model directory, or a hub id. Further short '
                             'names can be registered through IR_BASELINES_ENCODERS.')

    # -- architecture ------------------------------------------------------
    parser.add_argument('--t5-pooling', default='mean-all', choices=T5_POOLING_CHOICES,
                        help='How T5 hidden states are pooled. mean-all averages over '
                             'every position including padding; masked-mean excludes '
                             'padding. Applies to a T5 encoder only. Default: mean-all.')
    parser.add_argument('--poly-m', default=16, type=_positive_int('--poly-m'),
                        help='Poly-encoder m (paper uses 16, 64, 360). Default: 16')
    parser.add_argument('--me-bert-m', default=8, type=_positive_int('--me-bert-m'),
                        help='ME-BERT m (paper uses 8). Default: 8')
    parser.add_argument('--me-bert-proj-dim', default=None,
                        type=_positive_int('--me-bert-proj-dim'),
                        help='ME-BERT-k down-projection dim. Default: None (ME-BERT-768).')
    parser.add_argument('--shared-encoder', default=None, type=_bool_arg,
                        help='Tie the query and document towers. Reproduction choice; '
                             'ME-BERT defaults to tied, Poly-encoder to separate.')
    parser.add_argument('--logit-scale', action='store_true',
                        help='Add a learnable affine calibration a*s+b to the score. '
                             'Recommended with --loss bce; an extension, not a '
                             'reproduction setting.')
    parser.add_argument('--no-validate-inputs', action='store_true',
                        help='Skip the all-padding input check. It forces a host-device '
                             'sync each step; disable only after a smoke run.')

    # -- lengths -----------------------------------------------------------
    parser.add_argument('--max-len', default=512, type=_positive_int('--max-len'),
                        help='Max combined length for pair encoding. Default: 512')
    parser.add_argument('--max-query-len', default=20, type=_positive_int('--max-query-len'),
                        help='Max query length for dual encoding. Default: 20')
    parser.add_argument('--max-doc-len', default=512, type=_positive_int('--max-doc-len'),
                        help='Max document length for dual encoding. Default: 512')

    # -- runtime -----------------------------------------------------------
    parser.add_argument('--batch-size', default=8, type=_positive_int('--batch-size'),
                        help='Batch size. Default: 8')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers. Default: 0')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default: 42')
    parser.add_argument('--amp', choices=('none', 'fp16', 'bf16'), default='none',
                        help='Mixed precision dtype for TRAINING. Inference defaults to '
                             'full precision regardless. Default: none.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA. Default: False')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number. Default: 0')
    return parser


def validate_common_args(parser, args):
    """Checks that need to see more than one argument at a time."""
    spec = spec_for(args.model)
    encoding = spec.cls.ENCODING

    if args.num_workers < 0:
        parser.error('--num-workers must be >= 0')
    if encoding == 'dual' and args.max_query_len < 4:
        parser.error('--max-query-len must leave room for [CLS], a token and [SEP]')
    if encoding == 'pair' and args.max_len < 8:
        parser.error('--max-len must leave room for a query, a document and the '
                     'special tokens')

    # Report ignored settings rather than letting them look effective.
    if encoding == 'pair':
        if args.logit_scale:
            print(f'NOTE  --logit-scale has no effect on {args.model}, which scores '
                  f'through a classifier rather than an inner product.')
        if args.shared_encoder is not None:
            print(f'NOTE  --shared-encoder has no effect on {args.model}, which uses '
                  f'a single encoder over the concatenated pair.')
    else:
        if args.t5_pooling != 'mean-all':
            print(f'NOTE  --t5-pooling has no effect on {args.model}.')

    if args.model != 'poly-encoder' and args.poly_m != 16:
        print('NOTE  --poly-m is ignored except for poly-encoder.')
    if args.model != 'me-bert':
        if args.me_bert_m != 8:
            print('NOTE  --me-bert-m is ignored except for me-bert.')
        if args.me_bert_proj_dim is not None:
            print('NOTE  --me-bert-proj-dim is ignored except for me-bert.')


def default_loss(args) -> str:
    """The objective the chosen model expects, unless --loss overrides it."""
    return spec_for(args.model).cls.LOSS


def validate_loss(parser, args, loss: str) -> None:
    if loss not in LOSS_CHOICES:
        parser.error(f'--loss must be one of {LOSS_CHOICES}')
    cls = spec_for(args.model).cls
    if loss == 'ce-inbatch' and not cls.SUPPORTS_INBATCH:
        parser.error(
            f'--loss ce-inbatch is not available for {args.model}: it encodes the '
            f'query and document together, so there are no separate '
            f'representations to cross-score.')
    if cls.ENCODING == 'pair' and loss != 'cross-entropy':
        parser.error(
            f'{args.model} scores through a two-way classifier, so its objective is '
            f'cross-entropy. --loss {loss} cannot consume its output.')
    if cls.ENCODING == 'dual' and loss == 'cross-entropy':
        parser.error(
            f'{args.model} emits a single score per pair. Use --loss bce or '
            f'--loss ce-inbatch.')


def amp_dtype(name: str):
    import torch
    return {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[name]


def build_tokenizer(args) -> Tuple[Any, str]:
    """Returns (tokenizer, pretrained_name)."""
    spec = spec_for(args.model)
    pretrained = resolve_encoder(args.pretrain or spec.encoder)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    return tokenizer, pretrained


def build(args, tokenizer, pretrained):
    """Construct the model and resize its embeddings if the vocabulary grew."""
    model = build_model(args)
    model.resize_if_needed(len(tokenizer))
    return model


def make_dataset(args, path: str, tokenizer, train: bool, model,
                 positives_only: bool = False, binary_labels: bool = False):
    """A dataset encoded the way the model expects."""
    return RankingDataset(
        dataset=path,
        tokenizer=tokenizer,
        train=train,
        encoding=model.ENCODING,
        max_len=args.max_len,
        max_query_len=args.max_query_len,
        max_doc_len=args.max_doc_len,
        positives_only=positives_only,
        binary_labels=binary_labels,
    )


def architecture_config(args, model, pretrained) -> Dict[str, Any]:
    """
    The settings that must match between training and inference.

    Built from the model's own config_dict() plus the length settings, which
    are not model attributes but do change what the weights mean: a model
    trained at 512 tokens and evaluated at 250 sees truncated documents and
    reports a lower figure with no indication why.
    """
    config = dict(model.config_dict())
    config['encoding'] = model.ENCODING
    if model.ENCODING == 'pair':
        config['max_len'] = args.max_len
    else:
        config['max_query_len'] = args.max_query_len
        config['max_doc_len'] = args.max_doc_len
    return config