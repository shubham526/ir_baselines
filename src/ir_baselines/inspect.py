"""
Read the provenance of a checkpoint or a run file.

    python -m ir_baselines.inspect out/model.bin
    python -m ir_baselines.inspect runs/fold-0.run
    python -m ir_baselines.inspect runs/fold-0.run --json

A run file has no provenance of its own -- the TREC format is six fields per
line and every parser rejects a comment -- so this looks for the sibling
`<run>.provenance.json` written at inference time. If the run has been edited
or regenerated since, the digest recorded in the sibling will not match and
that is reported rather than passed over.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch

from . import provenance


def _print_kv(label: str, value: Any, indent: int = 2) -> None:
    if value not in (None, {}, [], ''):
        print(f'{" " * indent}{label:<22} {value}')


def _show_provenance(prov: Dict[str, Any], indent: int = 2) -> None:
    pad = ' ' * indent
    g = prov.get('git', {})
    if g.get('available'):
        mark = '  DIRTY -- the commit does not identify what ran' if g.get('dirty') else ''
        _print_kv('commit', f"{g.get('describe') or g.get('commit', '')[:12]}{mark}", indent)
        _print_kv('branch', g.get('branch'), indent)
        if g.get('dirty_files'):
            print(f'{pad}{"modified":<22} {", ".join(g["dirty_files"][:6])}'
                  f'{" ..." if len(g["dirty_files"]) > 6 else ""}')
    else:
        _print_kv('commit', 'not a git repository', indent)

    e = prov.get('environment', {})
    _print_kv('python', e.get('python'), indent)
    _print_kv('torch', f"{e.get('torch')}  (cuda {e.get('cuda')}, "
                       f"cudnn {e.get('cudnn')})", indent)
    _print_kv('transformers', e.get('transformers'), indent)
    _print_kv('numpy', e.get('numpy'), indent)
    if e.get('gpu'):
        _print_kv('gpu', ', '.join(e['gpu']), indent)
    _print_kv('host', f"{e.get('hostname')}  ({e.get('platform')})", indent)
    _print_kv('created', prov.get('created'), indent)
    _print_kv('command', prov.get('command'), indent)

    data = prov.get('data', {})
    if data:
        print(f'{pad}{"data":<22}')
        for path, d in data.items():
            if d.get('exists'):
                print(f'{pad}  {os.path.basename(path):<28} '
                      f'{d.get("lines", 0):>10,} lines  '
                      f'sha256:{d.get("sha256", "")[:16]}')
            else:
                print(f'{pad}  {os.path.basename(path):<28} MISSING')


def show_checkpoint(path: str) -> None:
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'model_state_dict' not in obj:
        print(f'{path}\n  a bare state_dict: no configuration and no provenance.')
        return

    print(f'{path}\n')
    print('training state')
    _print_kv('epoch', obj.get('epoch'))
    _print_kv('best metric', obj.get('best_metric'))
    for key, label in (('optimizer_state_dict', 'optimizer state'),
                       ('scheduler_state_dict', 'scheduler state'),
                       ('rng_state', 'rng state')):
        _print_kv(label, 'present' if key in obj else None)
    h = obj.get('history')
    if h and h.get('epoch'):
        _print_kv('epochs recorded', f"{h['epoch'][0]}..{h['epoch'][-1]}")
        if h.get('val_metric'):
            _print_kv('val metric', ', '.join(f'{v:.4f}' for v in h['val_metric'][-6:]))

    cfg = obj.get('config', {})
    if cfg:
        print('\nconfiguration')
        for k in ('model', 'pretrained', 'encoding', 'loss', 't5_pooling',
                  'shared_encoder', 'logit_scale', 'poly_m', 'me_bert_m',
                  'max_len', 'max_query_len', 'max_doc_len',
                  'batch_size', 'learning_rate', 'seed', 'optimizer'):
            _print_kv(k, cfg.get(k))
    else:
        print('\nconfiguration          none recorded -- this checkpoint cannot be '
              'verified against inference settings')

    prov = obj.get('provenance')
    if prov:
        print('\nprovenance')
        _show_provenance(prov)
    else:
        print('\nprovenance             none recorded -- the code, environment and '
              'data that produced these weights are not known')


def show_run(path: str) -> None:
    sidecar = path + '.provenance.json'
    print(f'{path}\n')
    stats = provenance.run_stats(path)
    print('run')
    _print_kv('topics', f"{stats['topics']:,}")
    _print_kv('pairs', f"{stats['pairs']:,}")
    _print_kv('mean depth', f"{stats['pairs'] / max(stats['topics'], 1):.1f}")
    _print_kv('sha256', stats['sha256'])

    if not os.path.exists(sidecar):
        print(f'\nNo {os.path.basename(sidecar)} beside this run, so what produced it '
              f'is not recorded.\nA run file cannot carry its own provenance: every '
              f'TREC parser rejects a comment line.')
        return

    with open(sidecar) as f:
        rec = json.load(f)

    recorded = rec.get('run', {}).get('sha256')
    if recorded and recorded != stats['sha256']:
        print(f'\nWARNING  the sibling records sha256:{recorded[:16]} but this file is '
              f'sha256:{stats["sha256"][:16]}.\n         The run has changed since the '
              f'provenance was written, so what follows describes a different file.')

    print('\nproduced by')
    _show_provenance(rec.get('produced_by', {}))

    ck = rec.get('checkpoint')
    if ck:
        print('\ncheckpoint')
        _print_kv('path', ck.get('path'))
        _print_kv('sha256', ck.get('sha256'))
        cfg = ck.get('config', {})
        for k in ('model', 'pretrained', 'encoding', 'loss', 't5_pooling', 'seed'):
            _print_kv(k, cfg.get(k))
        if ck.get('provenance'):
            print('\ncheckpoint was trained by')
            _show_provenance(ck['provenance'])


def main():
    parser = argparse.ArgumentParser(
        'Read the provenance of a checkpoint or a run file.')
    parser.add_argument('path', help='A checkpoint (.bin) or a run file.')
    parser.add_argument('--json', action='store_true',
                        help='Print the raw record instead of a summary.')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f'no such file: {args.path}')

    is_checkpoint = args.path.endswith(('.bin', '.pt', '.pth'))

    if args.json:
        if is_checkpoint:
            obj = torch.load(args.path, map_location='cpu', weights_only=False)
            out = {k: v for k, v in obj.items()
                   if k not in ('model_state_dict', 'optimizer_state_dict',
                                'scheduler_state_dict', 'scaler_state_dict',
                                'rng_state')} if isinstance(obj, dict) else {}
        else:
            sidecar = args.path + '.provenance.json'
            if not os.path.exists(sidecar):
                raise SystemExit(f'no provenance beside {args.path}')
            with open(sidecar) as f:
                out = json.load(f)
        json.dump(out, sys.stdout, indent=2, default=str)
        print()
        return

    if is_checkpoint:
        show_checkpoint(args.path)
    else:
        show_run(args.path)


if __name__ == '__main__':
    main()