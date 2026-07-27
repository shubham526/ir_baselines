"""
Checks that both encoding modes produce the keys their models expect, and that
the token_type_ids fallback works for tokenizers that do not emit them.

Uses a stub tokenizer, so it runs without downloading anything.
"""

import json
import os
import sys
import tempfile

from ir_baselines.data.dataset import RankingDataset

FAILED = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILED.append(name)


class StubTokenizer:
    """Minimal tokenizer. emits_token_type controls the fallback path."""

    def __init__(self, emits_token_type=True):
        self.emits_token_type = emits_token_type

    def __call__(self, text=None, text_pair=None, max_length=16, **kw):
        n = max_length
        out = {'input_ids': list(range(n)), 'attention_mask': [1] * n}
        if self.emits_token_type:
            out['token_type_ids'] = [0] * n
        return out


def write_jsonl(rows):
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    with os.fdopen(fd, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return path


rows = [{'query_id': 'q1', 'doc_id': 'd1', 'query': 'a query',
         'doc': 'a document', 'label': 1}]
path = write_jsonl(rows)

print('\nEncoding modes')
pair = RankingDataset(path, StubTokenizer(), train=False, encoding='pair', max_len=16)
item = pair[0]
check('pair mode gives input_ids / attention_mask / token_type_ids',
      {'input_ids', 'attention_mask', 'token_type_ids'} <= set(item))
check('pair mode does not give query_input_ids',
      'query_input_ids' not in item)

dual = RankingDataset(path, StubTokenizer(), train=False, encoding='dual',
                      max_query_len=8, max_doc_len=16)
item = dual[0]
check('dual mode gives query_* and doc_*',
      {'query_input_ids', 'query_attention_mask',
       'doc_input_ids', 'doc_attention_mask'} <= set(item))
check('dual mode respects max_query_len', len(item['query_input_ids']) == 8)
check('dual mode respects max_doc_len', len(item['doc_input_ids']) == 16)

print('\ntoken_type_ids fallback')
no_tt = RankingDataset(path, StubTokenizer(emits_token_type=False),
                       train=False, encoding='pair', max_len=16)
item = no_tt[0]
check('zeros supplied when the tokenizer emits none',
      item['token_type_ids'] == [0] * 16)

print('\nCollate')
batch = pair.collate([pair[0], pair[0]])
check('eval batch carries query_id and doc_id',
      'query_id' in batch and 'doc_id' in batch)
check('eval batch tensors are stacked', batch['input_ids'].shape == (2, 16))

train_rows = [{'query': 'q', 'doc': 'd', 'label': 1}]
tpath = write_jsonl(train_rows)
tr = RankingDataset(tpath, StubTokenizer(), train=True, encoding='pair', max_len=16)
tbatch = tr.collate([tr[0], tr[0]])
check('train batch omits query_id and doc_id',
      'query_id' not in tbatch and 'doc_id' not in tbatch)

print('\nInvalid encoding is rejected')
try:
    RankingDataset(path, StubTokenizer(), train=False, encoding='triple')
    check('bad encoding raises', False)
except ValueError:
    check('bad encoding raises', True)

os.unlink(path)
os.unlink(tpath)

print()
print('ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} check(s) failed')
sys.exit(1 if FAILED else 0)