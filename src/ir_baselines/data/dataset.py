"""
One dataset for both model families.

The encoding mode comes from the model's ENCODING attribute:

    'pair'  the query and document are tokenized together, giving
            input_ids / attention_mask / token_type_ids.
    'dual'  they are tokenized separately, giving query_* and doc_*.

Input is one JSON object per line:

    train mode  {'query', 'doc', 'label'}
    eval mode   {'query_id', 'doc_id', 'query', 'doc', 'label'}
"""

import json
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RankingDataset(Dataset):
    def __init__(self, dataset, tokenizer, train, encoding='pair',
                 max_len=512, max_query_len=20, max_doc_len=512):
        if encoding not in ('pair', 'dual'):
            raise ValueError(f"encoding must be 'pair' or 'dual', got {encoding!r}")
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._train = train
        self._encoding = encoding
        self._max_len = max_len
        self._max_query_len = max_query_len
        self._max_doc_len = max_doc_len
        self._read_data()
        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for line in tqdm(f, desc='reading')]
        if not self._examples:
            raise ValueError(f'no examples loaded from {self._dataset}')

    def _encode(self, text, text_pair=None, max_len=None):
        # The __call__ form rather than encode_plus: encode_plus was removed in
        # transformers v5. The arguments and the result are the same, so this
        # changes no token and no score.
        encoded = self._tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=max_len or self._max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        input_ids = list(encoded['input_ids'])
        attention_mask = list(encoded['attention_mask'])
        # T5 and DistilBERT tokenizers emit no token_type_ids, and neither
        # model uses them. Return zeros so the three-value shape holds for
        # every encoder.
        token_type_ids = encoded.get('token_type_ids')
        token_type_ids = list(token_type_ids) if token_type_ids is not None \
            else [0] * len(input_ids)
        return input_ids, attention_mask, token_type_ids

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]

        if self._encoding == 'pair':
            input_ids, attention_mask, token_type_ids = self._encode(
                example['query'], example['doc'], self._max_len)
            item = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }
        else:
            q_ids, q_mask, _ = self._encode(example['query'], None, self._max_query_len)
            d_ids, d_mask, _ = self._encode(example['doc'], None, self._max_doc_len)
            item = {
                'query_input_ids': q_ids,
                'query_attention_mask': q_mask,
                'doc_input_ids': d_ids,
                'doc_attention_mask': d_mask,
            }

        item['label'] = example['label']
        if not self._train:
            item['query_id'] = example['query_id']
            item['doc_id'] = example['doc_id']
        return item

    def collate(self, batch):
        tensor_keys = [k for k in batch[0] if k not in ('label', 'query_id', 'doc_id')]
        out = {k: torch.tensor([b[k] for b in batch]) for k in tensor_keys}
        # cross_entropy wants class indices, bce wants floats. The label is
        # stored as given and cast by the trainer, which knows the loss.
        out['label'] = torch.tensor([b['label'] for b in batch])
        if not self._train:
            out['query_id'] = [b['query_id'] for b in batch]
            out['doc_id'] = [b['doc_id'] for b in batch]
        return out
