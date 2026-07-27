"""
One dataset for both model families.

The encoding mode comes from the model's ENCODING attribute rather than from
the model's name:

    'pair'  the query and document are tokenized together as one sequence,
            giving input_ids / attention_mask / token_type_ids
    'dual'  they are tokenized separately, giving query_input_ids /
            query_attention_mask and doc_input_ids / doc_attention_mask

Input is one JSON object per line:

    train mode : {'query', 'doc', 'label'} and optionally 'query_id'
    eval mode  : {'query_id', 'doc_id', 'query', 'doc', 'label'}
"""

import json
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ENCODING_CHOICES = ('pair', 'dual')

#: Which tensor-valued keys each encoding produces. `collate` stacks exactly
#: these, so a model that asks for a key the encoding does not emit fails at
#: batch construction rather than deep inside the forward pass.
TENSOR_KEYS = {
    'pair': ('input_ids', 'attention_mask', 'token_type_ids'),
    'dual': ('query_input_ids', 'query_attention_mask',
             'doc_input_ids', 'doc_attention_mask'),
}


class RankingDataset(Dataset):
    """
    `query_id` is carried through in train mode when present, because in-batch
    cross-entropy needs it to avoid scoring a second positive for the same
    query as a negative.

    `positives_only` drops label <= 0 examples at load time, for in-batch
    training where the negatives come from the rest of the batch.

    `binary_labels` rejects graded labels. Both objectives here need a label in
    {0, 1}: BCEWithLogitsLoss silently accepts a target of 2 and optimises
    towards it, and cross-entropy over a two-way classifier would index out of
    range. A graded qrels value leaking into the training file is a mistake
    worth catching at load time either way.
    """

    def __init__(
            self,
            dataset: str,
            tokenizer,
            train: bool,
            encoding: str = 'pair',
            max_len: int = 512,
            max_query_len: int = 20,
            max_doc_len: int = 512,
            positives_only: bool = False,
            binary_labels: bool = False,
    ) -> None:
        if encoding not in ENCODING_CHOICES:
            raise ValueError(
                f'encoding must be one of {ENCODING_CHOICES}, received {encoding!r}')
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._train = train
        self._encoding = encoding
        self._max_len = max_len
        self._max_query_len = max_query_len
        self._max_doc_len = max_doc_len
        self._positives_only = positives_only
        self._binary_labels = binary_labels
        self._read_data()
        self._count = len(self._examples)

    # -- loading -----------------------------------------------------------

    def _read_data(self) -> None:
        with open(self._dataset, 'r') as f:
            examples = [json.loads(line) for line in tqdm(f, desc='reading')]

        if self._positives_only:
            before = len(examples)
            examples = [e for e in examples if float(e.get('label', 0)) > 0]
            print(f'positives_only: kept {len(examples)} of {before} examples')

        if not examples:
            raise ValueError(f'no examples loaded from {self._dataset}')

        missing = [k for k in ('query', 'doc', 'label') if k not in examples[0]]
        if missing:
            raise ValueError(
                f'{self._dataset}: first example is missing {missing}. '
                f'Expected one JSON object per line with query, doc and label.')

        if not self._train and 'doc_id' not in examples[0]:
            raise ValueError(
                f'{self._dataset}: evaluation data needs doc_id on every example, '
                f'so that a run file can be written.')

        if self._binary_labels:
            bad = {float(e['label']) for e in examples} - {0.0, 1.0}
            if bad:
                raise ValueError(
                    f'this objective requires labels in {{0, 1}}; {self._dataset} '
                    f'also contains {sorted(bad)}. Graded relevance values must '
                    f'be binarised before training, not passed through.'
                )

        # All-or-none. `collate` decides whether to emit query_id from the
        # first example in the batch, so a partially populated file either
        # raises KeyError or silently discards the ids depending on shuffle
        # order -- a bug whose symptom depends on the seed.
        presence = [('query_id' in e) for e in examples]
        if any(presence) and not all(presence):
            n = sum(presence)
            raise ValueError(
                f'{self._dataset}: query_id is present on {n} of {len(examples)} '
                f'examples. It must be on every example or none.'
            )

        self._examples = examples
        self._has_query_id = all(presence)
        if self._train and not self._has_query_id:
            print('NOTE  training file carries no query_id.')

    # -- tokenizing --------------------------------------------------------

    def _encode_text(self, text, text_pair, max_len) -> Tuple[List[int], List[int], List[int]]:
        # The __call__ form rather than encode_plus: encode_plus was removed in
        # transformers v5, so calling it pins this code to v4 without saying so.
        encoded = self._tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,          # Add '[CLS]' and '[SEP]'
            max_length=max_len,               # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        input_ids = list(encoded['input_ids'])
        attention_mask = list(encoded['attention_mask'])
        # T5 and DistilBERT tokenizers emit no token_type_ids, and neither
        # model uses them. Return zeros so the shape is the same for every
        # encoder.
        token_type_ids = encoded.get('token_type_ids')
        token_type_ids = list(token_type_ids) if token_type_ids is not None \
            else [0] * len(input_ids)
        return input_ids, attention_mask, token_type_ids

    def _encode(self, example) -> Dict[str, Any]:
        if self._encoding == 'pair':
            input_ids, attention_mask, token_type_ids = self._encode_text(
                example['query'], example['doc'], self._max_len)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

        q_ids, q_mask, _ = self._encode_text(example['query'], None, self._max_query_len)
        d_ids, d_mask, _ = self._encode_text(example['doc'], None, self._max_doc_len)
        return {
            'query_input_ids': q_ids,
            'query_attention_mask': q_mask,
            'doc_input_ids': d_ids,
            'doc_attention_mask': d_mask,
        }

    # -- interface ---------------------------------------------------------

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        item = self._encode(example)
        item['label'] = example['label']
        if 'query_id' in example:
            item['query_id'] = example['query_id']
        if not self._train:
            item['doc_id'] = example['doc_id']
        return item

    def collate(self, batch):
        out = {k: torch.tensor([b[k] for b in batch])
               for k in TENSOR_KEYS[self._encoding]}
        # Left as the dtype the file provides; the trainer casts to what the
        # objective needs, since that differs between them.
        out['label'] = torch.tensor([b['label'] for b in batch])
        if 'query_id' in batch[0]:
            out['query_id'] = [b['query_id'] for b in batch]
        if not self._train:
            out['doc_id'] = [b['doc_id'] for b in batch]
        return out