"""
The unified dataset: both encoding modes, the token_type_ids fallback, and the
validation that refuses data which would otherwise train silently against the
wrong thing.
"""

import pytest
import torch

from ir_baselines.data import RankingDataLoader, RankingDataset, TENSOR_KEYS


# ==========================================================  encoding

def test_pair_mode_emits_its_keys(eval_jsonl, stub_tokenizer):
    ds = RankingDataset(eval_jsonl, stub_tokenizer, train=False,
                        encoding='pair', max_len=16)
    item = ds[0]
    assert set(TENSOR_KEYS['pair']) <= set(item)
    assert 'query_input_ids' not in item


def test_dual_mode_emits_its_keys(eval_jsonl, stub_tokenizer):
    ds = RankingDataset(eval_jsonl, stub_tokenizer, train=False,
                        encoding='dual', max_query_len=8, max_doc_len=16)
    item = ds[0]
    assert set(TENSOR_KEYS['dual']) <= set(item)
    assert 'input_ids' not in item


def test_dual_mode_honours_both_lengths(eval_jsonl, stub_tokenizer):
    ds = RankingDataset(eval_jsonl, stub_tokenizer, train=False,
                        encoding='dual', max_query_len=8, max_doc_len=16)
    item = ds[0]
    assert len(item['query_input_ids']) == 8
    assert len(item['doc_input_ids']) == 16


def test_token_type_ids_default_to_zeros(eval_jsonl, stub_tokenizer_no_token_type):
    """
    T5 and DistilBERT tokenizers emit none. Without the fallback, the
    published T5 configuration raises KeyError on the first batch.
    """
    ds = RankingDataset(eval_jsonl, stub_tokenizer_no_token_type, train=False,
                        encoding='pair', max_len=16)
    assert ds[0]['token_type_ids'] == [0] * 16


def test_unknown_encoding_is_rejected(eval_jsonl, stub_tokenizer):
    with pytest.raises(ValueError, match='encoding must be'):
        RankingDataset(eval_jsonl, stub_tokenizer, train=False, encoding='triple')


# ===========================================================  collate

def test_eval_batch_carries_ids(eval_jsonl, stub_tokenizer):
    ds = RankingDataset(eval_jsonl, stub_tokenizer, train=False,
                        encoding='pair', max_len=16)
    batch = ds.collate([ds[0], ds[1]])
    assert batch['doc_id'] == ['d1', 'd2']
    assert batch['query_id'] == ['q1', 'q1']
    assert tuple(batch['input_ids'].shape) == (2, 16)


def test_train_batch_omits_ids(train_jsonl, stub_tokenizer):
    ds = RankingDataset(train_jsonl, stub_tokenizer, train=True,
                        encoding='pair', max_len=16)
    batch = ds.collate([ds[0], ds[1]])
    assert 'doc_id' not in batch
    assert 'query_id' not in batch


@pytest.mark.parametrize('encoding', ['pair', 'dual'])
def test_collate_stacks_exactly_the_encoding_keys(eval_jsonl, stub_tokenizer, encoding):
    ds = RankingDataset(eval_jsonl, stub_tokenizer, train=False,
                        encoding=encoding, max_len=16, max_query_len=8, max_doc_len=16)
    batch = ds.collate([ds[0], ds[1]])
    for key in TENSOR_KEYS[encoding]:
        assert isinstance(batch[key], torch.Tensor)
    other = 'dual' if encoding == 'pair' else 'pair'
    for key in TENSOR_KEYS[other]:
        assert key not in batch


# ========================================================  validation

def test_graded_labels_rejected_under_a_pointwise_objective(make_jsonl, stub_tokenizer):
    """
    BCEWithLogitsLoss accepts a target of 2 and optimises towards it, so a
    graded qrels value leaking into the training file trains the wrong
    objective without complaint.
    """
    path = make_jsonl([{'query': 'a', 'doc': 'b', 'label': 2}])
    with pytest.raises(ValueError, match='labels in'):
        RankingDataset(path, stub_tokenizer, train=True, encoding='pair',
                       binary_labels=True)


def test_partial_query_id_rejected(make_jsonl, stub_tokenizer):
    """
    collate decides whether to emit query_id from the first example in the
    batch, so a partially populated file fails differently depending on the
    shuffle seed.
    """
    path = make_jsonl([{'query_id': 'q', 'query': 'a', 'doc': 'b', 'label': 1},
                       {'query': 'a', 'doc': 'c', 'label': 0}])
    with pytest.raises(ValueError, match='query_id is present on'):
        RankingDataset(path, stub_tokenizer, train=True, encoding='pair')


def test_eval_data_without_doc_id_rejected(train_jsonl, stub_tokenizer):
    with pytest.raises(ValueError, match='doc_id'):
        RankingDataset(train_jsonl, stub_tokenizer, train=False, encoding='pair')


def test_missing_required_field_rejected(make_jsonl, stub_tokenizer):
    path = make_jsonl([{'query': 'a', 'label': 1}])
    with pytest.raises(ValueError, match='missing'):
        RankingDataset(path, stub_tokenizer, train=True, encoding='pair')


def test_empty_file_rejected(make_jsonl, stub_tokenizer):
    path = make_jsonl([])
    with pytest.raises(ValueError, match='no examples'):
        RankingDataset(path, stub_tokenizer, train=True, encoding='pair')


def test_positives_only_drops_negatives(train_jsonl, stub_tokenizer):
    ds = RankingDataset(train_jsonl, stub_tokenizer, train=True,
                        encoding='pair', positives_only=True)
    assert len(ds) == 1


# ========================================================  dataloader

def test_same_seed_gives_the_same_order(train_jsonl, stub_tokenizer):
    ds = RankingDataset(train_jsonl, stub_tokenizer, train=True,
                        encoding='pair', max_len=16)

    def order(seed):
        loader = RankingDataLoader(ds, batch_size=1, shuffle=True, seed=seed,
                                   pin_memory=False)
        return [b['label'].tolist() for b in loader]

    assert order(7) == order(7)


def test_drop_last_leaves_no_ragged_batch(train_jsonl, stub_tokenizer):
    """In-batch negatives need a full batch; a ragged final one has no
    negatives to draw on."""
    ds = RankingDataset(train_jsonl, stub_tokenizer, train=True,
                        encoding='pair', max_len=16)
    loader = RankingDataLoader(ds, batch_size=2, drop_last=True, pin_memory=False)
    assert all(b['label'].size(0) == 2 for b in loader)
