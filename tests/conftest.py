"""
Shared fixtures.

Two principles here. Nothing downloads: the encoder tests build a tiny BERT
from a config rather than fetching one, so the suite runs offline and in
seconds. And nothing global is stubbed: `transformers` stays real, because
`test_provenance` checks that the environment record names the actual version
in use, and a stubbed module would make that assertion pass while recording
something false.
"""

import json
import os
import subprocess

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: trains a model; deselect with -m "not slow"')


# ===========================================================  data files

@pytest.fixture
def make_jsonl(tmp_path):
    """Factory: a list of dicts in, a path to a JSONL file out."""
    counter = {'n': 0}

    def _make(rows, name=None):
        counter['n'] += 1
        path = tmp_path / (name or f'data{counter["n"]}.jsonl')
        with open(path, 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        return str(path)

    return _make


@pytest.fixture
def train_jsonl(make_jsonl):
    """
    Training shape: no query_id, no doc_id.

    Four rows with exactly ONE positive. Four so that a shuffled order is
    distinguishable and `drop_last` has something to drop; one positive so
    that `positives_only` has an unambiguous expected size.
    """
    return make_jsonl([
        {'query': 'alpha beta', 'doc': 'gamma delta epsilon', 'label': 1},
        {'query': 'alpha beta', 'doc': 'zeta eta theta', 'label': 0},
        {'query': 'iota kappa', 'doc': 'lambda mu nu', 'label': 0},
        {'query': 'iota kappa', 'doc': 'xi omicron pi', 'label': 0},
    ], name='train.jsonl')


@pytest.fixture
def eval_jsonl(make_jsonl):
    """Evaluation shape: query_id and doc_id present, as a run file needs."""
    return make_jsonl([
        {'query_id': 'q1', 'doc_id': 'd1', 'query': 'alpha beta',
         'doc': 'gamma delta', 'label': 1},
        {'query_id': 'q1', 'doc_id': 'd2', 'query': 'alpha beta',
         'doc': 'zeta eta', 'label': 0},
        {'query_id': 'q2', 'doc_id': 'd3', 'query': 'iota kappa',
         'doc': 'lambda mu', 'label': 1},
        {'query_id': 'q2', 'doc_id': 'd4', 'query': 'iota kappa',
         'doc': 'xi omicron', 'label': 0},
    ], name='eval.jsonl')


VOCAB = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + \
    'alpha beta gamma delta epsilon zeta eta theta iota kappa ' \
    'lambda mu nu xi omicron pi rho sigma tau upsilon'.split()


@pytest.fixture(scope='session')
def corpus(tmp_path_factory):
    """
    A small collection: train, dev, test and qrels over six topics.

    Session-scoped because several end-to-end tests train against it and
    rebuilding it per test would dominate the runtime.
    """
    import random
    rng = random.Random(0)
    words = VOCAB[5:]
    d = tmp_path_factory.mktemp('corpus')

    def text(n):
        return ' '.join(rng.choice(words) for _ in range(n))

    topics = [f't{i}' for i in range(6)]
    paths = {}
    for split in ('train', 'dev', 'test'):
        p = d / f'{split}.jsonl'
        with open(p, 'w') as f:
            for t in topics:
                for j in range(4):
                    f.write(json.dumps({
                        'query_id': t, 'doc_id': f'{t}_d{j}',
                        'query': text(3), 'doc': text(20),
                        'label': 1 if j == 0 else 0,
                    }) + '\n')
        paths[split] = str(p)

    q = d / 'qrels.txt'
    with open(q, 'w') as f:
        for t in topics:
            for j in range(4):
                f.write(f'{t} 0 {t}_d{j} {1 if j == 0 else 0}\n')
    paths['qrels'] = str(q)
    paths['topics'] = len(topics)
    return paths


# ===========================================================  tokenizers

class _StubTokenizer:
    """
    Enough of a tokenizer for the dataset to be exercised without weights.

    `emits_token_type` is the case that matters: T5 and DistilBERT tokenizers
    return no token_type_ids, and the dataset has to supply zeros rather than
    raise.
    """

    def __init__(self, emits_token_type=True):
        self.emits_token_type = emits_token_type

    def __call__(self, text=None, text_pair=None, max_length=16, **kw):
        out = {'input_ids': list(range(max_length)),
               'attention_mask': [1] * max_length}
        if self.emits_token_type:
            out['token_type_ids'] = [0] * max_length
        return out

    def __len__(self):
        return len(VOCAB)


@pytest.fixture
def stub_tokenizer():
    return _StubTokenizer()


@pytest.fixture
def stub_tokenizer_no_token_type():
    return _StubTokenizer(emits_token_type=False)


# ========================================================  tiny encoder

@pytest.fixture(scope='session')
def tiny_encoder(tmp_path_factory):
    """
    A real but very small BERT, built from a config rather than downloaded.

    Registered as the short name `tiny` through IR_BASELINES_ENCODERS, because
    the end-to-end tests invoke the entry points in a subprocess and an
    in-process registration would not reach them.

    Real rather than stubbed: these tests exercise tokenization, the forward
    pass, checkpoint saving and loading. A stub would leave all of that
    untested while appearing to pass.
    """
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = tmp_path_factory.mktemp('tiny_encoder')
    with open(d / 'vocab.txt', 'w') as f:
        f.write('\n'.join(VOCAB) + '\n')

    config = BertConfig(vocab_size=len(VOCAB), hidden_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128)
    BertModel(config).save_pretrained(d)
    BertTokenizerFast(vocab_file=str(d / 'vocab.txt')).save_pretrained(d)

    existing = os.environ.get('IR_BASELINES_ENCODERS')
    registry = json.loads(existing) if existing else {}
    registry['tiny'] = str(d)
    os.environ['IR_BASELINES_ENCODERS'] = json.dumps(registry)

    yield str(d)

    if existing is None:
        os.environ.pop('IR_BASELINES_ENCODERS', None)
    else:
        os.environ['IR_BASELINES_ENCODERS'] = existing


# ================================================================  git

@pytest.fixture
def git_repo(tmp_path):
    """A repository with one commit and one tracked file, `a.txt`."""
    repo = tmp_path / 'repo'
    repo.mkdir()

    def git(*args):
        subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)

    git('init', '-q', '.')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'test')
    # Untracked build artefacts must not count as a dirty tree, so the
    # repository is configured the way a real one would be.
    (repo / '.gitignore').write_text('__pycache__/\n*.pyc\n')
    (repo / 'a.txt').write_text('one\n')
    git('add', '-A')
    git('commit', '-qm', 'first')
    return str(repo)
