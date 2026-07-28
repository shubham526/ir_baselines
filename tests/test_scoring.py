"""
The multi-vector scoring functions against explicit implementations of the
equations in their papers.

This is the test that has to pass for the scoring to be trustworthy: the
einsums are compact, and both of the defects that reached publication were in
them. Each check builds the paper's expression with plain loops and compares.

No model weights are involved -- the encoder stack is bypassed and only the
tensor algebra is exercised, which is where the bugs were.
"""

import pytest
import torch

from ir_baselines.models.base import _masked_softmax
from ir_baselines.models.multi_vector import MEBERT, PolyEncoder

B, N, Q, M, D, L = 4, 5, 3, 6, 8, 7
TOL = 1e-4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def me():
    """An ME-BERT with no encoder: only the scoring methods are under test."""
    m = MEBERT.__new__(MEBERT)
    m.scale = None
    m.m = M
    return m


@pytest.fixture
def pe():
    p = PolyEncoder.__new__(PolyEncoder)
    p.scale = None
    p.poly_m = M
    return p


# ===========================================================  ME-BERT
# Luan et al., TACL 2021, Section 3:
#   score(x, y) = max_j <f1(x), fm_j(y)>
# one query vector, m document vectors, maximum inner product.

def test_score_pairs_is_max_over_document_vectors(me):
    q = torch.randn(B, D)
    d = torch.randn(B, M, D)
    mask = torch.ones(B, M)
    mask[0, 4:] = 0                     # short documents
    mask[2, 1:] = 0

    got = me.score_pairs(q, d, mask)

    want = torch.empty(B)
    for b in range(B):
        want[b] = max(float(torch.dot(q[b], d[b, j]))
                      for j in range(M) if mask[b, j])
    assert torch.allclose(got, want, atol=TOL)


def test_padding_cannot_win_the_maximum(me):
    """
    A padded slot holding a large value must not be selected. Without the
    mask, an all-padding document returns the masking sentinel and looks like
    a confident negative prediction rather than a malformed batch.
    """
    q = torch.ones(1, D)
    d = torch.zeros(1, M, D)
    d[0, 0] = 1.0                       # real token, score = D
    d[0, 3] = 100.0                     # padding, would score 100 * D
    mask = torch.zeros(1, M)
    mask[0, 0] = 1

    got = me.score_pairs(q, d, mask)
    assert torch.allclose(got, torch.tensor([float(D)]), atol=TOL)


def test_score_matrix_is_all_pairs_max(me):
    q = torch.randn(Q, D)
    d = torch.randn(N, M, D)
    mask = torch.ones(N, M)
    mask[1, 3:] = 0

    got = me.score_matrix(q, d, mask)

    want = torch.empty(Q, N)
    for i in range(Q):
        for n in range(N):
            want[i, n] = max(float(torch.dot(q[i], d[n, j]))
                             for j in range(M) if mask[n, j])
    assert torch.allclose(got, want, atol=TOL)


def test_me_bert_matrix_diagonal_equals_pairs(me):
    """The all-pairs matrix and the aligned-pair scores must agree."""
    q = torch.randn(B, D)
    d = torch.randn(B, M, D)
    mask = torch.ones(B, M)
    assert torch.allclose(me.score_matrix(q, d, mask).diagonal(),
                          me.score_pairs(q, d, mask), atol=TOL)


def test_me_bert_is_not_the_old_contracted_form(me):
    """
    The published rows came from 'bmd,bnd->bm', which contracts the query's m
    as well and is a different quantity. Asserted so a future edit cannot
    quietly reintroduce it.
    """
    q = torch.randn(B, D)
    d = torch.randn(B, M, D)
    mask = torch.ones(B, M)
    old = torch.einsum('bmd,bnd->bm', torch.randn(B, M, D), d).max(1).values
    assert not torch.allclose(old, me.score_pairs(q, d, mask), atol=TOL)


# =======================================================  Poly-encoder
# Humeau et al., ICLR 2020, Section 4.4:
#   y_ctxt^i = sum_j softmax_j(c_i . h_j) h_j
#   score    = y_ctxt . y_cand, after attending over the m context vectors.

def _context_vectors(codes, h, mask):
    scores = torch.einsum('md,bnd->bmn', codes, h)
    m_exp = mask.unsqueeze(1).expand(-1, codes.size(0), -1)
    probs = _masked_softmax(scores, m_exp, dim=-1)
    return torch.einsum('bmn,bnd->bmd', probs, h), probs


def test_context_vectors_are_masked_code_attention():
    h = torch.randn(B, L, D)
    codes = torch.randn(M, D)
    mask = torch.ones(B, L)
    mask[0, 5:] = 0
    mask[3, 2:] = 0

    got, _ = _context_vectors(codes, h, mask)

    want = torch.zeros(B, M, D)
    for b in range(B):
        for i in range(M):
            s = torch.tensor([float(torch.dot(codes[i], h[b, j])) if mask[b, j]
                              else -1e30 for j in range(L)])
            w = torch.softmax(s, 0)
            want[b, i] = sum(w[j] * h[b, j] for j in range(L))
    assert torch.allclose(got, want, atol=TOL)


def test_attention_puts_no_mass_on_padding():
    h = torch.randn(B, L, D)
    codes = torch.randn(M, D)
    mask = torch.ones(B, L)
    mask[0, 5:] = 0

    _, probs = _context_vectors(codes, h, mask)
    assert float(probs[0, :, 5:].sum()) == pytest.approx(0.0, abs=1e-6)


def test_poly_score_pairs_matches_the_paper(pe):
    h = torch.randn(B, L, D)
    codes = torch.randn(M, D)
    y_ctxt, _ = _context_vectors(codes, h, torch.ones(B, L))
    y_cand = torch.randn(B, D)

    got = pe.score_pairs(y_ctxt, y_cand)

    want = torch.empty(B)
    for b in range(B):
        s = torch.tensor([float(torch.dot(y_cand[b], y_ctxt[b, i]))
                          for i in range(M)])
        w = torch.softmax(s, 0)
        final = sum(w[i] * y_ctxt[b, i] for i in range(M))
        want[b] = torch.dot(final, y_cand[b])
    assert torch.allclose(got, want, atol=TOL)


def test_poly_score_matrix_is_all_pairs(pe):
    y_ctxt = torch.randn(Q, M, D)
    y_cand = torch.randn(N, D)

    got = pe.score_matrix(y_ctxt, y_cand)

    want = torch.empty(Q, N)
    for i in range(Q):
        for n in range(N):
            s = torch.tensor([float(torch.dot(y_cand[n], y_ctxt[i, k]))
                              for k in range(M)])
            w = torch.softmax(s, 0)
            final = sum(w[k] * y_ctxt[i, k] for k in range(M))
            want[i, n] = torch.dot(final, y_cand[n])
    assert torch.allclose(got, want, atol=TOL)


def test_poly_matrix_diagonal_equals_pairs(pe):
    y_ctxt = torch.randn(B, M, D)
    y_cand = torch.randn(B, D)
    assert torch.allclose(pe.score_matrix(y_ctxt, y_cand).diagonal(),
                          pe.score_pairs(y_ctxt, y_cand), atol=TOL)


def test_poly_codes_are_not_inert():
    """
    The published rows came from 'bn,bmd->bd', which contracts the code index
    and the sequence index together, making the softmax sum to one across the
    whole tensor and reducing the result to an unweighted sum. Asserted so the
    defect cannot return unnoticed.
    """
    y_ctxt = torch.randn(B, M, D)
    old = torch.einsum('bn,bmd->bd', torch.softmax(torch.randn(B, M), -1), y_ctxt)
    assert torch.allclose(old, y_ctxt.sum(1), atol=TOL), \
        'the old form should reduce to an unweighted sum'

    # the current form must not
    codes = torch.randn(M, D)
    h = torch.randn(B, L, D)
    current, _ = _context_vectors(codes, h, torch.ones(B, L))
    assert not torch.allclose(current, h.sum(1).unsqueeze(1).expand(-1, M, -1),
                              atol=TOL)


# ==========================================================  dtypes
# Mixed precision is where a hard-coded float32 sentinel fails: -3.4e38
# overflows float16 and raises at masked_fill time.

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_scoring_runs_in_every_dtype(me, pe, dtype):
    q = torch.randn(B, D, dtype=dtype)
    d = torch.randn(B, M, D, dtype=dtype)
    mask = torch.ones(B, M)
    mask[0, 3:] = 0
    out = me.score_pairs(q, d, mask)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()

    y_ctxt = torch.randn(B, M, D, dtype=dtype)
    y_cand = torch.randn(B, D, dtype=dtype)
    assert torch.isfinite(pe.score_pairs(y_ctxt, y_cand)).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_masked_softmax_runs_in_every_dtype(dtype):
    scores = torch.randn(B, M, L, dtype=dtype)
    mask = torch.ones(B, M, L)
    mask[..., 4:] = 0
    out = _masked_softmax(scores, mask, dim=-1)
    assert torch.isfinite(out).all()
    # float() first: numpy has no bfloat16, so pytest.approx cannot read the
    # tensor directly.
    assert float(out[..., 4:].sum()) == pytest.approx(0.0, abs=1e-3)


def test_masked_softmax_rejects_a_fully_masked_row():
    """A transformer input always has at least one real token, so this means
    malformed batching rather than a short sequence."""
    with pytest.raises(ValueError, match='fully masked'):
        _masked_softmax(torch.randn(2, 3, 4), torch.zeros(2, 3, 4), dim=-1)
