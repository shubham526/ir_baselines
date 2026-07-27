"""
Checks the multi-vector scoring functions against explicit implementations of
the expressions given in their papers.

BOTH TESTS FAIL IN THIS RELEASE, BY DESIGN. The ME-BERT and Poly-encoder
implementations here are the ones that produced the published rows, and
neither computes its paper's expression. The failures are the record of that;
see docs/known-issues.md. The revised implementations pass.

The models are instantiated without weights and their encoder is replaced by
a stub returning fixed tensors, so this calls the shipped forward() rather
than a copy of it, and runs in a second with no network access.
"""

import sys
import types

import torch

# Stub transformers before importing the models, so no weights are fetched.
_stub = types.ModuleType('transformers')


class _Cfg:
    hidden_size = 8

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


class _Model:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return torch.nn.Identity()


_stub.AutoConfig = _Cfg
_stub.AutoModel = _Model
_stub.DistilBertModel = type('DistilBertModel', (), {})
_stub.T5EncoderModel = type('T5EncoderModel', (), {})
sys.modules.setdefault('transformers', _stub)

from ir_baselines.models.multi_vector import MEBERT, PolyEncoder  # noqa: E402

torch.manual_seed(0)
B, M, D, L = 4, 6, 8, 7
FAILED = []


def check(name, implemented, paper, tol=1e-4):
    """Passes when the shipped implementation matches the paper expression."""
    d = (implemented - paper).abs().max().item()
    ok = d < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<50} maxdiff={d:.2e}")
    if not ok:
        FAILED.append(name)


def make_encoder(q_seq, d_seq, q_ids, d_ids):
    class _E(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kw):
            if input_ids is q_ids:
                return _Out(q_seq)
            return _Out(d_seq)
    return _E()


class _Out(tuple):
    """Supports both outputs[0] and outputs.last_hidden_state."""

    def __new__(cls, t):
        obj = super().__new__(cls, (t,))
        obj.last_hidden_state = t
        return obj


q_ids = torch.zeros(B, L, dtype=torch.long)
d_ids = torch.ones(B, L, dtype=torch.long)
q_seq = torch.randn(B, L, D)
d_seq = torch.randn(B, L, D)

# =========================================================== ME-BERT
# Luan et al., TACL 2021, Section 3:
#   f^(1)(x)   = h_1(x)                     the query is ONE vector
#   f^(m)(y)   = [h_1(y) ... h_m(y)]        the document is m vectors
#   score(x,y) = max_j <f^(1)(x), f^(m)_j(y)>
print('\nME-BERT   score(x,y) = max_j <f1(x), fm_j(y)>')

me = MEBERT.__new__(MEBERT)
torch.nn.Module.__init__(me)
me.m = M
me.encoder = make_encoder(q_seq, d_seq, q_ids, d_ids)

with torch.no_grad():
    implemented = me.forward(q_ids, torch.ones_like(q_ids), d_ids, torch.ones_like(d_ids))

paper = torch.empty(B)
for b in range(B):
    q_vec = q_seq[b, 0]
    paper[b] = max(float(torch.dot(q_vec, d_seq[b, j])) for j in range(M))

check('max is taken over the m document vectors', implemented, paper)

# ======================================================== Poly-encoder
# Humeau et al., ICLR 2020, Section 4.4:
#   y^i = sum_j w_j h_j,  (w_1..w_N) = softmax(c_i.h_1, ..., c_i.h_N)
# so each code attends over the SEQUENCE, giving m context vectors.
print('\nPoly-encoder   y^i = sum_j softmax_j(c_i . h_j) h_j')

pe = PolyEncoder.__new__(PolyEncoder)
torch.nn.Module.__init__(pe)
pe.poly_m = M
pe.poly_code_embeddings = torch.nn.Embedding(M, D)
torch.nn.init.normal_(pe.poly_code_embeddings.weight, D ** -0.5)
pe.encoder = make_encoder(q_seq, d_seq, q_ids, d_ids)

with torch.no_grad():
    implemented = pe.forward(q_ids, torch.ones_like(q_ids), d_ids, torch.ones_like(d_ids))

    codes = pe.poly_code_embeddings(torch.arange(M))
    q_rep = q_seq[:, 0, :]
    paper_ctxt = torch.empty(B, M, D)
    for b in range(B):
        for i in range(M):
            w = torch.softmax(torch.tensor(
                [float(torch.dot(codes[i], d_seq[b, j])) for j in range(L)]), dim=0)
            paper_ctxt[b, i] = sum(w[j] * d_seq[b, j] for j in range(L))
    w2 = torch.softmax(torch.einsum('bd,bmd->bm', q_rep, paper_ctxt), dim=-1)
    doc_emb = torch.einsum('bm,bmd->bd', w2, paper_ctxt)
    paper = torch.einsum('bd,bd->b', doc_emb, q_rep)

check('context vectors are formed from the sequence', implemented, paper)

with torch.no_grad():
    inert = torch.allclose(
        implemented, torch.einsum('bd,bd->b', d_seq.sum(dim=1), q_seq[:, 0, :]), atol=1e-3)
print(f'  note  the implemented form reduces to an unweighted sum over '
      f'document positions: {inert}')

print()
if FAILED:
    print(f'{len(FAILED)} of 2 checks failed. Expected in this release -- these '
          f'are the implementations that produced the published rows. See '
          f'docs/known-issues.md.')
    sys.exit(0)
print('ALL CHECKS PASSED')