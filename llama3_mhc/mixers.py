"""
mHC mixing layer: Sinkhorn-projected doubly stochastic hyper-connections.
==========================================================================
The single manifold mixer of this package, faithful to the mHC paper
(arXiv:2512.24880, Eq.8-9): a free parameter matrix is exponentiated to
positive support (total support) and projected onto the Birkhoff polytope by
`n_iters` alternating Sinkhorn row/column normalizations. The result is an
exactly doubly stochastic mixing matrix -- the manifold constraint that
restores the identity-mapping property of the residual stream.

Interface (shared with the wider Uni-mHC operator family, whose Cayley /
Givens / orthostochastic variants live in the parent project):
    W, S = mixer.mixing_matrices()   # W: real mixing applied to streams
                                     # S: doubly-stochastic energy matrix
    out  = mixer(x)                  # x: (..., n, d) real tensor -> (..., n, d)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

__all__ = [
    "SinkhornMHCResidual",
    "make_mixer",
    "doubly_stochastic_error",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def doubly_stochastic_error(S: torch.Tensor) -> Dict[str, float]:
    """Row-sum / col-sum / non-negativity violation of a candidate matrix S."""
    n = S.shape[-1]
    one = torch.ones(n, dtype=S.dtype, device=S.device)
    row_err = (S.sum(-1) - one).abs().max().item()
    col_err = (S.sum(-2) - one).abs().max().item()
    min_entry = S.min().item()
    return {"row_err": row_err, "col_err": col_err, "min_entry": min_entry}


class _ManifoldMixer(nn.Module):
    """Base class: subclasses implement `mixing_matrices()` returning (W, S)."""

    n_streams: int

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def energy_matrix(self) -> torch.Tensor:
        return self.mixing_matrices()[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the stream axis (-2) of x: (..., n, d) -> (..., n, d)."""
        W, _ = self.mixing_matrices()
        return torch.einsum("jk,...kd->...jd", W, x.to(W.dtype)).to(x.dtype)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# mHC: free matrix -> exp -> Sinkhorn-Knopp projection onto Birkhoff polytope
# ---------------------------------------------------------------------------


class SinkhornMHCResidual(_ManifoldMixer):
    """mHC mixing (arXiv:2512.24880, Eq.8-9): H(0) = exp(raw), then `n_iters`
    alternated row/column Sinkhorn normalizations plus a final row pass.
    Output is entrywise NON-NEGATIVE (convex-combination mixing)."""

    def __init__(self, n_streams: int = 4, n_iters: int = 20, **kwargs):
        super().__init__()
        self.n_streams = n_streams
        self.n_iters = n_iters
        self.raw = nn.Parameter(torch.randn(n_streams, n_streams) * 0.5)

    def mixing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.exp(self.raw)                          # positive support (total support)
        for _ in range(self.n_iters):
            H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # row-normalize
            H = H / H.sum(dim=-2, keepdim=True).clamp_min(1e-12)   # col-normalize
        H = H / H.sum(dim=-1, keepdim=True)              # final row pass -> rows exactly 1
        return H, H

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, n_iters={self.n_iters}, sinkhorn-birkhoff"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

_MIXERS = {
    "sinkhorn": SinkhornMHCResidual,
}


def make_mixer(kind: str, n_streams: int = 4, **kwargs) -> _ManifoldMixer:
    if kind not in _MIXERS:
        raise KeyError(f"unknown mixer '{kind}', choose from {list(_MIXERS)}")
    return _MIXERS[kind](n_streams=n_streams, **kwargs)
