"""llama3_mhc: Llama-3 backbone + mHC (Manifold-Constrained Hyper-Connections).

Single-architecture nano-scale reference implementation. The backbone is the
standard Llama-3 stack (RMSNorm, RoPE theta=500000, GQA, SwiGLU, tied
embeddings); the mHC layer expands the residual stream to n parallel streams
with per-token Sinkhorn-projected doubly stochastic mixing (paper Eq.7+Eq.8-9).
"""

from .mixers import (
    SinkhornMHCResidual,
    make_mixer,
    doubly_stochastic_error,
)
from .model import (
    LlamaHC,
    LlamaHCConfig,
    apply_rope,
    precompute_rope,
    RMSNorm,
)

__all__ = [
    "LlamaHC",
    "LlamaHCConfig",
    "SinkhornMHCResidual",
    "make_mixer",
    "doubly_stochastic_error",
    "apply_rope",
    "precompute_rope",
    "RMSNorm",
]
