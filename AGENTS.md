# Agent notes (read before writing code here)

This file encodes the working discipline from the parent project (Uni-mHC).
Follow it; it exists because each rule was learned the hard way.

## Verification discipline (non-negotiable)

1. **Citations**: every reference that enters a paper/README must be verified
   against its original source (arXiv abs page / GitHub repo HTTP status) in the
   same session. Plausible-sounding fabrications from memory or from other AIs
   are the #1 historical failure mode here (see Uni-mHC record: "Lu et al.
   2509.04798" never existed; "EΔ-MHC-Geo Transformer" never existed;
   facebookresearch/goru is a 404).
2. **New architecture code**: diff against reference implementations before
   calling it faithful. References used for the Llama backbone:
   `karpathy/llama2.c/model.py` and HF `transformers/.../modeling_llama.py`.
   Known deltas are documented in README (QK-norm omitted, init std 0.02).
3. **Three-layer testing** for any new operator/topology:
   math properties (norm preservation, double stochasticity per token),
   forward/backward gradient finiteness, 20-step real-data training convergence.
4. **Paper-number discipline**: table numbers are generated from logs
   (make_macros.py pattern in the parent repo), never hand-copied.

## Environment gotchas (this machine)

- **WSL has no direct internet** (NAT mode). Run network calls from Windows
  Python; WSL works for local training with the venv `~/venv-torch`.
- nanoGPT's `train.py` needs `--compile=False` here; boolean args must be
  `True/False`, not `1/0`. Vanilla trainer uses `--learning_rate`, not `--lr`.
- Long training: launch with `nohup setsid ... &`; trainer auto-resumes from
  the newest checkpoint. PC may shut down any time after 9:00 local.
- GitHub/Zenodo tokens: never paste secrets in chat; revoke after use.

## Provenance

Operators and topology were written from the mathematical definitions and
verified against primary sources: mHC paper (arXiv:2512.24880 — dynamic
annotations follow Eq.7 + Eq.8-9, as also implemented by the official
deepseek-ai/TileKernels kernels and AndreSlavescu/mHC.cu: flattened n*C stream
RMSNorm as the annotation source, direct linear projection (no tanh; tanh only
appears in the Eq.5 HC preliminary), gating factor alpha init 0.01, stream
expansion by replicating the embedding, exp input clamped at 20, Sinkhorn-Knopp
20 iterations; numeric equivalence 6e-8 vs the paper's iteration order),
go-mHC paper (arXiv:2604.02309; its
official repo itstorque/go-mHC exists but does not ship its core `hyper_conn.py`
module — `model.py` imports a module absent from the tree). The only vendored
external code is karpathy/nanoGPT (Apache-2.0).
