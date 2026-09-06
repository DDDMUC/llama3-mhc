"""scripts/plot_mhc_gain.py -- composite signal gain vs depth (mHC paper Fig).

Reproduces the mHC paper's core stability argument locally:
  - baseline: identity residual mixing (gain = 1 at any depth)
  - HC      : unconstrained random mixing matrices -> composite gain EXPLODES
  - mHC     : Sinkhorn-projected doubly stochastic matrices -> gain stays ~1

Because doubly stochastic matrices are closed under multiplication, the
composite product H_L ... H_0 has bounded gain for mHC but unbounded for HC.

Reference: mHC paper (arXiv:2512.24880), signal-gain analysis; visualization
idea after bassrehab/mhc-visualizer (MIT). This is a standalone numpy script.

Usage:  python scripts/plot_mhc_gain.py [--out assets/mhc_gain.png] [--n 4]
                                        [--depth 64] [--seeds 20]
"""
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--out", type=str, default="assets/mhc_gain.png")
p.add_argument("--n", type=int, default=4, help="number of residual streams")
p.add_argument("--depth", type=int, default=64)
p.add_argument("--seeds", type=int, default=20, help="random trials")
args = p.parse_args()

rng = np.random.default_rng(0)


def sinkhorn_knopp(M, iters=20):
    """Row/column alternating normalization onto doubly stochastic matrices."""
    H = np.exp(M)
    for _ in range(iters):
        H = H / H.sum(-1, keepdims=True).clip(min=1e-12)
        H = H / H.sum(-2, keepdims=True).clip(min=1e-12)
    return H / H.sum(-1, keepdims=True).clip(min=1e-12)


def max_gain(M):
    """Worst-case signal amplification: max absolute row sum (forward gain)."""
    return np.abs(M).sum(-1).max()


depths = np.arange(1, args.depth + 1)
gains = {"baseline": np.ones_like(depths, dtype=float), "hc": [], "mhc": []}

for _ in range(args.seeds):
    g_hc, g_mhc = [], []
    P_hc, P_mhc = np.eye(args.n), np.eye(args.n)  # running composite products
    for d in depths:
        M = rng.standard_normal((args.n, args.n))
        H_hc = M  # unconstrained N(0,1) (spectral norm >1 -> composite explodes)
        H_mhc = sinkhorn_knopp(M)  # doubly stochastic
        P_hc = H_hc @ P_hc
        P_mhc = H_mhc @ P_mhc
        g_hc.append(max_gain(P_hc))
        g_mhc.append(max_gain(P_mhc))
    gains["hc"].append(g_hc)
    gains["mhc"].append(g_mhc)

gains["hc"] = np.array(gains["hc"]).mean(0)
gains["mhc"] = np.array(gains["mhc"]).mean(0)

# plot
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
ax.plot(depths, gains["baseline"], label="baseline (identity)", color="black", lw=2)
ax.plot(depths, gains["hc"], label="HC (unconstrained)", color="#d62728", lw=2)
ax.plot(depths, gains["mhc"], label="mHC (Sinkhorn doubly stochastic)", color="#1f77b4", lw=2)
ax.set_xlabel("depth (layers)")
ax.set_ylabel("composite forward gain (max |row sum|)")
ax.set_title(f"Composite signal gain vs depth — n={args.n} streams "
             f"(mHC arXiv:2512.24880 argument)", fontsize=10)
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out)
print(f"saved: {args.out}")
print(f"gain @ depth {args.depth}: baseline={gains['baseline'][-1]:.2f} "
      f"HC={gains['hc'][-1]:.2e} mHC={gains['mhc'][-1]:.3f}")
