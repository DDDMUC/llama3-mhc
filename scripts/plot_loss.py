"""plot_loss.py -- render the train/val loss curve from a run's metrics.jsonl.
Usage:  python scripts/plot_loss.py runs/mhc [--out assets/loss_curve.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("run_dir", type=Path)
p.add_argument("--out", type=Path, default=None, help="output PNG (default: <run_dir>/loss_curve.png)")
args = p.parse_args()

rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
iters = [r["iter"] for r in rows]
out = args.out or args.run_dir / "loss_curve.png"

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(iters, [r["train"] for r in rows], label="train", color="#1f77b4", lw=1.5)
ax.plot(iters, [r["val"] for r in rows], label="val", color="#d62728", lw=1.5)
ax.set_xlabel("iter")
ax.set_ylabel("cross-entropy loss")
ax.set_title(f"Llama-3 + mHC (dynamic Sinkhorn, n=4) on shakespeare_char — {rows[-1]['iter']} iters",
             fontsize=10)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out)
print(f"saved: {out} ({len(rows)} eval points, final train {rows[-1]['train']:.3f} / val {rows[-1]['val']:.3f})")
