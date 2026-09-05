# llama3-mhc

**Llama-3 + mHC (Manifold-Constrained Hyper-Connections) at nano scale, from scratch.**

A single-architecture reference implementation that combines the standard
Llama-3 backbone with the mHC layer from
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (DeepSeek-AI). It follows
the nanoGPT/nanowhale tradition — a small, self-contained, reproducible
codebase you can train, benchmark, and sample on a single GPU.

- **Llama-3 backbone** — RMSNorm (fp32 cast), RoPE (`theta=500000`), SwiGLU,
  Grouped-Query Attention, no biases, tied input/output embedding.
- **mHC layer** — expands the residual stream to `n` parallel streams (default
  `n=4`) with per-token dynamic annotations (mHC Eq.7) and a Sinkhorn-Knopp
  projection onto doubly stochastic mixing matrices (Eq.8-9, 20 iterations).
- **Pausable trainer** — checkpoints every `--ckpt_every` steps, auto-resumes
  from the newest checkpoint, writes logs/metrics to `runs/<name>/`.
- **Mixed precision** — `--dtype {auto,fp32,bf16,fp16}` (auto = bf16 on
  supported CUDA), GradScaler for fp16, optional `--compile`.
- **Real tokenizer data** — `data/tinystories/`: tiktoken (cl100k_base) BPE on
  TinyStories (vocab 100277); `train.py` auto-selects uint16/uint32 by
  `meta['dtype']`.

The mHC layer is included because it's the method from the paper; the
Cayley/Givens/orthostochastic mixers (the wider Uni-mHC operator family) live
in the [parent Uni-mHC project](https://github.com/DDDMUC/Uni-mHC).

---

## Quick start (shakespeare_char)

```bash
# 1) prepare the data (once; downloads input.txt if missing)
python data/shakespeare_char/prepare.py

# 2) train the default mHC model (~17 min on an RTX 4060 Laptop GPU, 2000 iters)
python train.py --out_dir=runs/mhc
# or use a preset:  python train.py --config config/train_shakespeare_char.py

# 3) sample / chat
python sample.py --ckpt runs/mhc/ckpt_final.pt --max_new_tokens=500 --temperature=0.8 --top_k=40
python sample.py --ckpt runs/mhc/ckpt_final.pt --chat   # interactive REPL

# 4) benchmark (tok/s, MFU)
python bench.py --dataset data/shakespeare_char --dtype bf16
```

Quick setup with TinyStories + tiktoken (`config/train_tinystories.py`):

```bash
python data/tinystories/prepare.py --parquet path/to/tiny_stories.parquet --max_lines 20000
python train.py --config config/train_tinystories.py
```

The default `--mixer=sinkhorn` + `--dynamic_topology=True` is exactly the mHC
method from the paper. `--mixer=none` gives a vanilla single-stream Llama-3
baseline; `--no_dynamic_topology` switches to static read/write vectors.

---

## Architecture

**Backbone** (standard Llama-3, component-diffed against the HF
`transformers` Llama and `karpathy/llama2.c` references):

- RMSNorm with fp32 cast (`eps=1e-5`), pre-norm.
- RoPE with `theta=500000`, HF `rotate_half` convention (mathematically
  equivalent to the Meta complex-pair formulation).
- SwiGLU MLP (`d_ffn = round_to_multiple(8*d/3, 64)`).
- GQA (`n_kv_heads<=n_head`, default 2 for the 6L config).
- Tied input/output embedding (`lm_head.weight is wte.weight`).

**mHC layer** (the "M" in `llama3-mhc`): the residual stream of each layer is
replaced by `n` parallel streams. For each stage (attention/FFN) one fused
projection generates per-token read/hold/write annotations from RMSNorm of the
flattened stream (Eq.7), the mixing matrix is projected onto doubly stochastic
matrices by Sinkhorn-Knopp (Eq.8-9, 20 iterations, no tanh — Eq.5's tanh is
the HC preliminary, not mHC).

Key properties: each token's mixing is a convex combination (all entries ≥0,
rows sum to 1), so the identity mapping is restored and the forward/backward
signal gain stays bounded. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full
math.

## Verification

- **Backbone** — component-diffed against HF Llama and llama2.c; RoPE verified
  for norm preservation and relative-position shift invariance.
- **mHC mixer** — verified for double stochasticity (row/col error <=1e-6),
  and numeric equivalence to the paper's iteration order (6e-8).
- **Training** — `tests/smoke.py` runs 7 checks (RoPE math, tying+grads,
  double stochasticity, dynamic topology, none path, optimizer grouping +
  generate, 20-step real-data convergence). `ALL SMOKE TESTS PASSED` on CUDA.
- **Live evidence** — after the 2000-iter canonical run, all 12 trained mixing
  matrices still satisfy double stochasticity (worst row/col sum error
  `1.19e-07`, all entries nonnegative) and the loss curve is monotonic
  (val `4.23 → 1.5094 @ iter 600`).

| metric | value |
|---|---|
| config | Llama-3 6L/384d, GQA 2 KV heads, n=4 streams, dynamic mHC |
| params | 9.93M |
| best val | 1.5094 @ iter 600 |
| final train | 0.3711 @ 2000 iters |
| final val | 2.3821 @ 2000 iters (overfitting past ~600, expected on 1M-char data) |
| wall time | ~17 min (RTX 4060 Laptop, ~16k tok/s) |

| TinyStories demo (6L/384d, tiktoken, bf16) | value |
|---|---|
| data | 20,000 stories → 4.25M tokens (cl100k_base, vocab 100277) |
| params | 48.41M |
| best val | 4.229 @ iter 200 (demo, ~30 min) |
| final train | 4.3318 @ 200 iters |
| mixing row_err | 0.00e+00 after training |

**Evaluation tooling** — `eval.py` scores ARC-style multiple-choice questions by
next-token NLL. On the 200-iter TinyStories base model it gets **0.280 (14/50)**
on ARC-Easy (50-example subset), which is near random — expected for a
pretrained base model that has not been SFT'd. This is an eval-tool demo, not a
capability claim.

Loss curve: `assets/loss_curve.png` (rendered from `runs/mhc/metrics.jsonl`
by `scripts/plot_loss.py`).

## Repo layout

```
llama3_mhc/     model.py (backbone + mHC blocks), mixers.py (Sinkhorn math)
data/           shakespeare_char dataset (prepare.py + .bin + meta.pkl)
tests/          smoke.py — 7-check suite
scripts/        plot_loss.py — metrics.jsonl -> loss curve PNG
train.py        pausable nanoGPT-style trainer (--config support)
sample.py       generation + --chat REPL
bench.py        throughput/MFU benchmark
eval.py         multiple-choice eval (ARC, next-token NLL)
config/         preset configs (train_shakespeare_char.py, train_tinystories.py)
assets/         canonical run's loss curve + sample text
runs/mhc/       canonical run's logs/metrics (weights not included, see below)
```

- Weights: checkpoints are git-ignored (`*.pt`); the canonical run's
  checkpoints (119MB × 2) were archived outside the repo to
  `E:\Uni-mHC\runs_archive_llama3mhc\`. `runs/mhc/` keeps only the logs
  (12K) that back the README numbers.
- Iterate after interruption: `python train.py --out_dir=runs/mhc` resumes
  automatically from the newest checkpoint.

## Related work

- **HC** — Zhu et al., *Hyper-connections*, [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
  (the method mHC builds on; verified via mHC paper's bibliography, Aug 2026).
- **mHC** — Xie et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*,
  [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).
- **nanowhale** — Hugging Face's DeepSeek-V4 mini-recreation, which also uses
  Sinkhorn-normalized Hyper-Connections. [github.com/huggingface/nanowhale](https://github.com/huggingface/nanowhale).

## License

MIT for this code. `data/shakespeare_char/prepare.py` is vendored from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (Apache-2.0). The HF
`transformers` reference is Apache-2.0 (used for diffing only, not shipped).
