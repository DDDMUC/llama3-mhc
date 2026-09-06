<!-- llama3-mhc README (bilingual: Chinese first, English below) -->

[](#zh) | [**English →**](#en)

---

<a id="zh"></a>
# llama3-mhc

**Llama-3 + mHC（流形约束超连接）—— nano 级从零参考实现。**

一个将标准 Llama-3 骨干与
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)（DeepSeek-AI）中 mHC 层
结合起来的**单一架构参考仓库**。它与 nanoGPT / nanochat 同属一类
**nano 级·单架构·教程式参考仓库**——小而自包含、可复现，在单张 GPU 上即可
训练、基准测试与采样。

- **Llama-3 骨干** — RMSNorm（fp32 计算）、RoPE（`theta=500000`）、SwiGLU、
  分组查询注意力（GQA）、无偏置、输入/输出嵌入绑定。
- **mHC 层** — 将残差流扩展为 `n` 条并行流（默认 `n=4`），带逐 token 动态
  注记（mHC Eq.7）与 Sinkhorn-Knopp 双随机矩阵投影（Eq.8-9，20 次迭代）。
- **可暂停训练器** — 每 `--ckpt_every` 步存档，自动从最新 checkpoint 恢复，
  日志/指标写入 `runs/<name>/`。
- **混合精度** — `--dtype {auto,fp32,bf16,fp16}`（auto = CUDA 支持时 bf16），
  fp16 用 GradScaler，可选用 `--compile`。
- **真实 tokenizer 数据** — `data/tinystories/`：TinyStories 用 tiktoken
  （cl100k_base）BPE 编码（词表 100277）；`train.py` 按 `meta['dtype']`
  自动选择 uint16/uint32。
- **基准与评估** — `bench.py`（tok/s、MFU）与 `eval.py`（多项选择 ARC 评估）。
- **KV-cache 推理** — `sample.py --kvcache` 用增量解码加速生成；与朴素 `generate`
  数学等价（logits 差 ~1e-6，已由 `tests/check_kvcache.py` 门禁验证）。默认用朴素
  `generate` 以保证逐 token 可复现；`--kvcache` 供推理提速。

---

## 快速开始（shakespeare_char）

```bash
# 1) 准备数据（一次性；若缺失会自动下载 input.txt）
python data/shakespeare_char/prepare.py

# 2) 训练默认 mHC 模型（RTX 4060 Laptop 约 17 分钟，2000 步）
python train.py --out_dir=runs/mhc
# 或用预设：  python train.py --config config/train_shakespeare_char.py

# 3) 采样 / 聊天
python sample.py --ckpt runs/mhc/ckpt_final.pt --max_new_tokens=500 --temperature=0.8 --top_k=40
python sample.py --ckpt runs/mhc/ckpt_final.pt --chat   # 交互式 REPL

# 4) 基准测试（tok/s, MFU）
python bench.py --dataset data/shakespeare_char --dtype bf16
```

用 TinyStories + tiktoken 快速上手（`config/train_tinystories.py`）：

```bash
python data/tinystories/prepare.py --parquet path/to/tiny_stories.parquet --max_lines 20000
python train.py --config config/train_tinystories.py
```

默认 `--mixer=sinkhorn` + `--dynamic_topology=True` 正是论文中的 mHC 方法。
`--mixer=none` 给出裸单流 Llama-3 基线；`--no_dynamic_topology` 切换到静态
读写向量。

---

## 架构

**骨干**（标准 Llama-3，逐组件对照 HF `transformers` Llama 与
`karpathy/llama2.c` 参考实现）：

- RMSNorm（fp32 计算，`eps=1e-5`），pre-norm。
- RoPE，`theta=500000`，HF `rotate_half` 约定（与 Meta 复数配对公式数学等价）。
- SwiGLU MLP（`d_ffn = round_to_multiple(8*d/3, 64)`）。
- GQA（`n_kv_heads<=n_head`，默认 6L 配置下为 2）。
- 输入/输出嵌入绑定（`lm_head.weight is wte.weight`）。

**mHC 层**（"mHC" 之所在）：每层的残差流被替换为 `n` 条并行流。对每个阶段
（注意力/FFN），一个融合投影从展平流的 RMSNorm 产生逐 token 的读/写/注记
（Eq.7），混合矩阵经 Sinkhorn-Knopp 投影到双随机矩阵（Eq.8-9，20 次迭代，
无 tanh——Eq.5 的 tanh 是 HC 预备式，非 mHC）。

关键性质：每个 token 的混合是凸组合（全部元素 ≥0，行和=1），从而恢复恒等
映射，前向/反向信号增益保持有界。完整数学见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 验证

- **骨干** — 逐组件对照 HF Llama 与 llama2.c；RoPE 已验证范数保持与相对
  位移不变性。
- **mHC 混合器** — 已验证双随机性（行/列误差 ≤1e-6），与论文迭代顺序数值
  等价（6e-8）。
- **训练** — `tests/smoke.py` 跑 8 项检查（RoPE 数学、绑定+梯度、双随机性、
  动态拓扑、none 路径、优化器分组+generate、20 步真实数据收敛、bf16 AMP）。
  `ALL SMOKE TESTS PASSED`（CUDA）。
- **实测** — 2000 步 canonical 运行后，全部 12 个训练后混合矩阵仍满足双随机
  性（最差行/列和误差 `1.19e-07`，全元素非负）。

| 配置 | 值 |
|---|---|
| shakespeare_char（6L/384d） | 9.93M 参数，2000 步，best val **1.5094 @ iter 600**，~17 分钟 |
| TinyStories demo（6L/384d, tiktoken, bf16） | 48.41M 参数，200 iters，best val **4.229**，~30 分钟 |

**评估工具** — `eval.py` 以 next-token NLL 对 ARC 式多项选择题评分。200-iter
TinyStories 基础模型在 ARC-Easy（50 例子集）上得 **0.280 (14/50)**，接近
随机——符合未 SFT 的基础模型的预期。**这是评估工具可用性演示，不是能力
声明。**

Loss 曲线：`assets/loss_curve.png`（由 `scripts/plot_loss.py` 从
`runs/mhc/metrics.jsonl` 生成）。

**mHC 稳定性论证** — `scripts/plot_mhc_gain.py` 本地复现 mHC 论文的核心论点
（可视化思路借鉴 [bassrehab/mhc-visualizer](https://github.com/bassrehab/mhc-visualizer),
MIT）：复合信号增益 vs 深度 —— baseline（恒等）1.0、HC（无约束矩阵）在深度 64
爆炸至 ~1e17、mHC（Sinkhorn 双随机）保持 ~1.0。输出 `assets/mhc_gain.png`。

## 仓库结构

```
llama3_mhc/     model.py（骨干 + mHC 块）, mixers.py（Sinkhorn 数学）
data/           shakespeare_char/ 与 tinystories/（tiktoken）数据集 + eval/
tests/          smoke.py — 8 项检查
scripts/        plot_loss.py — metrics.jsonl -> loss 曲线 PNG
train.py        可暂停训练器（支持 --config）
sample.py       生成 + --chat REPL
bench.py        吞吐量/MFU 基准
eval.py         多项选择评估（ARC, next-token NLL）
config/         预设（train_shakespeare_char.py, train_tinystories.py）
assets/         canonical 运行的 loss 曲线 + 采样文本
runs/mhc/       canonical 运行的日志/指标（权重不包含，见下）
```

- 权重：checkpoint 被 git-ignore（`*.pt`）；canonical 运行的 checkpoint
  （119MB × 2）已归档到仓库外 `E:\Uni-mHC\runs_archive_llama3mhc\`。
  `runs/mhc/` 仅保留支撑 README 数字的日志（12K）。
- 中断后继续：`python train.py --out_dir=runs/mhc` 自动从最新 checkpoint 恢复。

## 参考与来源（本仓库实际参考了什么）

以下是为实现本仓库而**实际**对照/借鉴的来源（不含任何"仅风格相似"的仓库）：

| 我们代码/文档 | 来源 | 性质 |
|---|---|---|
| Llama-3 骨干（RMSNorm/RoPE/GQA/SwiGLU） | `karpathy/llama2.c model.py` + HF `transformers modeling_llama.py` | 逐组件 diff 验证 |
| mHC 公式（Eq.7 + Eq.8-9, 20 迭代, exp clamp, 流复制） | mHC 论文 (arXiv:2512.24880) + `AndreSlavescu/mHC.cu` + `deepseek-ai/TileKernels` | 对照论文公式 + 数值细节 |
| trainer（pause/resume/eval）与 bench 结构 | `karpathy/nanoGPT train.py` / `bench.py` | 结构借鉴（代码中已标注"Differences from nanoGPT"） |
| `data/shakespeare_char/prepare.py` | `karpathy/nanoGPT`（Apache-2.0） | vendored 复制 |

**我们未参考/未对比**：`huggingface/nanowhale` 及任何其它"nano"仓库。nanowhale（HF DeepSeek-V4 迷你复刻）的 Hyper-Connections 是 `hc_mult=4` + **Sinkhorn 仅 2 次迭代 + softmax 初始**——**与 mHC 论文（20 次, exp 初始）不同**，且**非本仓库实现依据**。本仓库的 mHC 忠实于论文，与 nanowhale 无关。

---

## 相关工作

- **HC** — Zhu et al., *Hyper-connections*, [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
  （mHC 所基于的方法；经 mHC 论文参考文献核实）。
- **mHC** — Xie et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*,
  [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).

## 许可

代码 MIT。`data/shakespeare_char/prepare.py` 取自
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)（Apache-2.0）。HF
`transformers` 参考用于对照（仅 diff，不随发布；Apache-2.0）。

---

<a id="en"></a>
# llama3-mhc

**Llama-3 + mHC (Manifold-Constrained Hyper-Connections) at nano scale, from scratch.**

A single-architecture reference implementation that combines the standard
Llama-3 backbone with the mHC layer from
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (DeepSeek-AI). It belongs
to the same class of **nano-scale, single-architecture, tutorial-style
reference repos** as nanoGPT / nanochat — small, self-contained,
reproducible, trainable on a single GPU.

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
- **Benchmark & eval** — `bench.py` (tok/s, MFU) and `eval.py` (multiple-choice ARC).
- **KV-cache inference** — `sample.py --kvcache` uses incremental decode for faster
  generation; mathematically equivalent to plain `generate` (logits differ ~1e-6,
  verified by `tests/check_kvcache.py`). Plain `generate` stays the default for
  token-level reproducibility; `--kvcache` is the speed path.

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
- **Training** — `tests/smoke.py` runs 8 checks (RoPE math, tying+grads,
  double stochasticity, dynamic topology, none path, optimizer grouping +
  generate, 20-step real-data convergence, bf16 AMP). `ALL SMOKE TESTS PASSED`
  on CUDA.
- **Live evidence** — after the 2000-iter canonical run, all 12 trained mixing
  matrices still satisfy double stochasticity (worst row/col sum error
  `1.19e-07`, all entries nonnegative).

| config | value |
|---|---|
| shakespeare_char (6L/384d) | 9.93M params, 2000 iters, best val **1.5094 @ iter 600**, ~17 min |
| TinyStories demo (6L/384d, tiktoken, bf16) | 48.41M params, 200 iters, best val **4.229**, ~30 min |

**Evaluation tooling** — `eval.py` scores ARC-style multiple-choice questions by
next-token NLL. On the 200-iter TinyStories base model it gets **0.280 (14/50)**
on ARC-Easy (50-example subset), near random — expected for a pretrained base
model that has not been SFT'd. This is an eval-tool demo, not a capability
claim.

Loss curve: `assets/loss_curve.png` (rendered from `runs/mhc/metrics.jsonl`
by `scripts/plot_loss.py`).

**mHC stability argument** — `scripts/plot_mhc_gain.py` reproduces the mHC
paper's core argument locally (visualization idea after
[bassrehab/mhc-visualizer](https://github.com/bassrehab/mhc-visualizer), MIT):
composite signal gain vs depth — baseline (identity) 1.0, HC (unconstrained
matrices) explodes to ~1e17 at depth 64, mHC (Sinkhorn doubly stochastic)
stays ~1.0. Output: `assets/mhc_gain.png`.

## Repo layout

```
llama3_mhc/     model.py (backbone + mHC blocks), mixers.py (Sinkhorn math)
data/           shakespeare_char/ and tinystories/ (tiktoken) datasets + eval/
tests/          smoke.py — 8-check suite
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

## References & provenance (what this repo actually drew from)

Definitions below are what this repo **actually** consulted/reused (not any
"style-similar" repo):

| our code/docs | source | relation |
|---|---|---|
| Llama-3 backbone (RMSNorm/RoPE/GQA/SwiGLU) | `karpathy/llama2.c model.py` + HF `transformers modeling_llama.py` | component-diff verified |
| mHC math (Eq.7 + Eq.8-9, 20 iters, exp clamp, stream replication) | mHC paper (arXiv:2512.24880) + `AndreSlavescu/mHC.cu` + `deepseek-ai/TileKernels` | against paper formulas + numeric details |
| trainer (pause/resume/eval) + bench structure | `karpathy/nanoGPT train.py` / `bench.py` | structural borrow (annotated "Differences from nanoGPT") |
| `data/shakespeare_char/prepare.py` | `karpathy/nanoGPT` (Apache-2.0) | vendored |

**We did NOT consult/compare** `huggingface/nanowhale` or any other "nano" repo.
nanowhale (HF's DeepSeek-V4 mini-recreation) uses Hyper-Connections with
`hc_mult=4` + **only 2 Sinkhorn iterations + softmax init** — **not mHC
(20 iters, exp init)** and not a basis for this repo's implementation. Our mHC
is faithful to the paper and unrelated to nanowhale.

---

## Related work

- **HC** — Zhu et al., *Hyper-connections*, [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
  (the method mHC builds on; verified via mHC paper's bibliography).
- **mHC** — Xie et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*,
  [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).

## License

MIT for this code. `data/shakespeare_char/prepare.py` is vendored from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (Apache-2.0). The HF
`transformers` reference is Apache-2.0 (used for diffing only, not shipped).
