"""bench.py -- quick throughput/MFU benchmark for llama3-mhc.

Loosely follows karpathy/nanoGPT bench.py, but builds the LlamaHC (mHC)
model with the same --dtype/bf16 path as train.py. Reports ms/iter, tok/s,
and MFU (model FLOPs utilization) for the training step.

Usage:
  python bench.py --dataset data/shakespeare_char --dtype bf16 --steps 20
"""
import argparse
import math
import pickle
from pathlib import Path
import time

import numpy as np
import torch
from contextlib import nullcontext

from llama3_mhc import LlamaHC, LlamaHCConfig

p = argparse.ArgumentParser()
p.add_argument("--dataset", type=Path, default=Path("data/shakespeare_char"))
p.add_argument("--steps", type=int, default=20)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_kv_heads", type=int, default=2)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

data_dir = Path(__file__).resolve().parent / args.dataset
with open(data_dir / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
data_dtype = np.uint32 if meta.get("dtype", "uint16") == "uint32" else np.uint16
train_data = np.memmap(data_dir / "train.bin", dtype=data_dtype, mode="r")

def get_batch():
    ix = torch.randint(len(train_data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((train_data[i:i + args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((train_data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
    if dev == "cuda":
        x, y = x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
    else:
        x, y = x.to(dev), y.to(dev)
    return x, y

model = LlamaHC(LlamaHCConfig(
    n_layer=args.n_layer, n_head=args.n_head, n_kv_heads=args.n_kv_heads,
    n_embd=args.n_embd, block_size=args.block_size, vocab_size=vocab_size,
    mixer="sinkhorn", dynamic_topology=True,
)).to(dev)
model.train()
optimizer = model.configure_optimizers(1e-2, 1e-4, (0.9, 0.95), dev)

dtype = args.dtype
ptdtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
ctx = nullcontext() if (dev == "cpu" or dtype == "fp32") else torch.amp.autocast(dev, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "fp16"))

# num params
n_params = sum(p.numel() for p in model.parameters())
n_tokens = args.batch_size * args.block_size

# FLOPs per forward token (PaLM formula): 6*N matmul FLOPs/param for fwd+bwd +
# 4*L*H*D per token for attention (PaLM uses ~6N + 4*L*H*D). We use a rough
# fwd+bwd estimate: 6*N + 12*L*H*D (nanoGPT-ish).
flops_per_token = 6 * n_params + 12 * args.n_layer * args.n_head * (args.n_embd // args.n_head) * args.block_size
flops_per_step = flops_per_token * n_tokens

# device peak FLOPs (approximate; used only for a rough MFU figure)
PEAK_FLOPS = {"cuda": 242e12, "cpu": 500e9}.get(dev, 1e12)  # 4060 Laptop bf16 ~242 TFLOPS

X, Y = get_batch()
print(f"bench: {dev} dtype={dtype} params={n_params/1e6:.2f}M "
      f"batch={args.batch_size}x{args.block_size} steps={args.steps}")
# warmup 3 steps
for _ in range(3):
    with ctx:
        _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if dtype == "fp16":
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()

# timed loop
torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.time()
for _ in range(args.steps):
    with ctx:
        _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if dtype == "fp16":
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    X, Y = get_batch()
torch.cuda.synchronize() if dev == "cuda" else None
dt = time.time() - t0
ms_per_iter = dt / args.steps * 1000
tok_s = n_tokens / (dt / args.steps)
mfu = flops_per_step / (dt / args.steps) / PEAK_FLOPS * 100
print(f"time per iter: {ms_per_iter:.2f}ms | {tok_s/1e3:.1f}k tok/s | loss {loss.item():.3f}")
print(f"MFU (vs ~{PEAK_FLOPS/1e12:.0f} TFLOPS peak): {mfu:.1f}%")
