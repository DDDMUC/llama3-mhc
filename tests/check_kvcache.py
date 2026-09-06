"""tests/check_kvcache.py -- correctness gate for the mHC KV-cache generator.

Contract: generate_kvcache must produce, at every decode step, the SAME
logits as a full forward over the growing prefix (only float roundoff may
differ). If max logit diff over all steps exceeds tol, the KV cache is wrong.

Usage:  python tests/check_kvcache.py [--iters 3] [--tol 1e-3] [--device cuda]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llama3_mhc import LlamaHC, LlamaHCConfig

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=3)
p.add_argument("--tol", type=float, default=1e-3)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
dev = args.device

# small dynamic-mHC model, fp32 for tight tolerance
cfg = LlamaHCConfig(n_layer=2, n_head=6, n_kv_heads=2, n_embd=192,
                    block_size=64, vocab_size=65, mixer="sinkhorn",
                    dynamic_topology=True)
m = LlamaHC(cfg).to(dev).eval()

def full_last_logits(idx):
    """Last-position logits of a full forward over idx (all tokens)."""
    lg, _ = m(idx)
    return lg[:, -1, :]  # (B, V)

def kv_decode_logits(m, idx, n_new):
    """Run the KV-cache path over idx, appending argmax tokens, returning per-step logits."""
    B = idx.size(0)
    hs = cfg.n_embd // cfg.n_head
    nkv = cfg.n_kv_heads
    n_layer = cfg.n_layer
    max_len = cfg.block_size
    cache = {
        "k": torch.zeros(n_layer, B, nkv, max_len, hs, device=dev, dtype=torch.float32),
        "v": torch.zeros(n_layer, B, nkv, max_len, hs, device=dev, dtype=torch.float32),
        "seq_len": idx.size(1),
    }
    # prefill over idx -> initial logits
    tok = m.transformer.wte(idx)
    streams = tok.unsqueeze(2).expand(-1, -1, cfg.n_streams, cfg.n_embd).contiguous()
    cos, sin = m.rope_cos, m.rope_sin
    for li, block in enumerate(m.transformer.h):
        streams = m._block_forward_capture(block, streams, cos, sin, cache, li)
    x = torch.einsum("k,btkd->btd", m.read_out, streams)
    x = m.transformer.ln_f(x)
    logits = m.lm_head(x[:, [-1], :])[:, -1, :]  # (B, V)
    out = idx
    logits_per_step = []
    for _ in range(n_new):
        logits_per_step.append(logits)
        nxt = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat((out, nxt), dim=1)
        # decode the newly sampled token (cache its K/V at pos, predict next)
        last = nxt
        tok = m.transformer.wte(last)
        streams = tok.unsqueeze(2).expand(-1, -1, cfg.n_streams, cfg.n_embd).contiguous()
        for li, block in enumerate(m.transformer.h):
            streams = m._block_forward_decode(block, streams, cos, sin, cache, li)
        m._advance_seq(cache)
        x = torch.einsum("k,btkd->btd", m.read_out, streams)
        x = m.transformer.ln_f(x)
        logits = m.lm_head(x[:, [-1], :])[:, -1, :]
    return logits_per_step

torch.manual_seed(0)
worst = 0.0
for it in range(args.iters):
    prefix = torch.randint(0, 65, (2, 12)).to(dev)
    n_new = 4
    # full path: grow prefix by argmax each step (same schedule as kv)
    full_logits = []
    cur = prefix
    for _ in range(n_new):
        full_logits.append(full_last_logits(cur))
        nxt = full_logits[-1].argmax(dim=-1, keepdim=True)
        cur = torch.cat((cur, nxt), dim=1)
    kv_logits = kv_decode_logits(m, prefix, n_new)
    # compare step-by-step
    for s, (lf, lk) in enumerate(zip(full_logits, kv_logits)):
        d = (lf - lk).abs().max().item()
        worst = max(worst, d)
        if d > args.tol:
            print(f"[iter {it} step {s}] MISMATCH logits diff {d:.4f} > tol {args.tol}")
print(f"worst logits diff over {args.iters} iters x {n_new} steps: {worst:.2e}")
print("KV CACHE CHECK " + ("PASSED" if worst <= args.tol else "FAILED"))
