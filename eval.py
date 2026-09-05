"""eval.py -- multiple-choice evaluation for llama3-mhc.

Reads a local json of ARC-style questions (from data/eval/), scores each choice
by the model's negative log-likelihood under a prompt, and reports accuracy.

Usage:
  python eval.py --ckpt runs/mhc/ckpt_final.pt --data data/eval/arc_easy_test50.json
                  --dataset data/tinystories --max_len 256

Note: a pretrained base model scores near random on ARC; this is an
evaluation-tool demonstration, not a capability claim.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from llama3_mhc import LlamaHC, LlamaHCConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--data", type=Path, default=Path("data/eval/arc_easy_test50.json"))
p.add_argument("--dataset", type=Path, default=Path("data/tinystories"))
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

device = args.device
state = torch.load(args.ckpt, map_location=device, weights_only=True)
model = LlamaHC(LlamaHCConfig(**dict(state["model_args"]))).to(device).eval()
model.load_state_dict(state["model"])

# tokenizer from dataset meta
with open(Path(__file__).resolve().parent / args.dataset / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
if "tokenizer" in meta:
    import tiktoken
    enc = tiktoken.get_encoding(meta["tokenizer"])
    def encode(s): return enc.encode(s)
    def decode(ids): return enc.decode(ids)
else:
    # char-level (stoi/itos)
    itos = meta["itos"]
    stoi = meta["stoi"]
    def encode(s): return [stoi[c] for c in s if c in stoi] or [stoi["\n"]]
    def decode(ids): return "".join(itos[i] for i in ids)

items = json.loads(open(args.data, encoding="utf-8").read())
correct = 0
choice_scores = {chr(ord('A') + i): [] for i in range(4)}

@torch.no_grad()
def score(prompt):
    ids = encode(prompt)[-args.max_len:]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    # standard next-token targets: y[i] = ids[i+1], last position repeats
    y_ids = ids[1:] + ids[-1:]
    y = torch.tensor([y_ids], device=device, dtype=torch.long)
    _, loss = model(x, y)  # full-sequence NLL (next-token)
    return loss.item()

for item in items:
    q = item["question"]
    choices = item["choices"]
    ans = item["answerKey"]
    best, best_nll = None, None
    for idx, ch in enumerate(choices):
        lab = "ABCD"[idx]
        nll = score(q + "\n" + lab + ". " + ch)  # few-shot-less: prompt "Q\nA. choice"
        choice_scores[lab].append(nll)
        if best_nll is None or nll < best_nll:
            best_nll = nll; best = lab
    if best == ans:
        correct += 1

acc = correct / len(items)
print(f"eval: {len(items)} examples | accuracy {acc:.3f} ({correct}/{len(items)})")
print("per-choice mean NLL:", {k: round(float(np.mean(v)), 3) for k, v in choice_scores.items()})
