"""eval.py -- evaluation for llama3-mhc (multiple-choice + math).

Two modes, auto-detected from the loaded json:
  * multiple-choice (ARC / MMLU): each item has `question` + `choices` (list).
    Each choice is scored by the model's next-token NLL under the prompt
    "Q\nX. choice"; the argmin-NLL choice is the prediction.
  * math (GSM8K): each item has `question` + `answer`. The model generates a
    completion; the first integer in the completion is compared to the answer's
    final number (extracted after '####').

Honest caveat: a pretrained base model (esp. a 200-iter TinyStories one) scores
near random / ~0 on these tasks. This is evaluation-tooling demonstration, not
a capability claim.

Usage:
  python eval.py --ckpt runs/tinystories_6l384/ckpt_final.pt \
      --data data/eval/arc_easy_test50.json --dataset data/tinystories
  python eval.py --ckpt ... --data data/eval/gsm8k_test20.json --dataset ...
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from llama3_mhc import LlamaHC, LlamaHCConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--dataset", type=Path, default=Path("data/tinystories"))
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--max_new_tokens", type=int, default=64)
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
else:
    stoi = meta["stoi"]
    def encode(s): return [stoi[c] for c in s if c in stoi] or [stoi["\n"]]

items = json.loads(open(args.data, encoding="utf-8").read())

@torch.no_grad()
def nll(prompt):
    ids = encode(prompt)[-args.max_len:]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    y_ids = ids[1:] + ids[-1:]
    y = torch.tensor([y_ids], device=device, dtype=torch.long)
    _, loss = model(x, y)
    return loss.item()

@torch.no_grad()
def generate(prompt, max_new=args.max_new_tokens):
    ids = encode(prompt)[-args.max_len:]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    y = model.generate(x, max_new, temperature=0.2, top_k=40)
    return y[0].tolist()[len(ids):]

# detect format
if all("choices" in it for it in items):
    # multiple-choice (ARC / MMLU)
    correct = 0
    for it in items:
        q, choices, ans = it["question"], it["choices"], it.get("answerKey") or "ABCD"[it["answer"]]
        nlls = [nll(q + "\n" + lab + ". " + ch) for lab, ch in zip("ABCD", choices)]
        pred = "ABCD"[int(np.argmin(nlls))]
        if pred == ans:
            correct += 1
    print(f"eval: {len(items)} MC examples | accuracy {correct/len(items):.3f} "
          f"({correct}/{len(items)}) [random ≈ {1/len(choices):.3f}]")
else:
    # math (GSM8K): compare generated first integer vs answer's final number
    correct = 0
    for it in items:
        q, ans = it["question"], it["answer"]
        gold_m = re.search(r"####\s*(-?\d+)", ans)
        gold = gold_m.group(1) if gold_m else None
        out_ids = generate("Q: " + q + "\nA:")
        # decode ids to text for integer extraction
        if "tokenizer" in meta:
            text = enc.decode(out_ids)
        else:
            itos = meta["itos"]
            text = "".join(itos.get(i, "?") for i in out_ids)
        pred_m = re.search(r"-?\d+", text)
        pred = pred_m.group(0) if pred_m else None
        if gold is not None and pred == gold:
            correct += 1
    print(f"eval: {len(items)} GSM8K examples | accuracy {correct/len(items):.3f} "
          f"({correct}/{len(items)}) [expect ~0 for base model]")
