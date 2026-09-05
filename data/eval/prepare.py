"""data/eval/prepare.py -- download ARC-Easy eval subset (code only, data not in git).

Reads the first `--n` examples of ARC-Easy (test split) from HuggingFace
`allenai/ai2_arc` via the `datasets` library, writes arc_easy_test{n}.json
(local, git-ignored) for eval.py. Requires network + `datasets`; run on a
machine that can reach the HF Hub (WSL has no internet by default; use
Windows python here).

Usage:
  python data/eval/prepare.py --n 50
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50, help="number of test examples")
p.add_argument("--out", type=str, default=None, help="output json (default: here)")
args = p.parse_args()

out = Path(args.out) if args.out else Path(__file__).resolve().parent / f"arc_easy_test{args.n}.json"

from datasets import load_dataset  # noqa: E402  (network)
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
items = []
for i, item in enumerate(ds):
    if i >= args.n:
        break
    items.append({
        "question": item["question"],
        "choices": list(item["choices"]["text"]),
        "answerKey": item["answerKey"],
        "id": item["id"],
    })
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f)
print(f"wrote {out} ({len(items)} examples, ARC-Easy test)")
