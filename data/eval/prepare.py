"""data/eval/prepare.py -- download eval subsets (code only, data not in git).

Reads the first `--n` examples of a benchmark from HuggingFace `datasets`
library and writes `<dataset>_test<n>.json` (local, git-ignored) for eval.py.

Supported datasets:
  arc    -> allenai/ai2_arc, ARC-Easy, split test   (multiple-choice)
  gsm8k  -> openai/gsm8k, main, split test          (math word problems)
  mmlu   -> cais/mmlu, <--subject>, split test      (multiple-choice)

Requires network + `datasets` (run where HF Hub is reachable; WSL has no
internet by default — use Windows python).

Usage:
  python data/eval/prepare.py --dataset arc --n 50
  python data/eval/prepare.py --dataset gsm8k --n 20
  python data/eval/prepare.py --dataset mmlu --subject abstract_algebra --n 20
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["arc", "gsm8k", "mmlu"], default="arc")
p.add_argument("--subject", type=str, default=None,
               help="MMLU subject (e.g. abstract_algebra); required for --dataset mmlu")
p.add_argument("--n", type=int, default=50, help="number of test examples")
p.add_argument("--out", type=str, default=None, help="output json (default: here)")
args = p.parse_args()

here = Path(__file__).resolve().parent
name = args.dataset if args.dataset != "mmlu" else f"mmlu_{args.subject}"
out = Path(args.out) if args.out else here / f"{name}_test{args.n}.json"

from datasets import load_dataset  # noqa: E402  (network)

if args.dataset == "arc":
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
    items = []
    for i, item in enumerate(ds):
        if i >= args.n:
            break
        items.append({"question": item["question"],
                      "choices": list(item["choices"]["text"]),
                      "answerKey": item["answerKey"], "id": item["id"]})
elif args.dataset == "gsm8k":
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    items = []
    for i, item in enumerate(ds):
        if i >= args.n:
            break
        items.append({"question": item["question"], "answer": item["answer"]})
elif args.dataset == "mmlu":
    assert args.subject, "--subject required for MMLU"
    ds = load_dataset("cais/mmlu", args.subject, split="test", streaming=True)
    items = []
    for i, item in enumerate(ds):
        if i >= args.n:
            break
        items.append({"question": item["question"],
                      "choices": list(item["choices"]),
                      "answer": item["answer"]})

with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f)
print(f"wrote {out} ({len(items)} examples, {args.dataset})")
