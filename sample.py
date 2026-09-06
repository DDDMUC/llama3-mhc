"""sample.py -- generate text from (or chat with) a llama3-mhc checkpoint.

Dataset-aware: the codec (meta.pkl) is resolved from the dataset recorded in
the checkpoint, using the same repo-root convention as train.py. If the
checkpoint predates the data/ layout (records "data/shakespeare_char"), a
fallback tries the old nanoGPT/ location so older checkpoints stay loadable.
"""
import argparse
import pickle
from pathlib import Path

import torch

from llama3_mhc import LlamaHC, LlamaHCConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--out", type=Path, default=None)
p.add_argument("--max_new_tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--prompt", type=str, default="\n")
p.add_argument("--chat", action="store_true",
               help="REPL mode: type a prompt, get a continuation (empty line exits)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

torch.manual_seed(args.seed)  # None -> entropy-seeded
device = args.device
state = torch.load(args.ckpt, map_location=device, weights_only=True)
model = LlamaHC(LlamaHCConfig(**dict(state["model_args"]))).to(device)
model.load_state_dict(state["model"])
model.eval()

# Resolve the dataset dir recorded in the ckpt: repo-root data/ first, then old nanoGPT/ fallback.
repo_root = Path(__file__).resolve().parent
dataset_rel = state["config"]["dataset"]
dataset_dir = repo_root / dataset_rel
if not (dataset_dir / "meta.pkl").exists():
    dataset_dir = repo_root / "nanoGPT" / dataset_rel
    if not (dataset_dir / "meta.pkl").exists():
        raise FileNotFoundError(f"meta.pkl not found for dataset {dataset_rel}")
with open(dataset_dir / "meta.pkl", "rb") as f:
    meta = pickle.load(f)

if "tokenizer" in meta:  # tiktoken-tokenized dataset (e.g. TinyStories)
    import tiktoken
    enc = tiktoken.get_encoding(meta["tokenizer"])
    encode = enc.encode
    decode = enc.decode
else:  # char-level dataset (shakespeare_char)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi] or [stoi["\n"]]
    decode = lambda ids: "".join(itos[i] for i in ids)


def generate(prompt: str) -> str:
    x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    return decode(y[0].tolist())


best_val = state.get("best_val_loss")
if not args.chat:
    print(f"--- sample from {args.ckpt.name} (iter {state.get('iter_num')}, "
          f"best_val {best_val:.4f}) ---")
    text = generate(args.prompt)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved: {args.out}")
else:
    print(f"chat REPL -- {args.ckpt.name} (iter {state.get('iter_num')}); "
          f"empty line or Ctrl-D exits")
    while True:
        try:
            prompt = input("\nyou> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        print("model>", generate(prompt), sep="\n")
