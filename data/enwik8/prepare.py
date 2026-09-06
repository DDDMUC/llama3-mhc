"""prepare.py -- build the enwik8 byte-level dataset for llama3-mhc.

The enwik8 dataset (100MB of Wikipedia XML) is read as raw bytes, split 90/10
into train/val, and written as uint8 token ids (vocab 256) + meta.pkl.

Byte-level is the standard enwik8 setting (Hutter prize; loss is reported as
bits-per-char). The model can report bpc = loss / ln(2).

Data is NOT in git. Download it first (needs network; run outside WSL):
  curl -L -o enwik8.zip https://mattmahoney.net/dc/enwik8.zip
  # extract enwik8 (100MB) next to this script

Usage:
  python data/enwik8/prepare.py [--src path/to/enwik8] [--val-frac 0.1]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", type=str, default=None,
               help="path to the enwik8 file (default: <this dir>/enwik8)")
p.add_argument("--val-frac", type=float, default=0.1)
args = p.parse_args()

here = Path(__file__).resolve().parent
src = Path(args.src) if args.src else here / "enwik8"
if not src.exists():
    raise SystemExit(f"{src} not found — download enwik8.zip from "
                     "https://mattmahoney.net/dc/enwik8.zip and extract it here")

data = src.read_bytes()
print(f"enwik8 bytes: {len(data):,}")

n = len(data)
split = int(n * (1 - args.val_frac))
train = np.frombuffer(data[:split], dtype=np.uint8)
val = np.frombuffer(data[split:], dtype=np.uint8)
train.tofile(here / "train.bin")
val.tofile(here / "val.bin")

meta = {"vocab_size": 256, "dataset": "enwik8", "dtype": "uint8",
        "n_bytes": n, "train_bytes": len(train), "val_bytes": len(val)}
with open(here / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"wrote train.bin ({len(train):,} bytes) + val.bin ({len(val):,}) + meta.pkl "
      f"(vocab 256, byte-level)")
