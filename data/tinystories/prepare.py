"""prepare.py -- build a TinyStories token dataset for llama3-mhc.

Reads the TinyStories parquet (HF roneneldan/TinyStories, sub-shard
train-00000), takes the first `--max_lines` stories, encodes them with a
tiktoken BPE (cl100k_base by default), and writes train.bin / val.bin
(uint32; tiktoken ids exceed uint16) + meta.pkl next to this script.

The data itself is NOT in git — only this generator code is. A fresh clone
can reproduce byte-identical data by running:

    python data/tinystories/prepare.py            # auto-downloads parquet
    # or:  python data/tinystories/prepare.py --parquet /path/to/local.parquet

The fixed default source URL + --max_lines=20000 + cl100k_base gives the
canonical dataset: 20,000 stories / 4,253,128 tokens (verify after prep).
"""
import argparse
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

# fixed source shard (HF TinyStories train-00000-of-00004 parquet)
DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet"
)

p = argparse.ArgumentParser()
p.add_argument("--parquet", type=str, default="__auto__",
               help="path to a TinyStories train parquet; default: auto-download DEFAULT_PARQUET_URL")
p.add_argument("--txt", type=str, default=None,
               help="path to a raw .txt file (alternative to --parquet; split at blank lines)")
p.add_argument("--max_lines", type=int, default=20000,
               help="number of stories to take (balance coverage vs token count)")
p.add_argument("--tokenizer_name", type=str, default="cl100k_base")
p.add_argument("--out", type=str, default=None, help="output dir (default: script dir)")
args = p.parse_args()

here = Path(__file__).resolve().parent
out_dir = Path(args.out) if args.out else here

# --- load text ------------------------------------------------------------
if args.txt:
    stories = Path(args.txt).read_text(encoding="utf-8").split("\n\n")
    stories = [s.strip() for s in stories if s.strip()]
elif args.parquet and args.parquet != "__auto__":
    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet)
    stories = [s.as_py() for s in t.column("text")]
else:
    # auto-download the fixed source shard
    import pyarrow.parquet as pq
    cache = here / "tiny_stories_train0.parquet"
    if not cache.exists():
        print(f"downloading {DEFAULT_PARQUET_URL} ...")
        urlretrieve(DEFAULT_PARQUET_URL, cache)
    t = pq.read_table(cache)
    stories = [s.as_py() for s in t.column("text")]

stories = stories[: args.max_lines]
text = "\n\n".join(stories)
print(f"stories: {len(stories):,} | total chars: {len(text):,}")

# --- tokenize ----------------------------------------------------------------
import tiktoken  # noqa: E402

enc = tiktoken.get_encoding(args.tokenizer_name)
ids = enc.encode(text)
print(f"tokens: {len(ids):,} (vocab_size {enc.n_vocab})")

# --- split / save -------------------------------------------------------------
# tiktoken cl100k_base vocab is 100277 > uint16; store as uint32
split = int(len(ids) * 0.9)
train_ids = np.array(ids[:split], dtype=np.uint32)
val_ids = np.array(ids[split:], dtype=np.uint32)
train_ids.tofile(out_dir / "train.bin")
val_ids.tofile(out_dir / "val.bin")
meta = {"vocab_size": enc.n_vocab, "tokenizer": args.tokenizer_name,
        "dataset": "TinyStories", "n_stories": len(stories),
        "dtype": "uint32",  # tiktoken ids exceed uint16
        "n_tokens": len(ids), "train_tokens": len(train_ids), "val_tokens": len(val_ids)}
with open(out_dir / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"wrote {out_dir / 'train.bin'} ({len(train_ids):,} tokens) + val.bin ({len(val_ids):,}) + meta.pkl")
