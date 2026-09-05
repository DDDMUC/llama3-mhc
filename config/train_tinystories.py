"""config/train_tinystories.py -- preset for a TinyStories run with tiktoken.

Used as:  python train.py --config config/train_tinystories.py
(extra CLI args override, e.g. --max_iters=1000)

Data must exist: python data/tinystories/prepare.py --parquet ... (see its header).
Config: 6L/384d, GQA 2 KV heads, bf16, 200 iters demo (~30 min on RTX 4060,
48.4M params, best_val ~4.23 at 200 iters on 4.25M tokens).
"""
out_dir = "runs/tinystories"
dataset = "data/tinystories"

n_layer = 6
n_head = 6
n_kv_heads = 2
n_embd = 384
block_size = 256
dropout = 0.0

max_iters = 200
batch_size = 32
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 20
lr_decay_iters = 200
weight_decay = 0.1
eval_interval = 50
eval_iters = 10
ckpt_every = 100
seed = 1337
dtype = "bf16"
