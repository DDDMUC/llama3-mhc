"""config/train_shakespeare_char.py -- preset for the char-level demo.

Used as:  python train.py --config config/train_shakespeare_char.py
(extra CLI args override, e.g. --max_iters=500)

Matches the canonical 2000-iter run (best val 1.5094 @ iter 600, ~17 min on
an RTX 4060 Laptop, 9.93M params).
"""
out_dir = "runs/mhc_shakespeare"
dataset = "data/shakespeare_char"

n_layer = 6
n_head = 6
n_kv_heads = 2
n_embd = 384
block_size = 256
dropout = 0.0

max_iters = 2000
batch_size = 32
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000
weight_decay = 0.1
eval_interval = 100
eval_iters = 50
ckpt_every = 200
seed = 1337
