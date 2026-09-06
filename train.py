"""
train.py -- pausable nanoGPT-style training for Llama-3 + mHC.

Single-package version of the old nanoGPT/train_hc.py. Differences from
karpathy/nanoGPT (train.py):
  * builds LlamaHC from llama3_mhc.model (n residual streams + mHC mixer)
  * default path = the mHC paper method: mixer=sinkhorn + dynamic topology
    (mHC Eq.7 + Eq.8-9); --mixer=none gives the vanilla single-stream baseline
  * pausable: saves ckpt_latest.pt every --ckpt_every iters, auto-resumes from
    the newest checkpoint in --out_dir when --init_from=resume (default),
    appends eval metrics to metrics.jsonl (restart-safe), and saves
    ckpt_final.pt at the end.
Run (shakespeare_char):
  python train.py --out_dir=runs/mhc
Resume after interruption (same command): it picks up where it left off.
"""

import argparse
import json
import math
import pickle
from contextlib import nullcontext
import sys
import time
from pathlib import Path

import numpy as np
import torch

from llama3_mhc import LlamaHC, LlamaHCConfig

# -----------------------------------------------------------------------------


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(
        (data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        (data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, ctx, eval_iters, autocast_ctx=nullcontext()):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, *ctx)
            with autocast_ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def find_latest_ckpt(out_dir: Path):
    cands = [c for c in (out_dir / "ckpt_latest.pt", out_dir / "ckpt_final.pt") if c.exists()]
    if not cands:
        return None
    def itern(p):
        try:
            meta = json.loads((p.parent / (p.stem.replace("ckpt_", "meta_ckpt_") + ".json")).read_text())
            return meta.get("iter", 0)
        except Exception:
            return 0
    return max(cands, key=itern)


def save_ckpt(model, optimizer, iter_num, best_val, out_dir: Path, name):
    # Paths -> str so the payload stays weights_only=True loadable
    cli_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(g_args).items()}
    raw = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
           "model_args": vars(model.config), "iter_num": iter_num, "best_val_loss": best_val,
           "config": cli_cfg}
    torch.save(raw, out_dir / f"{name}.pt")
    (out_dir / f"meta_{name}.json").write_text(json.dumps({"iter": iter_num, "best_val_loss": best_val}))


# -----------------------------------------------------------------------------


g_args = None
ctx_device_type = "cpu"


def main():
    global g_args, ctx_device_type
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None,
                   help="path to a Python config that sets defaults (CLI args override)")
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).resolve().parent / "runs" / "mhc")
    p.add_argument("--init_from", default="resume", choices=["scratch", "resume"])
    # mHC
    p.add_argument("--mixer", default="sinkhorn", choices=["sinkhorn", "none"],
                   help="sinkhorn: mHC (paper method); none: vanilla single-stream baseline")
    p.add_argument("--n_streams", type=int, default=4,
                   help="residual streams n (forced to 1 for --mixer=none)")
    p.add_argument("--b_res_init", type=float, default=0.0,
                   help="mHC b_res diagonal scale; 0 -> sqrt(n), e.g. 4.0 for a stricter identity init")
    p.add_argument("--dynamic_topology", dest="dynamic_topology", action="store_true", default=True,
                   help="paper-faithful per-token annotations (mHC Eq.7); default ON")
    p.add_argument("--no_dynamic_topology", dest="dynamic_topology", action="store_false",
                   help="static read/write vectors instead of per-token annotations")
    p.add_argument("--n_kv_heads", type=int, default=0,
                   help="GQA kv heads (0 -> n_head, plain MHA)")
    p.add_argument("--tie_weights", dest="tie_weights", action="store_true", default=True,
                   help="share wte/lm_head embedding; default ON in this repo")
    p.add_argument("--no_tie_weights", dest="tie_weights", action="store_false")
    # data / model (nanoGPT shakespeare_char defaults)
    p.add_argument("--dataset", type=Path, default=Path("data/shakespeare_char"))
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    # train
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--decay_lr", dest="decay_lr", action="store_true", default=True)
    p.add_argument("--no_decay_lr", dest="decay_lr", action="store_false")
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--lr_decay_iters", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "fp32", "bf16", "fp16"],
                   help="training dtype: auto -> bf16 if CUDA supports it else fp16")
    p.add_argument("--seed", type=int, default=1337)
    g_args = p.parse_args()
    args = g_args

    # --config: run a Python file that sets defaults; CLI args (passed after
    # --config) override. Keys must be valid argparse dests; unknown keys error.
    if args.config is not None:
        cfg_ns = {}
        exec(compile(open(args.config).read(), str(args.config), "exec"), {"__name__": "__config__"}, cfg_ns)
        for key, val in cfg_ns.items():
            if key.startswith("_"):
                continue
            if not hasattr(args, key):
                raise SystemExit(f"config key '{key}' is not a train.py arg")
            # only apply if CLI didn't explicitly set it (argparse defaults are already in)
            # We cannot distinguish "CLI explicitly set" from "default" here, so we
            # let CLI win when the value differs from the argparse default.
            if getattr(args, key) == p.get_default(key):
                setattr(args, key, val)
    g_args = args

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.txt", "a", encoding="utf-8")

    def log(msg):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx_device_type = device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(__file__).resolve().parent / args.dataset
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    data_dtype = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32}.get(
        meta.get("dtype", "uint16"), np.uint16)
    train_data = np.memmap(data_dir / "train.bin", dtype=data_dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=data_dtype, mode="r")

    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=args.block_size, vocab_size=vocab_size,
                      n_kv_heads=args.n_kv_heads, dynamic_topology=args.dynamic_topology,
                      tie_weights=args.tie_weights, dropout=args.dropout,
                      n_streams=args.n_streams, mixer=args.mixer, b_res_init=args.b_res_init)
    model = LlamaHC(LlamaHCConfig(**model_args)).to(device)

    # resume bookkeeping ------------------------------------------------------
    iter_num = 0
    best_val_loss = 1e9
    state = None
    if args.init_from == "resume":
        ck = find_latest_ckpt(out_dir)
        if ck is not None:
            state = torch.load(ck, map_location=device, weights_only=True)
            iter_num = state["iter_num"]
            best_val_loss = state["best_val_loss"]
            model.load_state_dict(state["model"])
            log(f"resumed from {ck.name} at iter {iter_num} (best_val {best_val_loss:.4f})")
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device)
    if args.init_from == "resume" and iter_num > 0 and state is not None:
        optimizer.load_state_dict(state["optimizer"])

    if args.compile:
        if sys.platform == "linux":
            log("using torch.compile")
            model = torch.compile(model)
        else:
            log("warning: --compile ignored (supported on linux only)")

    # dtype / autocast context (nanoGPT pattern: auto -> bf16 if supported else fp16)
    device_type = "cuda" if device == "cuda" else "cpu"
    if args.dtype == "auto":
        dtype = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if torch.cuda.is_available() else "fp32")
    else:
        dtype = args.dtype
    ptdtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    autocast_ctx = nullcontext() if (device_type == "cpu" or dtype == "fp32") else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "fp16"))
    log(f"dtype={dtype} (autocast={autocast_ctx is not nullcontext()}, GradScaler={scaler.is_enabled()})")

    metrics_f = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    ctx = (train_data, val_data, args.block_size, args.batch_size, device)
    topology = "none" if args.mixer == "none" else ("dynamic" if args.dynamic_topology else "static")

    def get_lr(it):
        if not args.decay_lr:
            return args.learning_rate
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
        if it > args.lr_decay_iters:
            return args.min_lr
        ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        return args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * ratio))

    X, Y = get_batch("train", *ctx)
    t0 = time.time()
    tokens_seen = 0
    model.train()
    log(f"start: mixer={args.mixer} topology={topology} n_streams={model.config.n_streams} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={device} "
        f"resume_iter={iter_num} max_iters={args.max_iters} out_dir={out_dir}")

    while True:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(model, ctx, args.eval_iters, autocast_ctx)
            tok_s = tokens_seen / max(time.time() - t0, 1e-9)
            log(f"iter {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                f"lr {lr:.2e}, {tok_s/1e3:.0f}k tok/s")
            metrics_f.write(json.dumps({"iter": iter_num, **losses, "lr": lr,
                                        "mixer": args.mixer, "topology": topology,
                                        "elapsed_s": round(time.time() - t0, 1)}) + "\n")
            metrics_f.flush()
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if iter_num >= args.max_iters:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_final")
            log(f"done: reached max_iters={args.max_iters}, best_val={best_val_loss:.4f}, "
                f"final ckpt saved. total {time.time()-t0:.1f}s this session")
            break

        try:
            with autocast_ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch("train", *ctx)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        except KeyboardInterrupt:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_latest")
            log(f"interrupted at iter {iter_num} -- ckpt_latest.pt saved, rerun same command to resume")
            raise

        tokens_seen += X.numel()
        iter_num += 1

        if iter_num % args.ckpt_every == 0:
            save_ckpt(model, optimizer, iter_num, best_val_loss, out_dir, "ckpt_latest")
            ms = (time.time() - t0) / max(iter_num - state["iter_num"] if state else iter_num, 1) * 1000
            log(f"iter {iter_num:5d}: checkpoint saved ({ms:.0f} ms/iter avg)")

    # final mixing sanity report
    report = model.mixing_report()
    (out_dir / "mixing_report.json").write_text(json.dumps(report, indent=2))
    if report:
        worst = max(r["row_err"] for r in report)
        log(f"mixing report: worst row_err={worst:.2e} over {len(report)} mixers -> mixing_report.json")
    else:
        log("mixing report: no mixers (mixer=none)")


if __name__ == "__main__":
    main()
