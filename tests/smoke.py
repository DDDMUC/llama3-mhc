"""tests/smoke.py -- standalone smoke suite for the llama3-mhc repo.
Run:  python tests/smoke.py   (auto-uses CUDA if available)
Covers: RoPE math, weight tying + gradient flow, mHC mixer double
stochasticity, static + paper dynamic topology, vanilla none path,
b_res_init knob, optimizer decay grouping, autoregressive generate,
short real-data training.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama3_mhc import LlamaHC, LlamaHCConfig, apply_rope, precompute_rope  # noqa: E402
from llama3_mhc.mixers import SinkhornMHCResidual, doubly_stochastic_error  # noqa: E402


def grads_finite(model):
    return all((p.grad is None) or torch.isfinite(p.grad).all().item() for p in model.parameters())


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    X = torch.randint(0, 65, (2, 64), device=dev)
    Y = torch.randint(0, 65, (2, 64), device=dev)

    # 1) RoPE math
    hs, T = 16, 32
    cos, sin = precompute_rope(hs, T, 500000.0, dev)
    x = torch.randn(2, 3, T, hs, device=dev)
    y = apply_rope(x, cos, sin)
    nx = x[..., :hs // 2] ** 2 + x[..., hs // 2:] ** 2
    ny = y[..., :hs // 2] ** 2 + y[..., hs // 2:] ** 2
    assert (nx - ny).abs().max() < 1e-4, "RoPE norm preservation failed"
    v = torch.randn(1, 1, 1, hs, device=dev)
    q = v.expand(1, 1, T, hs)
    k = torch.randn(1, 1, 1, hs, device=dev).expand(1, 1, T, hs)
    s = torch.einsum("bhmt,bhnt->bhmn", apply_rope(q, cos, sin), apply_rope(k, cos, sin))
    assert (s[0, 0, :-1, :-1] - s[0, 0, 1:, 1:]).abs().max() < 1e-3, "RoPE shift invariance failed"
    print("RoPE math                      OK")

    # 2) weight tying + gradient flow (repo defaults: mHC, dynamic topology)
    base = dict(n_layer=2, n_head=6, n_kv_heads=2, n_embd=192, block_size=64,
                vocab_size=65, mixer="sinkhorn", dynamic_topology=True)
    m = LlamaHC(LlamaHCConfig(**base)).to(dev)  # repo default: tied
    assert m.lm_head.weight is m.transformer.wte.weight, "weights not tied"
    _, loss = m(X, Y)
    loss.backward()
    assert grads_finite(m)
    assert m.transformer.wte.weight.grad is not None and m.transformer.wte.weight.grad.abs().sum() > 0
    print(f"tying + grads                  OK (params {sum(p.numel() for p in m.parameters())/1e6:.3f}M)")

    # 3) mHC mixer: double stochasticity (row/col sums == 1) + static topology
    errs = doubly_stochastic_error(SinkhornMHCResidual(n_streams=4).mixing_matrices()[1])
    assert max(errs["row_err"], errs["col_err"]) < 1e-6 and errs["min_entry"] >= 0, errs
    mm = LlamaHC(LlamaHCConfig(**{**base, "dynamic_topology": False})).to(dev)
    _, l = mm(X, Y)
    l.backward()
    assert grads_finite(mm)
    print("mHC mixer (static + dbl-stoch) OK")

    # 4) paper dynamic topology (Eq.7 + Eq.8-9) + mixing report
    rep = m.mixing_report()
    assert len(rep) == 4, f"expected 2 stages x 2 layers, got {len(rep)}"
    assert max(r["row_err"] for r in rep) < 1e-5, rep
    blk = m.transformer.h[0]
    assert abs(float(blk.alpha["attn_pre"].detach()) - 0.01) < 1e-9, "alpha must init to the paper's 0.01"
    # dynamic blocks must not carry unused static-path params (DDP safety)
    assert blk.read_attn is None and blk.mixer_attn is None, "dynamic topology must not build static params"
    dead = ("read_attn", "read_mlp", "write_attn", "write_mlp")
    assert not any(n_.endswith(k) for n_, _ in m.named_parameters() for k in dead), \
        "dead static-path parameters found under dynamic topology"
    # capture the stream tensor entering block 0: all n streams must replicate the embedding
    seen = {}
    h = blk.register_forward_hook(lambda mod, inp, out: seen.setdefault("S", inp[0].detach()))
    m(X)
    h.remove()
    S = seen["S"]
    assert torch.allclose(S[:, :, 0], S[:, :, -1]), "streams must start as replicated embedding"
    # b_res_init knob: diagonal scale must land exactly on b_res
    mb = LlamaHC(LlamaHCConfig(**{**base, "b_res_init": 4.0})).to(dev)
    assert torch.allclose(mb.transformer.h[0].b_res["attn"], 4.0 * torch.eye(4, device=dev)), \
        "b_res_init must scale the identity diagonal"
    print("dynamic topology (Eq.7+Eq.9)   OK")

    # 5) vanilla none path: single stream, no mixer, empty report
    nm = LlamaHC(LlamaHCConfig(**{**base, "mixer": "none"})).to(dev)
    assert nm.config.n_streams == 1, "mixer=none must force a single stream"
    assert nm.transformer.h[0].mixer_attn is None, "mixer=none must not build mixers"
    _, l = nm(X, Y)
    l.backward()
    assert grads_finite(nm)
    assert nm.mixing_report() == []
    print("vanilla none path              OK")

    # 6) optimizer decay grouping + autoregressive generate path
    o = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    assert len(o.param_groups) == 2, "expected decay + no-decay groups"
    decay_ids = {id(p) for p in o.param_groups[0]["params"]}
    nodecay_ids = {id(p) for p in o.param_groups[1]["params"]}
    assert not decay_ids & nodecay_ids, "param groups overlap"
    name_of = {id(p): n for n, p in m.named_parameters()}
    assert {name_of[i] for i in decay_ids} | {name_of[i] for i in nodecay_ids} == set(name_of.values()), \
        "groups must cover every parameter exactly once"
    protected = ("theta", "b_res", "b_post", "b_pre", "alpha", "read_", "write_", "norm_", "ln_f", "mixer")
    for n_, p in m.named_parameters():
        if any(k in n_ for k in protected):
            assert id(p) in nodecay_ids, f"{n_} must be weight-decay free"
    for n_ in ("transformer.wte.weight", "transformer.h.0.attn.wq.weight",
               "transformer.h.0.mlp.w_gate.weight"):
        assert id(dict(m.named_parameters())[n_]) in decay_ids, f"{n_} must receive weight decay"
    # targets=None branch + top-k sampling (generate was previously uncovered)
    with torch.no_grad():
        logits, _ = m(X[:, :8])
        assert logits.shape == (2, 1, 65), "targets=None must return last-position logits"
        out = m.generate(X[:, :8], 5, temperature=0.8, top_k=10)
    assert out.shape == (2, 13) and out.min() >= 0 and out.max() < 65
    print("optimizer groups + generate    OK")

    # 7) short real-data training (paper path: dynamic mHC)
    data_dir = Path(__file__).resolve().parent.parent / "data" / "shakespeare_char"
    if (data_dir / "train.bin").exists():
        td = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        ix = torch.randint(len(td) - 256, (8,))
        x = torch.stack([torch.from_numpy(td[i:i + 256].astype(np.int64)) for i in ix]).to(dev)
        y = torch.stack([torch.from_numpy(td[i + 1:i + 257].astype(np.int64)) for i in ix]).to(dev)
        cfg = LlamaHCConfig(n_layer=4, n_head=6, n_kv_heads=2, n_embd=256, block_size=256,
                            vocab_size=65, mixer="sinkhorn")
        mm = LlamaHC(cfg).to(dev)
        o = mm.configure_optimizers(0.1, 1e-3, (0.9, 0.95), dev)
        ls = []
        for _ in range(20):
            _, l = mm(x, y)
            o.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
            o.step()
            ls.append(l.item())
        assert ls[-1] < ls[0], "training did not converge"
        print(f"20-step real training          OK ({ls[0]:.3f} -> {ls[-1]:.3f})")
    else:
        print("20-step real training          SKIPPED (prepare data first)")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
