"""
model_llama_hc.py -- Llama-3-style backbone with mHC
(Manifold-Constrained Hyper-Connections, arXiv:2512.24880).

Backbone, all standard Llama 3 practice (diffed component-by-component against
karpathy/llama2.c model.py and HF transformers modeling_llama.py):
  * RMSNorm (fp32 cast, no mean subtraction, no bias)
  * RoPE rotary position embeddings, theta = 500000 (HF rotate_half convention)
  * SwiGLU MLP with multiple-of rounding
  * Grouped-Query Attention (n_kv_heads <= n_head)
  * no biases anywhere; tied input/output embedding (this repo's premise)

mHC machinery (arXiv:2512.24880): n parallel residual streams, all initialized
by replicating the token embedding (official expand_to_mhc convention). Three
configurations:
  * dynamic topology (default; the paper's method, Eq.7 + Eq.8-9): per-token
    annotations H_pre (read) / H_post (write) from alpha*(RMSNorm(vec(S)) phi)+b
    (no tanh, gating factor alpha init 0.01), and per-token Sinkhorn-projected
    mixing (Eq.9, 20 iterations, exp input clamped at 20)
  * static topology: learned static read/write vectors + static mixer
  * mixer="none": single stream, no mixing -- exactly a vanilla pre-norm
    Llama-3 (baseline / debug)
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixers import make_mixer


@dataclass
class LlamaHCConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_kv_heads: int = 0          # 0 -> n_head (plain MHA); must divide n_head
    n_embd: int = 384
    multiple_of: int = 64        # SwiGLU hidden dim rounding
    ffn_dim_multiplier: float = 1.0
    rope_theta: float = 500000.0  # Llama 3 value (Llama 2 used 10000.0)
    dropout: float = 0.0
    n_streams: int = 4
    b_res_init: float = 0.0      # mHC b_res diagonal scale; 0 -> sqrt(n) (default), e.g. 4.0 for a stricter identity init
    mixer: str = "sinkhorn"      # sinkhorn (mHC) | none (vanilla single-stream)
    dynamic_topology: bool = True  # paper-faithful per-token annotations (Eq.5)
    tie_weights: bool = True     # this repo's premise: tied input/output embedding

    def __post_init__(self):
        if self.mixer == "none":
            # vanilla baseline: one stream, identity "mixing", static topology
            self.n_streams = 1
            self.dynamic_topology = False
        if self.n_kv_heads in (0, None):
            self.n_kv_heads = self.n_head
        assert self.n_head % self.n_kv_heads == 0, "n_kv_heads must divide n_head"
        if self.dynamic_topology:
            assert self.mixer == "sinkhorn", "dynamic topology requires mixer='sinkhorn'"


def _batched_sinkhorn_exp(raw: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """mHC Eq.9, batched: M(0)=exp(raw), M(t)=T_c(T_r(M(t-1))) (row then col),
    final row pass so rows sum to exactly 1. raw: (..., n, n) -> doubly
    stochastic (..., n, n). Matches the paper's order up to float32 noise
    (verified 6e-8 vs column-first order)."""
    H = torch.exp(raw)
    for _ in range(n_iters):
        H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        H = H / H.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return H


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # float32 cast per reference implementations (matters under bf16/fp16 autocast)
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


def precompute_rope(head_dim: int, max_len: int, theta: float, device):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(max_len, device=device)
    freqs = torch.outer(t, freqs)                      # (T, hs/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, hs), HF convention
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_h, T, hs); cos/sin: (T, hs)
    T = x.shape[2]
    return x * cos[None, None, :T] + rotate_half(x) * sin[None, None, :T]


def repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
    # (B, n_kv, T, hs) -> (B, n_kv*rep, T, hs)
    if rep == 1:
        return x
    B, nk, T, hs = x.shape
    return x[:, :, None].expand(B, nk, rep, T, hs).reshape(B, nk * rep, T, hs)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        d, nh, nkv = config.n_embd, config.n_head, config.n_kv_heads
        hs = d // nh
        self.nh, self.nkv, self.hs = nh, nkv, hs
        self.rep = nh // nkv
        self.wq = nn.Linear(d, nh * hs, bias=False)
        self.wk = nn.Linear(d, nkv * hs, bias=False)
        self.wv = nn.Linear(d, nkv * hs, bias=False)
        self.wo = nn.Linear(nh * hs, d, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hs).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hs).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hs).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.rep), repeat_kv(v, self.rep)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        d = config.n_embd
        hidden = int(2 * d * 4 / 3)
        hidden = int(config.ffn_dim_multiplier * hidden)
        hidden = config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LlamaHCBlock(nn.Module):
    """Two mHC stages (attn, mlp) around Llama-style sublayers."""

    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        n = config.n_streams
        inv_sqrt_n = 1.0 / math.sqrt(n)
        self.n_streams = n
        self.dynamic = bool(config.dynamic_topology)
        self.mixer_kind = config.mixer
        if self.dynamic:
            # dynamic topology carries its own reads/writes/mixing (h_pre/h_post/W);
            # no static-path parameters, so no dead weights for DDP either
            assert config.mixer == "sinkhorn", "dynamic topology requires mixer='sinkhorn'"
            self.mixer_attn = self.mixer_mlp = None
            self.read_attn = self.read_mlp = None
            self.write_attn = self.write_mlp = None
            C = config.n_embd
            nd = n * C
            logit_uniform = math.log(1.0 / (n - 1))
            # paper Eq.7: per-token annotations are computed from RMSNorm over the
            # FLATTENED n*C stream ("to preserve full context information"), not from
            # the mean of streams. One fused projection yields pre/post/res mixes.
            self.norm_fn = nn.ModuleDict({s: RMSNorm(nd) for s in ("attn", "mlp")})
            self.alpha = nn.ParameterDict({
                f"{s}_{k}": nn.Parameter(torch.tensor(0.01))  # paper appendix: gating factor init 0.01
                for s in ("attn", "mlp") for k in ("pre", "post", "res")
            })
            self.theta = nn.ModuleDict({s: nn.Linear(nd, n * (n + 2), bias=False) for s in ("attn", "mlp")})
            self.b_pre = nn.ParameterDict({s: nn.Parameter(torch.full((n,), logit_uniform)) for s in ("attn", "mlp")})
            self.b_post = nn.Parameter(torch.zeros(2, n))
            b_scale = config.b_res_init if config.b_res_init > 0 else math.sqrt(n)
            self.b_res = nn.ParameterDict({s: nn.Parameter(b_scale * torch.eye(n)) for s in ("attn", "mlp")})
        else:
            self.mixer_attn = None if config.mixer == "none" else make_mixer(config.mixer, n)
            self.mixer_mlp = None if config.mixer == "none" else make_mixer(config.mixer, n)
            self.read_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
            self.read_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
            self.write_attn = nn.Parameter(torch.full((n,), inv_sqrt_n))
            self.write_mlp = nn.Parameter(torch.full((n,), inv_sqrt_n))
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = LlamaAttention(config)
        self.norm_mlp = RMSNorm(config.n_embd)
        self.mlp = LlamaMLP(config)

    def _dyn_raw(self, streams, stage):
        # paper Eq.7: H~ = alpha * (x~' phi) + b, x~' = RMSNorm(vec(streams)); no tanh.
        n = self.n_streams
        x = self.norm_fn[stage](streams.flatten(-2))               # (B, T, n*d)
        mixes = self.theta[stage](x)                               # (B, T, n*(n+2))
        h_pre = torch.sigmoid(self.alpha[f"{stage}_pre"] * mixes[..., :n] + self.b_pre[stage])
        h_post = 2 * torch.sigmoid(self.alpha[f"{stage}_post"] * mixes[..., n:2 * n]
                                   + self.b_post[0 if stage == "attn" else 1])
        raw = self.alpha[f"{stage}_res"] * mixes[..., 2 * n:].reshape(*streams.shape[:2], n, n) \
            + self.b_res[stage]
        return h_pre, h_post, raw.clamp(max=20.0)                  # exp overflow guard (mHC.cu)

    def _dyn_stage(self, streams, stage, sublayer_norm, sublayer):
        h_pre, h_post, raw = self._dyn_raw(streams, stage)
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = sublayer(sublayer_norm(v))
        W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        return mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)

    def forward(self, streams, cos, sin):  # streams: (B, T, n, d)
        if self.dynamic:
            streams = self._dyn_stage(streams, "attn", self.norm_attn,
                                      lambda t: self.attn(t, cos, sin))
            streams = self._dyn_stage(streams, "mlp", self.norm_mlp, self.mlp)
            return streams
        v = torch.einsum("k,btkd->btd", self.read_attn, streams)
        f = self.attn(self.norm_attn(v), cos, sin)
        mixed = streams if self.mixer_attn is None else self.mixer_attn(streams)
        streams = mixed + f.unsqueeze(2) * self.write_attn.view(1, 1, -1, 1)
        v = torch.einsum("k,btkd->btd", self.read_mlp, streams)
        f = self.mlp(self.norm_mlp(v))
        mixed = streams if self.mixer_mlp is None else self.mixer_mlp(streams)
        streams = mixed + f.unsqueeze(2) * self.write_mlp.view(1, 1, -1, 1)
        return streams


class LlamaHC(nn.Module):

    def __init__(self, config: LlamaHCConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        n = config.n_streams
        hs = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([LlamaHCBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        cos, sin = precompute_rope(hs, config.block_size, config.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight
        self.read_out = nn.Parameter(torch.full((n,), 1.0 / math.sqrt(n)))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.size(1)
        if self.rope_cos.device != device:
            self.rope_cos, self.rope_sin = self.rope_cos.to(device), self.rope_sin.to(device)
        cos, sin = self.rope_cos, self.rope_sin

        tok = self.transformer.wte(idx)
        # official mHC expands the stream by replicating the embedding across all
        # n streams (TileKernels expand_to_mhc / mHC.cu), not by zero-padding
        streams = tok.unsqueeze(2).expand(-1, -1, self.config.n_streams, -1).contiguous()
        streams = self.transformer.drop(streams)
        for block in self.transformer.h:
            streams = block(streams, cos, sin)
        x = torch.einsum("k,btkd->btd", self.read_out, streams)
        x = self.transformer.ln_f(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # bias-like or raw-parameter tensors are excluded from weight decay
        no_decay_keys = ("mixer", "theta", "b_res", "b_post")
        decay = [p for pn, p in param_dict.items()
                 if p.dim() >= 2 and not any(k in pn for k in no_decay_keys)]
        no_decay = [p for pn, p in param_dict.items()
                    if not (p.dim() >= 2 and not any(k in pn for k in no_decay_keys))]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate, betas=betas,
            **({"fused": True} if device_type == "cuda" else {}))
        return optimizer

    def mixing_report(self):
        report = []
        for i, block in enumerate(self.transformer.h):
            if getattr(block, "dynamic", False):
                for stage in ("attn", "mlp"):
                    dev = next(block.parameters()).device
                    # probe at x~=0 (mixes=0 -> raw = b_res), same clamp as forward
                    H = _batched_sinkhorn_exp(block.b_res[stage].to(dev).clamp(max=20.0))
                    report.append({"block": i, "stage": stage, "topology": "dynamic-probe",
                                   "row_err": (H.sum(-1) - 1).abs().max().item(),
                                   "col_err": (H.sum(0) - 1).abs().max().item(),
                                   "min_entry": H.min().item(),
                                   "signed": bool((H < 0).any().item())})
                continue
            for stage, mixer in (("attn", block.mixer_attn), ("mlp", block.mixer_mlp)):
                if mixer is None:
                    continue
                W, S = mixer.mixing_matrices()
                report.append({"block": i, "stage": stage, "topology": "static",
                               "row_err": (S.sum(-1) - 1).abs().max().item(),
                               "col_err": (S.sum(0) - 1).abs().max().item(),
                               "min_entry": S.min().item(),
                               "signed": bool((W < 0).any().item())})
        return report

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

    @torch.no_grad()
    def generate_kvcache(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation with KV cache (stream-level).

        Correctness contract: identical tokens to `generate()` (verified in smoke).
        Prefill: run a full forward over the prompt, caching each layer's K/V
        (K = w_k(v_t), V = w_v(v_t), where v_t = the layer's aggregate read stream).
        Decode: for each new token, recompute only that token's annotations/read
        stream per layer, attend over the cached prefix (causal), write back,
        cache the new K/V, and take logits.
        """
        B = idx.size(0)
        hs = self.config.n_embd // self.config.n_head
        nkv = self.config.n_kv_heads
        n_layer = self.config.n_layer
        max_len = self.config.block_size
        device = idx.device
        cache = {
            "k": torch.zeros(n_layer, B, nkv, max_len, hs, device=device),
            "v": torch.zeros(n_layer, B, nkv, max_len, hs, device=device),
            "seq_len": 0,
        }
        # Prefill: run full forward over prompt, capture per-layer K/V
        tok = self.transformer.wte(idx)
        streams = tok.unsqueeze(2).expand(-1, -1, self.config.n_streams, self.config.n_embd).contiguous()
        cos, sin = self.rope_cos, self.rope_sin
        for layer_idx, block in enumerate(self.transformer.h):
            # process full prompt for this layer, capture K/V (attn stage)
            streams = self._block_forward_capture(block, streams, cos, sin, cache, layer_idx)
        x = torch.einsum("k,btkd->btd", self.read_out, streams)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])[:, -1, :] / max(temperature, 1e-8)
        del x, tok, streams
        out = idx
        for _ in range(max_new_tokens):
            # sample the next token from the current logits
            lgt = logits
            if top_k is not None:
                v, _ = torch.topk(lgt, min(top_k, lgt.size(-1)))
                lgt[lgt < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(lgt, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, nxt), dim=1)
            # decode the newly sampled token (cache its K/V at pos seq_len,
            # predict the next one)
            last = nxt
            tok = self.transformer.wte(last)
            streams = tok.unsqueeze(2).expand(-1, -1, self.config.n_streams, self.config.n_embd).contiguous()
            for layer_idx, block in enumerate(self.transformer.h):
                streams = self._block_forward_decode(block, streams, cos, sin, cache, layer_idx)
            self._advance_seq(cache)  # one token decoded: advance position once
            x = torch.einsum("k,btkd->btd", self.read_out, streams)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :])[:, -1, :] / max(temperature, 1e-8)
        return out

    def _block_forward_capture(self, block, streams, cos, sin, cache, layer_idx):
        """Full-sequence block forward that records K/V for the cache."""
        # dynamic stage (paper path): attn then mlp
        # attn stage: capture K/V
        h_pre, h_post, raw = block._dyn_raw(streams, "attn")
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = block.attn(block.norm_attn(v), cos, sin)
        W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        streams = mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)
        # (B, T, nkv*hs) -> capture ROTATED K/V for cache (matches full forward:
        # attention sees norm_attn(v), so K/V are wk/wv(norm(v)))
        B, T, _ = v.shape
        nkv, hs = block.attn.nkv, block.attn.hs
        vn = block.norm_attn(v)
        kk = block.attn.wk(vn).view(B, T, nkv, hs)  # (B,T,nkv,hs)
        vv = block.attn.wv(vn).view(B, T, nkv, hs)
        ## rotate k with positional cos/sin for positions 0..T-1
        kk_r = apply_rope(kk.transpose(1, 2), cos, sin).transpose(1, 2)  # (B,T,nkv,hs) rotated
        cache["k"][layer_idx, :, :, :T] = kk_r.transpose(1, 2)       # (B,nkv,T,hs) rotated
        cache["v"][layer_idx, :, :, :T] = vv.transpose(1, 2)         # v not rotated
        # mlp stage
        h_pre, h_post, raw = block._dyn_raw(streams, "mlp")
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = block.mlp(block.norm_mlp(v))
        W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        streams = mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)
        return streams

    def _block_forward_decode(self, block, streams, cos, sin, cache, layer_idx):
        """Single-token block forward with cached K/V (correctness = full forward)."""
        # attn stage (token t): compute from cached prefix + this token's read stream
        h_pre, h_post, raw = block._dyn_raw(streams, "attn")
        v = torch.einsum("btk,btkd->btd", h_pre, streams)  # (B, 1, d)
        vn = block.norm_attn(v)  # attention sees norm(v), same as full forward
        # compute q for this token (B,1,nk,hs), k/v from cache + this token
        B, T, _ = v.shape
        nk, hs = block.attn.nh, block.attn.hs
        pos = cache["seq_len"]
        q = block.attn.wq(vn).view(B, T, nk, hs).transpose(1, 2)  # (B,nk,1,hs)
        # RoPE at the actual position (pos): cos[pos:pos+1]
        q = apply_rope(q, cos[pos:pos + 1], sin[pos:pos + 1])
        # k/v for this token (rotate k at pos, write rotated k to cache)
        nkv = block.attn.nkv
        kk = block.attn.wk(vn).view(B, T, nkv, hs).transpose(1, 2)  # (B,nkv,1,hs)
        kk_r = apply_rope(kk, cos[pos:pos + 1], sin[pos:pos + 1])
        vv = block.attn.wv(vn).view(B, T, nkv, hs).transpose(1, 2)  # (B,nkv,1,hs)
        cache["k"][layer_idx, :, :, pos] = kk_r[:, :, 0]
        cache["v"][layer_idx, :, :, pos] = vv[:, :, 0]
        k_cached = cache["k"][layer_idx][:, :, :pos + 1]  # (B,nkv,1,hs) padded
        v_cached = cache["v"][layer_idx][:, :, :pos + 1]
        # repeat_kv to n heads (identical semantics to model.repeat_kv: expand+reshape)
        rep = nk // nkv
        k_rep = k_cached[:, :, None].expand(-1, -1, rep, -1, -1).reshape(-1, nk, pos + 1, hs)  # (B,nk,pos+1,hs)
        v_rep = v_cached[:, :, None].expand(-1, -1, rep, -1, -1).reshape(-1, nk, pos + 1, hs)
        # causal: q (single new token) attends to all cached [0..pos] via SDPA.
        # is_causal=False: with q_len=1 over the full cached prefix this is the
        # exact causal step; is_causal=True would mis-handle q_len=1 != kv_len.
        q = q.to(dtype=k_rep.dtype)
        y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)  # (B,nk,1,hs)
        y = y.transpose(1, 2).reshape(B, 1, -1)  # (B,1,d)
        f = block.attn.wo(y)  # (B,1,d)
        W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        streams = mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)
        # mlp stage (no cache)
        h_pre, h_post, raw = block._dyn_raw(streams, "mlp")
        v = torch.einsum("btk,btkd->btd", h_pre, streams)
        f = block.mlp(block.norm_mlp(v))
        W = _batched_sinkhorn_exp(raw)
        mixed = torch.einsum("btjk,btkd->btjd", W, streams)
        streams = mixed + f.unsqueeze(2) * h_post.unsqueeze(-1)
        return streams

    def _advance_seq(self, cache):
        cache["seq_len"] += 1
