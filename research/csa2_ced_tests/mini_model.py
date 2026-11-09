"""Mini GQA transformer for the V4.1-ideas micro-benchmarks (scratch only, NOT the pipeline).

Attention-sharing patterns, scaled from the DeepSeek-V4.1-Flash report:
  own    : standard GQA baseline — every layer computes its own K/V
  shared : YOCO/CSA2-Reuse style — one layer in the upper half computes K/V
           (the owner), every later borrow layer reuses that exact K/V and
           only computes its own Q.
  proj   : CED eq.(1) verbatim — every borrow layer projects K/V from the
           boundary hidden state H_mid with per-layer weight matrices.
Optional sliding window on borrow layers = CSA2 local branch. RoPE + causal
SDPA (Turing-safe). Callers manage fp32 master + fp16 autocast + GradScaler.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.w.float()).to(dt)


class Rope:
    def __init__(self, dim, base=10000.0, max_seq=4096, device=None):
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_seq, device=device).float()
        f = torch.outer(t, inv)
        self.cos, self.sin = f.cos(), f.sin()

    def __call__(self, x):  # x: [B,H,T,Dh]
        T = x.shape[-2]
        c, s = self.cos[:T].to(x.device, x.dtype)[None, None], self.sin[:T].to(x.device, x.dtype)[None, None]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def swa_mask(T, window, device, dtype):
    row = torch.arange(T, device=device)[:, None]
    col = torch.arange(T, device=device)[None, :]
    keep = (col <= row) & (col > row - window)
    m = torch.zeros(T, T, device=device, dtype=dtype)
    m.masked_fill_(~keep, float("-inf"))
    return m


class Attention(nn.Module):
    """kv_mode: own | borrow_proj | borrow_shared; window=None -> full context."""

    def __init__(self, d, n_heads, kv_heads, kv_mode, window):
        super().__init__()
        assert d % n_heads == 0 and n_heads % kv_heads == 0
        self.h, self.kvh, self.dh, self.d = n_heads, kv_heads, d // n_heads, d
        self.g = n_heads // kv_heads
        self.kv_mode, self.window = kv_mode, window
        self.rope = None  # injected by Block
        self.q = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if kv_mode == "own":
            self.k = nn.Linear(d, kv_heads * self.dh, bias=False)
            self.v = nn.Linear(d, kv_heads * self.dh, bias=False)
        elif kv_mode == "borrow_proj":
            self.kp = nn.Linear(d, kv_heads * self.dh, bias=False)
            self.vp = nn.Linear(d, kv_heads * self.dh, bias=False)

    def forward(self, h, boundary=None, reused=None):
        B, T, _ = h.shape
        q = self.q(h).view(B, T, self.h, self.dh).transpose(1, 2)
        if self.kv_mode == "own":
            k = self.k(h).view(B, T, self.kvh, self.dh).transpose(1, 2)
            v = self.v(h).view(B, T, self.kvh, self.dh).transpose(1, 2)
            k = self.rope(k)
        elif self.kv_mode == "borrow_proj":
            k = self.kp(boundary).view(B, T, self.kvh, self.dh).transpose(1, 2)
            v = self.vp(boundary).view(B, T, self.kvh, self.dh).transpose(1, 2)
            k = self.rope(k)
        else:
            k, v = reused  # rope applied once by the owner layer
        q = self.rope(q)
        if self.g > 1:
            kk = k.repeat_interleave(self.g, dim=1)
            vv = v.repeat_interleave(self.g, dim=1)
        else:
            kk, vv = k, v
        mask = swa_mask(T, self.window, h.device, q.dtype) if self.window else None
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask,
                                             is_causal=mask is None)
        out = out.transpose(1, 2).reshape(B, T, self.d)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.up = nn.Linear(d, ffn, bias=False)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, kv_heads, kv_mode, window, rope, d_ffn):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, kv_heads, kv_mode, window)
        self.attn.rope = rope
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ffn)

    def forward(self, h, boundary=None, reused=None):
        h = h + self.attn(self.n1(h), boundary, reused)
        h = h + self.mlp(self.n2(h))
        return h


class MiniLM(nn.Module):
    """layers total; top half borrows KV from boundary hidden state H_mid (output
    of block mid-1). borrow_mode: None | shared | proj. window: SWA length on
    borrow layers (None = full borrowed context)."""

    def __init__(self, vocab, d=256, layers=6, n_heads=8, kv_heads=4, ffn=None,
                 borrow_mode=None, window=None, rope_theta=10000.0, device=None):
        super().__init__()
        assert layers % 2 == 0
        self.vocab, self.d, self.layers = vocab, d, layers
        self.mid = layers // 2
        self.borrow_mode = borrow_mode
        self.embed = nn.Embedding(vocab, d)
        self.rope = Rope(d // n_heads, base=rope_theta, max_seq=4096, device=device)
        d_ffn = ffn if ffn else int(d * 8 / 3)
        self.blocks = nn.ModuleList()
        for l in range(layers):
            in_top = l >= self.mid
            if not in_top or borrow_mode is None:
                kv_mode = "own"
            elif borrow_mode == "shared":
                kv_mode = "own" if l == self.mid else "borrow_shared"
            else:
                kv_mode = "borrow_proj"
            window_l = window if (in_top and borrow_mode is not None
                                  and kv_mode != "own") else None
            self.blocks.append(Block(d, n_heads, kv_heads, kv_mode, window_l,
                                     self.rope, d_ffn))
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied embeddings (repo convention)
        self.apply(self._init)
        if device is not None:
            self.to(device)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        h = self.embed(idx)
        boundary, reused = None, None
        for i, blk in enumerate(self.blocks):
            mode = blk.attn.kv_mode
            if mode == "own":
                if i == self.mid and self.borrow_mode == "shared":
                    # owner layer (YOCO producer): K/V from its own pre-attn input
                    B, T, _ = h.shape
                    a = blk.attn
                    src = blk.n1(h)
                    k = a.k(src).view(B, T, a.kvh, a.dh).transpose(1, 2)
                    v = a.v(src).view(B, T, a.kvh, a.dh).transpose(1, 2)
                    reused = (a.rope(k), v)
                h = blk(h)
                if i == self.mid - 1:
                    boundary = h  # == owner input in shared mode (output of mid-1)
            elif mode == "borrow_proj":
                h = blk(h, boundary=boundary)
            else:                                            # borrow_shared
                h = blk(h, reused=reused)
        hf = self.nf(h)
        return self.head(hf)