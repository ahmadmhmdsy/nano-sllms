"""Bidirectional char Transformer encoder + 15-class head (D1, DESIGN section 2).

Llama-style blocks (RMSNorm + RoPE + SwiGLU + GQA) with the causal mask OFF —
SDPA (sm_75/Turing: no flash-attn, fp16-only). The classification head is a
single shared linear (hidden -> 15) applied only at Arabic base positions
(passed in for every batch); passthrough positions carry label -1 and are
ignored by the loss, but remain VISIBLE as context tokens.

Naming/config keys follow the repo's diac_* config family. Local code with
the same block math as the M3 line (no M3-line file touched).

Shared-KV extension (E-62, config-gated, default OFF):
Each layer carries a kv_mode resolved from model.kv_pattern:
  new   - layer computes (and writes) its own K/V (standard behaviour).
  reuse - layer reads the PREVIOUS writing layer's stored ("past") K/V from
          the shared per-forward cache and re-ropes them at its own positions.
  edit  - reuse + a learned SharedKVEditor on the raw cached K/V: a
          tanh-gated residual reindex blend over {identity, roll+1, roll-1}
          along the sequence axis plus per-kv-head residual scales
          (all init => identity).
The cache stores PRE-RoPE K/V so every consumer adapts them to its own
position frame (rotate/reindex/v-scale are per-consumer, cache is shared).
Default (no model.kv_pattern) keeps every layer "new": byte-identical
behaviour, params and state-dict keys vs. the pre-extension model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

KV_MODES = ("new", "reuse", "edit")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dtype = torch.float32

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


_ROPE_CACHE = {}

def rope_freqs(seq_len, head_dim, device, theta=10000.0):
    """Cached (train-speed optimization 2026-09-24): every layer re-computed
    cos/sin from scratch each train step; identical output, zero kernel work
    after warmup. Cache is tiny and keyed defensively."""
    key = (seq_len, head_dim, str(device), theta)
    hit = _ROPE_CACHE.get(key)
    if hit is not None:
        return hit
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)                      # [L, hd/2]
    pair = (torch.cos(f), torch.sin(f))
    if len(_ROPE_CACHE) > 64:
        _ROPE_CACHE.clear()
    _ROPE_CACHE[key] = pair
    return pair


def apply_rope(x, cos, sin):
    """x [B, H, L, D]; rotate even/odd pairs (head_dim even assumed)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SharedKVEditor(nn.Module):
    """Learned per-layer edit of a shared/past KV read (init == exact identity).

    reindex gates [2]: tanh-gated residual reindex along the sequence axis
    (axis 2 of [B, n_kv, L, D]): both k and v get
    x' = x + a1*(roll(x,-1)-x) + a2*(roll(x,+1)-x), a_i = tanh(gate_i),
    gates init 0 => x' == x at init. k/v reindex TOGETHER (a (k_i, v_i)
    entry is a paired slot; reindex points stay coherent).
    k_scale/v_scale [n_kv]: per-kv-head multiplicative residual 1+tanh(s),
    s init 0 => factor 1 at init.
    """

    def __init__(self, n_kv):
        super().__init__()
        self.reindex_gates = nn.Parameter(torch.zeros(2))
        self.k_scale = nn.Parameter(torch.zeros(n_kv))
        self.v_scale = nn.Parameter(torch.zeros(n_kv))

    def forward(self, k, v):
        # k, v: [B, n_kv, L, D] raw (pre-rope) cache entries
        a = torch.tanh(self.reindex_gates.float())
        k = k + a[0] * (torch.roll(k, -1, dims=2) - k) \
              + a[1] * (torch.roll(k, 1, dims=2) - k)
        v = v + a[0] * (torch.roll(v, -1, dims=2) - v) \
              + a[1] * (torch.roll(v, 1, dims=2) - v)
        k = k * (1.0 + torch.tanh(self.k_scale.float()))[:, None, None]
        v = v * (1.0 + torch.tanh(self.v_scale.float()))[:, None, None]
        return k.to(v.dtype), v


class GQAAttention(torch.nn.Module):
    def __init__(self, hidden, n_heads, n_kv, causal=False, kv_mode="new",
                 uses_edit=False):
        super().__init__()
        assert hidden % n_heads == 0
        assert kv_mode in KV_MODES, kv_mode
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = hidden // n_heads
        self.kv_rep = n_heads // n_kv
        self.causal = causal
        self.kv_mode = kv_mode
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, n_kv * self.head_dim, bias=False)
        self.v = torch.nn.Linear(hidden, n_kv * self.head_dim, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)
        # shared-layer arms cycle modules over more USES than modules; a use
        # may override the module's registered mode, so the editor exists when
        # ANY use can be 'edit' (default behaviour byte-identical: no override
        # -> editor only when kv_mode == 'edit').
        self.editor = SharedKVEditor(n_kv) if (kv_mode == "edit" or uses_edit) else None

    def forward(self, x, cache=None, kv_mode=None, attn_mask=None):
        """cache: shared per-forward list of (k, v) pre-rope entries.

        mode 'new': compute own K/V, WRITE (k, v) to the cache.
        mode reuse/edit: READ the last cache entry (past KV), optionally edit
        it, re-rope at this layer's positions, and attend. The kv_mode arg
        overrides the registered mode for THIS use (shared-layer arms cycle
        per-use semantics; default None = the module's own mode).
        """
        mode = kv_mode or self.kv_mode
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if mode == "new":
            k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            if cache is not None:
                cache.append((k, v))
        else:
            assert cache, "reuse/edit layer needs a prior new-mode KV entry"
            k, v = cache[-1]
            if mode == "edit" and self.editor is not None:
                k, v = self.editor(k, v)
        cos, sin = rope_freqs(T, self.head_dim, x.device)
        q = apply_rope(q, cos, sin).to(v.dtype)   # fp32 rope math -> cast back (fp16-only GPU)
        k = apply_rope(k, cos, sin).to(v.dtype)
        if self.kv_rep > 1:
            k = k.repeat_interleave(self.kv_rep, dim=1)
            v = v.repeat_interleave(self.kv_rep, dim=1)
        # SDPA fast kernels (flash/efficient) require head_dim % 8 == 0; with
        # e.g. hidden 160/8=20 (or 272/8=34) every call silently falls back to
        # the MATH path which decomposes to FP32 SGEMM bmms on this fp16-only
        # Turing GPU (train-speed profiling 2026-09-24: 5.5 ms vs 1.5 ms).
        # For those shapes run the explicit math in fp16 gemms (+fp32 softmax,
        # correctness-consistent with the fp32-scale softmax). For %8==0 we
        # keep the byte-identical legacy SDPA call.
        # SDPA FORBIDS is_causal together with an explicit attn_mask
        # ("Explicit attn_mask should not be set when is_causal=True"; torch 2.14
        # raises RuntimeError). Every bench.py piece arrives WITH a boolean
        # key-padding mask, so a causal arm benched through the registered decode
        # path could never run at all - the reason no causal D-line arm has ever
        # been measured, not a modelling result. Fold causality into that boolean
        # mask instead: verified equivalent to the math path to fp32 rounding
        # (3.6e-07 max abs diff, 5x5 case, E-084 preflight).
        is_causal = self.causal
        if self.causal and attn_mask is not None and attn_mask.dtype == torch.bool:
            tq, tk = q.shape[-2], T
            tri = torch.ones(tq, tk, device=q.device, dtype=torch.bool).tril(
                diagonal=tk - tq)
            attn_mask = attn_mask & tri
            is_causal = False
        if self.head_dim % 8 == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                                 attn_mask=attn_mask)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))
            if is_causal:
                tri = torch.ones(tq if False else q.shape[-2], T,
                                 device=q.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~tri, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            p = F.softmax(scores.float() * (self.head_dim ** -0.5), dim=-1).to(v.dtype)
            out = torch.matmul(p, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class SwiGLU(torch.nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.gate = torch.nn.Linear(hidden, ffn, bias=False)
        self.up = torch.nn.Linear(hidden, ffn, bias=False)
        self.down = torch.nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(torch.nn.Module):
    """Pre-norm Llama block WITHOUT causal masking (bidirectional).

    kv_mode: 'new' (own KV, default / byte-identical pre-extension),
    'reuse' (shared past KV read) or 'edit' (read + learned editor).
    """

    def __init__(self, hidden, n_heads, n_kv, ffn, dropout=0.0, causal=False,
                 kv_mode="new", uses_edit=False):
        super().__init__()
        self.n1 = RMSNorm(hidden)
        self.attn = GQAAttention(hidden, n_heads, n_kv, causal, kv_mode,
                                 uses_edit=uses_edit)
        self.n2 = RMSNorm(hidden)
        self.mlp = SwiGLU(hidden, ffn)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, cache=None, kv_mode=None, attn_mask=None):
        x = x + self.dropout(self.attn(self.n1(x), cache, kv_mode, attn_mask))
        x = x + self.dropout(self.mlp(self.n2(x)))
        return x


class DiacritizerModel(torch.nn.Module):
    """Bidirectional encoder -> per-Arabic-position 15-class logits.

    Labels tensor uses -1 for non-target positions (passthrough/padding);
    those positions NEVER receive a prediction and never enter the loss.

    kv_modes: per-layer list of 'new'/'reuse'/'edit'; a per-forward shared
    KV cache carries every 'new' layer's K/V (pre-rope) to downstream
    reuse/edit consumers (the cache dies with the forward - no cross-step
    state, so training semantics are unchanged).
    """

    def __init__(self, vocab_size, hidden=192, layers=4, n_heads=6, n_kv=2,
                 ffn=None, max_ctx=512, dropout=0.0, n_classes=15, causal=False,
                 kv_modes=None, shared_layers=0):
        super().__init__()
        ffn = ffn or 4 * hidden
        if kv_modes is None:
            kv_modes = ["new"] * layers
        assert len(kv_modes) == layers and all(mm in KV_MODES for mm in kv_modes)
        self.kv_modes = list(kv_modes)
        # E-63 shared-layer lease: n_mod DISTINCT blocks cycled over "layers"
        # USES (module i serves uses i, i+n_mod, ...). Per-use kv_mode (the
        # kv_modes[i] of each use) is applied at call time; a module registers
        # the FIRST use's mode and gets an editor if ANY of its uses is 'edit'.
        # 0/default = one block per use (byte-identical to the pre-E-63 arch).
        self.use_modes = list(kv_modes)
        self.shared_layers = int(shared_layers or 0)
        n_mod = layers if self.shared_layers <= 0 else min(self.shared_layers, layers)
        assert n_mod <= layers
        mod_uses = [[] for _ in range(n_mod)]
        for u in range(len(self.kv_modes)):
            mod_uses[u % n_mod].append(u)
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        mods_list = []
        for uses in mod_uses:
            mods_list.append(Block(hidden, n_heads, n_kv, ffn, dropout, causal=causal,
                                   kv_mode=self.kv_modes[uses[0]],
                                   uses_edit=any(self.kv_modes[uu] == "edit" for uu in uses)))
        self.layers = torch.nn.ModuleList(mods_list)
        self._mod_uses = mod_uses
        self.loss_label_smooth = 0.0
        self.loss_logit_adj = None
        self.final = RMSNorm(hidden)
        self.head = torch.nn.Linear(hidden, n_classes, bias=False)
        self.max_ctx = max_ctx
        self.loss_label_smooth = 0.0
        self.loss_logit_adj = None

    def hidden(self, input_ids, attn_mask=None):
        x = self.embed(input_ids)
        cache = []                                  # shared KV store (this forward)
        n_mod = len(self.layers)
        for u in range(len(self._mod_uses)):
            x = self.layers[u % n_mod](x, cache, kv_mode=self.kv_modes[u], attn_mask=attn_mask)
        return self.final(x)

    def forward(self, input_ids, labels=None, attn_mask=None):
        h = self.hidden(input_ids, attn_mask)
        logits = self.head(h)                       # [B, T, n_classes]
        out = {"logits": logits}
        if labels is not None:
            mask = labels >= 0
            sel = logits[mask].float()
            adj = getattr(self, "loss_logit_adj", None)
            if adj is not None:
                sel = sel + adj          # logit adjustment also rebalances the argmax intent
            loss = F.cross_entropy(sel, labels[mask].long(),
                                   label_smoothing=getattr(self, "loss_label_smooth", 0.0))
            out["loss"] = loss
            out["acc"] = (sel.argmax(-1) == labels[mask].long()).float().mean()
        return out


def _resolve_kv_modes(m, layers):
    """model.kv_pattern -> per-layer kv_mode list. Default/None/'new' = all-new
    (byte-identical pre-extension behaviour)."""
    p = m.get("kv_pattern", "new")
    if p in (None, "new", "", "none"):
        return ["new"] * layers
    if p == "cycle3":
        return [KV_MODES[i % 3] for i in range(layers)]      # new, reuse, edit, ...
    if p in ("reuse_all", "edit_all"):
        mode = p[:-4]
        return ["new"] + [mode] * (layers - 1)               # layer 0 must write
    if isinstance(p, (list, tuple)):
        assert len(p) == layers, f"kv_pattern length {len(p)} != layers {layers}"
        modes = [str(mm).strip().lower() for mm in p]
        assert all(mm in KV_MODES for mm in modes), modes
        if modes[0] != "new":
            modes[0] = "new"  # first layer must write the KV a reader can use
            print({"kv_pattern": "first layer forced to 'new' (cache source)"})
        return modes
    raise ValueError(f"bad kv_pattern: {p!r} (use 'new'|'cycle3'|'reuse_all'|"
                     f"'edit_all'|explicit list of {KV_MODES})")


def build_from_config(cfg, vocab_size):
    m = cfg.get("model", cfg)
    layers = int(m.get("layers", 4))
    return DiacritizerModel(
        causal=m.get("causal", False),
        vocab_size=vocab_size,
        hidden=m.get("hidden", 192),
        layers=layers,
        n_heads=m.get("n_heads", 6),
        n_kv=m.get("n_kv", 2),
        ffn=m.get("ffn") or None,
        max_ctx=m.get("ctx", 512),
        dropout=m.get("dropout", 0.0),
        kv_modes=_resolve_kv_modes(m, layers),
        shared_layers=int(m.get("shared_layers", 0)),
    )
