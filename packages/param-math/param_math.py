"""Shared param math — the ONE canonical implementation for webui/ and
scripts/ (Python) and for viz/ and flow/ (TypeScript: src/index.ts).
Computed from spec shapes, NEVER hand-written numbers in UI text.

Dense anchors (unit-tested): nano_SLMs 226,526,208 · SmolLM2-135M 134,515,008.
Norm accounting: pre-norm x2 per layer + 1 final norm = (2L+1)*d — counted ONCE.

Dependency-free (stdlib only) so both import contexts work: the Gradio app
(webui/, imported with webui/ on sys.path) and the repo-root scripts
(scripts/). The directory name has a hyphen, so it cannot be a Python
package name — consumers put this directory on sys.path and `import param_math`.
"""
from __future__ import annotations


def head_dim(d: int, heads: int) -> int:
    return d // heads


def kv_dim(d: int, heads: int, kv: int) -> int:
    return kv * head_dim(d, heads)


def p_embed(d: int, vocab: int) -> int:
    return vocab * d


def p_attention_dense(d: int, heads: int, kv: int) -> int:
    kvd = kv_dim(d, heads, kv)
    return 2 * d * d + 2 * d * kvd  # WQ,WO (square) + WK,WV (d x kvdim)


def p_ffn_dense(d: int, ffn: int) -> int:
    return 3 * d * ffn  # gate + up + down (SwiGLU, no bias)


def p_norms(d: int, layers: int) -> int:
    return (2 * layers + 1) * d


def p_layer_dense(d: int, heads: int, kv: int, ffn: int) -> int:
    return p_attention_dense(d, heads, kv) + p_ffn_dense(d, ffn) + 2 * d


def p_total_dense(layers: int, d: int, heads: int, kv: int, ffn: int, vocab: int) -> int:
    """Dense total — norms counted exactly once (NOT inside the per-layer loop)."""
    return (
        p_embed(d, vocab)
        + layers * (p_attention_dense(d, heads, kv) + p_ffn_dense(d, ffn))
        + p_norms(d, layers)
    )


# ---- hybrid (linear-attention + full-attention + sparse MoE) ----
# `hybrid` mirrors the camelCase keys of the viz model YAMLs' `hybrid` block.

def layer_kind_hybrid(layer: int, hybrid: dict) -> str:
    full_every = hybrid.get("fullEvery") or 4
    return "full" if layer % full_every == 0 else "linear"


def p_mixer_linear(d: int, linear_mixer: dict) -> int:
    in_proj = d * (linear_mixer["kHeads"] + linear_mixer["qHeads"] + linear_mixer["vHeads"]) * linear_mixer["headDim"]
    out_proj = linear_mixer["vHeads"] * linear_mixer["headDim"] * d
    conv = linear_mixer["conv"] * (linear_mixer["kHeads"] + linear_mixer["vHeads"]) * linear_mixer["headDim"]
    return in_proj + out_proj + conv


def p_mixer_full(d: int, heads: int, kv: int, head_dim_full: int) -> int:
    q = heads * head_dim_full
    kvd = kv * head_dim_full
    return d * q + 2 * d * kvd + q * d


def p_router(d: int, experts_total: int) -> int:
    return d * experts_total


def p_moe_layer(d: int, total: int, shared: int, inter: int) -> int:
    per_expert = 3 * d * inter
    return total * per_expert + shared * per_expert + p_router(d, total)


def n_full_layers(layers: int, hybrid: dict) -> int:
    return layers // (hybrid.get("fullEvery") or 4)


def n_linear_layers(layers: int, hybrid: dict) -> int:
    return layers - n_full_layers(layers, hybrid)


def p_total_hybrid(layers: int, d: int, heads: int, kv: int, ffn: int, vocab: int,
                   hybrid: dict | None = None) -> int:
    h = hybrid or {}
    experts = h.get("experts") or {"total": 0, "shared": 0, "active": 0, "inter": ffn}
    full_mixer = h.get("fullMixer") or {}
    head_dim_full = full_mixer.get("headDim") or round(d / heads)
    lin_per = p_mixer_linear(d, h["linearMixer"]) if h.get("linearMixer") else 0
    full_per = p_mixer_full(d, heads, kv, head_dim_full)
    moe_per = p_moe_layer(d, experts["total"], experts["shared"], experts["inter"])
    extras = (h.get("extras") or {}).get("params", 0)
    return (
        2 * p_embed(d, vocab)  # untied: input embedding + separate output head
        + n_full_layers(layers, h) * full_per
        + n_linear_layers(layers, h) * lin_per
        + layers * moe_per
        + extras
    )


def is_hybrid(arch_kind: str | None) -> bool:
    return arch_kind == "hybridLinearMoe"


def fmt_p(n: int) -> str:
    return f"{n:,}"


def fmt_b(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return fmt_p(n)


def p_total(layers: int, d: int, heads: int, kv: int, ffn: int, vocab: int,
            arch_kind: str | None = None, hybrid: dict | None = None) -> int:
    if is_hybrid(arch_kind):
        return p_total_hybrid(layers, d, heads, kv, ffn, vocab, hybrid)
    return p_total_dense(layers, d, heads, kv, ffn, vocab)
