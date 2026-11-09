r"""Model Explorer data model: param math, block graph, two-level info.
Pure logic + data — no Gradio here (model_tab.py owns wiring). Every number
is computed live from artifacts (real headers when a checkpoint exists, the
projected formula otherwise, always flagged). Read-only, CPU-only.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from artifacts import module_param_totals

# Shared param math lives in packages/param-math — the same canonical formulas
# the viz/flow TypeScript package owns. The directory name has a hyphen, so it
# cannot be a Python package name: put it on sys.path and import the module.
# Works from app.py (webui/ on sys.path), the webui tests, and tests/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARAM_MATH_DIR = _REPO_ROOT / "packages" / "param-math"
if str(_PARAM_MATH_DIR) not in sys.path:
    sys.path.insert(0, str(_PARAM_MATH_DIR))

import param_math

logger = logging.getLogger(__name__)

KIND_COLOR = {"tok": "#7c3aed", "embed": "#d97706", "norm": "#2563eb",
              "attn": "#6d28d9", "ffn": "#059669", "out": "#b45309"}


def head_dim(dims: dict) -> int:
    return param_math.head_dim(int(dims["hidden"]), max(int(dims["heads"]), 1))


def projected_params(dims: dict) -> int:
    """Param formula; VERIFIED against the three real finals:
    smoke 12,323,072 / pilot 100,682,496 / target 226,526,208.
    Delegates to the shared packages/param-math module (p_total_dense) —
    the same canonical formulas scripts/status.py uses, so the two stay in
    exact agreement. Tied embeddings add no lm_head; untied add vocab*h."""
    total = param_math.p_total_dense(
        layers=int(dims["layers"]), d=int(dims["hidden"]),
        heads=int(dims["heads"]), kv=int(dims["kv_heads"]),
        ffn=int(dims["ffn"]), vocab=int(dims["vocab"]))
    if not dims["tie"]:
        total += int(dims["vocab"]) * int(dims["hidden"])
    return total


def kv_cache_kib_per_token(dims: dict) -> float:
    """2 (K and V) x layers x kv_heads x head_dim x 2 B (fp16), in KiB.
    Target = 16.0 KiB/token (independently measured on this machine)."""
    b = 2 * int(dims["layers"]) * int(dims["kv_heads"]) * head_dim(dims) * 2
    return b / 1024


def _block(bid, kind, title, layer, params, real, tensors, io, pairs, plain, tech):
    return {"id": bid, "kind": kind, "title": title, "layer": layer,
            "params": int(params), "real": bool(real), "tensors": tensors,
            "io": io, "config_pairs": pairs, "plain": plain, "tech": tech}


def build_graph(dims: dict, header: dict | None, label: str, lora: dict | None):
    """Ordered block list for the selected model. Real per-block params and
    tensor names come from the safetensors header when present; otherwise
    the projected formula (every block flagged real=False)."""
    h = int(dims["hidden"])
    seq = min(int(dims["ctx"]) or 64, 64)
    L = int(dims["layers"])
    hd, heads, kv = head_dim(dims), int(dims["heads"]), int(dims["kv_heads"])
    vocab = int(dims["vocab"])
    totals = module_param_totals(header) if header else {}
    real = bool(header)
    blocks: list[dict] = []

    def fmt(n: int) -> str:
        return f"{n:,}"

    blocks.append(_block(
        "tok", "tok", "Tokenizer (CodeLlama 32k)", None, 0, True, {},
        ("[text]", f"[1, {seq}] ids"),
        [("vocab_size", vocab), ("tokenizer", "codellama/CodeLlama-7b-hf")],
        f"Turns your text into one of {vocab:,} token IDs the model knows. "
        "A sentencepiece vocabulary trained for code — indentation and "
        "underscores are single tokens.",
        f"sentencepiece BPE, vocab_size {vocab:,}. Not a learned module: no "
        "params. The Trace box runs the REAL tokenizer files from the "
        "checkpoint."))

    emb_params = totals.get("embed", vocab * h)
    emb_dtype = next(iter((header or {}).values()), {"dtype": "fp32"})["dtype"]
    blocks.append(_block(
        "embed", "embed", "Token embedding (lookup table)", None, emb_params,
        real, {k: v for k, v in (header or {}).items() if "embed_tokens" in k},
        (f"[1, {seq}]", f"[1, {seq}, {h}]"),
        [("vocab", vocab), ("hidden", h), ("tied_lm_head", dims["tie"])],
        f"A lookup table: token ID -> a row of {h} learned numbers. "
        + ("TIED with the output layer: the same table reads tokens in and "
           "scores tokens out — halves the biggest cost in a small model. "
           if dims["tie"] else "Separate from the output layer here. "),
        f"params {fmt(emb_params)} = vocab {vocab:,} x hidden {h}, dtype "
        f"{emb_dtype}. Row i of the table is what token i becomes."))

    for i in range(L):
        pre = totals.get(f"L{i}.input_layernorm", h)
        blocks.append(_block(
            f"L{i}.pre_attn_norm", "norm",
            f"Layer {i} - RMSNorm (pre-attention)", i, pre, real, {},
            (f"[1, {seq}, {h}]",) * 2,
            [("rms_eps", dims["rms_eps"]), ("hidden", h)],
            "Rescales each token's vector to a stable size before attention. "
            "The cheaper cousin of LayerNorm — no mean subtraction, just RMS. "
            "Keeps deep stacks trainable.",
            f"weight [{h}] ({fmt(pre)} params), eps {dims['rms_eps']}. "
            "y = x / sqrt(mean(x^2) + eps) * weight."))
        attn_params = totals.get(f"L{i}.self_attn", 0) or (
            heads * hd * h + 2 * kv * hd * h + heads * hd * h)
        attn_tensors = ({k: v for k, v in (header or {}).items()
                         if k.startswith(f"model.layers.{i}.self_attn")}
                        if header else {})
        blocks.append(_block(
            f"L{i}.attn", "attn",
            f"Layer {i} - GQA attention (RoPE, causal, +residual)", i,
            attn_params, real, attn_tensors,
            (f"[1, {seq}, {h}]", f"[1, {seq}, {h}]"),
            [("heads", heads), ("kv_heads", kv), ("head_dim", hd),
             ("rope_theta", dims["rope_theta"]), ("seq", seq)],
            f"Every token looks back at earlier tokens and decides what to "
            f"copy. {heads} query heads share {kv} key/value heads (GQA) — the "
            f"K/V side is cut {heads // max(kv, 1)}x, saving memory at almost "
            "no quality cost. Causal: a token never sees the future. RoPE "
            "rotates vectors by position so order is baked in without a "
            "position table.",
            f"q_proj [{h}x{heads * hd}] ({fmt(heads * hd * h)}), k/v_proj "
            f"[{h}x{kv * hd}] ({fmt(kv * hd * h)} each), o_proj "
            f"[{heads * hd}x{h}]. rope_theta {dims['rope_theta']}. KV-cache "
            f"{kv_cache_kib_per_token(dims):.1f} KiB/token for THIS model "
            "(2 x layers x kv_heads x head_dim x 2 B). SDPA backend "
            "(sm_75: no flash-attn)."))
        post = totals.get(f"L{i}.post_attention_layernorm", h)
        blocks.append(_block(
            f"L{i}.post_attn_norm", "norm", f"Layer {i} - RMSNorm (pre-FFN)",
            i, post, real, {}, (f"[1, {seq}, {h}]",) * 2,
            [("rms_eps", dims["rms_eps"])],
            "Same stabilizer again, before the FFN. Two norms per block is "
            "the pre-norm layout — it is what makes residual streams "
            "trainable at depth.",
            f"weight [{h}] ({fmt(post)} params), eps {dims['rms_eps']}."))
        ffn_params = totals.get(f"L{i}.mlp", 0) or (3 * h * int(dims["ffn"]))
        ffn_tensors = ({k: v for k, v in (header or {}).items()
                        if k.startswith(f"model.layers.{i}.mlp")}
                       if header else {})
        blocks.append(_block(
            f"L{i}.ffn", "ffn", f"Layer {i} - SwiGLU FFN", i, ffn_params,
            real, ffn_tensors,
            (f"[1, {seq}, {h}]", f"[1, {seq}, {h}]"),
            [("ffn", dims["ffn"]), ("hidden", h)],
            f"The 'thinking' MLP: expands each token to {dims['ffn']} numbers, "
            f"filters through a gate, back to {h}. Most of the model's "
            "parameters live here, and this is where token knowledge is "
            "stored.",
            f"gate/up [{h}x{dims['ffn']}] + down [{dims['ffn']}x{h}] = "
            f"3*h*ffn = {fmt(3 * h * int(dims['ffn']))}. SwiGLU: "
            "down( SiLU(gate(x)) * up(x) )."))

    blocks.append(_block(
        "final_norm", "norm", "Final RMSNorm", None,
        totals.get("model.norm", h), real, {},
        (f"[1, {seq}, {h}]",) * 2, [("rms_eps", dims["rms_eps"])],
        "One last rescale before scoring tokens. Same job as the layer norms.",
        f"weight [{h}] ({fmt(totals.get('model.norm', h))} params)."))

    lm_params = totals.get("lm_head", 0) or (0 if dims["tie"] else vocab * h)
    lm_tensors = ({k: v for k, v in (header or {}).items()
                   if k.startswith("lm_head")} if header else {})
    blocks.append(_block(
        "lm_head", "out", "lm_head (output projection)", None, lm_params,
        real and bool(lm_params), lm_tensors,
        (f"[1, {seq}, {h}]", f"[1, {seq}, {vocab}]"),
        [("vocab", vocab), ("tied", dims["tie"])],
        ("TIED with the embedding table — zero extra parameters; the same "
         "lookup table is transposed to score every token."
         if dims["tie"] and not lm_params else
         f"Projects each token to {vocab:,} raw scores (logits)."),
        f"logits = h @ W_out ({fmt(lm_params) if lm_params else 'tied: 0'} "
        f"params; output [1, {seq}, {vocab:,}]). Softmax comes later, at "
        "sampling time."))

    blocks.append(_block(
        "logits", "out", "Logits -> next token", None, 0, True, {},
        (f"[1, {seq}, {vocab}]", "[1, vocab] probs"),
        [("vocab", vocab)],
        "The final vector is compared against every row of the vocabulary: "
        "softmax turns scores into probabilities. The Chat tab's temperature "
        "/ top-p / top-k knobs all act on this distribution.",
        "argmax = greedy; temperature / top-p / top-k = sampling. Trained "
        "against cross-entropy on the true next token."))

    if lora:
        for b in blocks:
            if b["kind"] in ("attn", "ffn"):
                b["plain"] += (f" LoRA r={lora['r']} adapters are trained on "
                               "this block's projections; base weights frozen.")
    return blocks


def shape_walk(blocks):
    """[(id, in_shape, out_shape)] in graph order — the prompt-trace walk."""
    return [(b["id"], b["io"][0], b["io"][1]) for b in blocks]


def tokenize_trace(run_path: Path | None, text: str) -> str:
    """REAL CodeLlama tokenizer from the checkpoint dir (local files only,
    never the network). Markdown: token count + id list. No weights loaded."""
    if run_path is None:
        return ("*Trace needs a trained run (it loads the tokenizer files "
                "from the checkpoint directory). Pick 'From trained run'.*")
    if not text.strip():
        return "*Type some text to trace.*"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(run_path), local_files_only=True)
    logger.info("tokenizer loaded for prompt trace: %s", run_path)
    ids = tok(text, add_special_tokens=False).input_ids
    toks = tok.convert_ids_to_tokens(ids)
    shown = list(zip(toks, ids))[:24]
    lines = [f"**{len(ids)} tokens** (showing first {len(shown)}):", ""]
    lines += [f"`{t!r}` -> id {i}" for t, i in shown]
    if len(ids) > len(shown):
        lines.append(f"... +{len(ids) - len(shown)} more")
    lines += ["", "These IDs index rows of the embedding table — that is "
              "the model's actual input."]
    return "\n".join(lines)


def render_svg(blocks, selected_id: str = "") -> str:
    """Self-contained inline SVG stack (no external assets; clicking is
    wired by model_tab.py's shim — data-bid carries the block id). The
    layer owning the selected block is expanded; others collapse."""
    sel_layer = next((b["layer"] for b in blocks if b["id"] == selected_id), None)
    expand = sel_layer if sel_layer is not None else 0
    rows: list[tuple[str, str, str, bool]] = []  # (bid, label, color, is_sel)
    for b in blocks:
        if b["layer"] is None:
            rows.append((b["id"], b["title"], KIND_COLOR[b["kind"]],
                         b["id"] == selected_id))
        elif b["id"].endswith("pre_attn_norm"):
            rows.append((f"L{b['layer']}",
                         f"Layer {b['layer']}: norm - attn - norm - ffn (click to expand)",
                         "#475569", False))
        if b["layer"] == expand:
            rows.append((b["id"], b["title"].split(" - ", 1)[1],
                         KIND_COLOR[b["kind"]], b["id"] == selected_id))
    W, RH, GAP = 330, 30, 6
    H = len(rows) * (RH + GAP) + 8
    parts = ['<svg viewBox="0 0 %d %d" style="width:100%%;max-width:350px;'
             'font-family:ui-sans-serif,system-ui,sans-serif">' % (W, H)]
    y = 4
    for bid, label, color, sel in rows:
        disp = label if len(label) <= 44 else label[:43] + "..."
        parts.append(
            f'<g class="blk" data-bid="{bid}" style="cursor:pointer" '
            f'onclick="window.__dshtModelSelect && '
            f"window.__dshtModelSelect('{bid}')\">"
            f'<rect x="4" y="{y}" width="{W - 8}" height="{RH}" rx="6" '
            f'fill="{color}" fill-opacity="{"1" if sel else "0.85"}" '
            f'stroke="#0f172a" stroke-width="{"2" if sel else "1"}"/>'
            f'<text x="14" y="{y + 20}" font-size="12.5" fill="white">{disp}</text>'
            f"</g>")
        y += RH + GAP
    parts.append("</svg>")
    return "".join(parts)
