r"""Read-only artifact readers for the Model tab (WEBUI_PRD.md §5 U12/U13).

Hard rules (PRD §1): the UI wraps, never reimplements; this module is
read-only + CPU-only — it opens small JSON files and safetensors HEADERS
(tensor names, shapes, dtypes) and never reads tensor data, never loads
weights, never allocates GPU memory, never writes. (Reading a header does
import the torch module as a side effect of the safetensors PT backend —
no tensor data and no GPU memory are touched.) Safe beside a live run.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    """Read a small JSON file; None when missing/corrupt (never raises)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def hf_config_dims(path: Path) -> dict | None:
    """Dims dict from a checkpoint's config.json (HF LlamaConfig layout)."""
    c = load_json(path / "config.json")
    if not c:
        return None
    return {
        "layers": int(c.get("num_hidden_layers", 0)),
        "hidden": int(c.get("hidden_size", 0)),
        "heads": int(c.get("num_attention_heads", 0)),
        "kv_heads": int(c.get("num_key_value_heads", 0)),
        "ffn": int(c.get("intermediate_size", 0)),
        "ctx": int(c.get("max_position_embeddings", 0)),
        "vocab": int(c.get("vocab_size", 0)),
        "tie": bool(c.get("tie_word_embeddings", False)),
        "rms_eps": float(c.get("rms_norm_eps", 1e-5)),
        "rope_theta": float(c.get("rope_theta", 10000.0)),
    }


def yaml_config_dims(cfg: dict) -> dict:
    """Dims dict from a repo configs/*.yaml model block (status.py layout).

    rms_eps / rope_theta mirror src/model.py build_config's constants.
    """
    m = cfg.get("model", {})
    return {
        "layers": int(m.get("layers", 0)),
        "hidden": int(m.get("hidden", 0)),
        "heads": int(m.get("heads", 0)),
        "kv_heads": int(m.get("kv_heads", 0)),
        "ffn": int(m.get("ffn", 0)),
        "ctx": int(m.get("ctx", 0)),
        "vocab": int(cfg.get("tokenizer", {}).get("vocab_size", 0)),
        "tie": bool(m.get("tie_embeddings", True)),
        "rms_eps": 1e-5,
        "rope_theta": 10000.0,
    }


def safetensors_header(path: Path) -> dict | None:
    """name -> {"shape": [...], "dtype": str} for every *.safetensors under
    path. HEADERS ONLY — the mmap reads ~KBs of JSON per file, never tensor
    data. None when the dir has no readable safetensors at all; otherwise a
    dict (possibly from a partial scan), so LoRA adapter files still count.
    """
    if not path.is_dir():
        return None
    header: dict = {}
    for f in sorted(path.glob("*.safetensors")):
        try:
            from safetensors import safe_open
            with safe_open(str(f), framework="pt") as fh:
                for name in fh.keys():
                    sl = fh.get_slice(name)
                    header[name] = {"shape": list(sl.get_shape()),
                                    "dtype": str(sl.get_dtype())}
        except Exception:  # noqa: BLE001 - a corrupt file must not kill the tab
            continue
    return header or None


def tensor_params(header: dict) -> dict[str, int]:
    """name -> element count."""
    out = {}
    for name, meta in header.items():
        n = 1
        for d in meta["shape"]:
            n *= int(d)
        out[name] = n
    return out


def module_param_totals(header: dict) -> dict[str, int]:
    """Aggregate tensor params by dotted module prefix.

    LlamaForCausalLM names look like:
      model.embed_tokens.weight                     -> "embed"
      model.layers.<i>.self_attn.*                  -> "L<i>.self_attn"
      model.layers.<i>.mlp.*                        -> "L<i>.mlp"
      model.layers.<i>.input_layernorm.weight       -> "L<i>.input_layernorm"
      model.layers.<i>.post_attention_layernorm.*   -> "L<i>.post_attention_layernorm"
      model.norm.weight                             -> "model.norm"
      lm_head.weight (untied only)                  -> "lm_head"
    """
    out: dict[str, int] = {}
    for name, n in tensor_params(header).items():
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            key = f"L{parts[2]}.{parts[3]}"
        elif "embed_tokens" in parts:
            key = "embed"
        elif name.startswith("lm_head"):
            key = "lm_head"
        elif len(parts) >= 3 and parts[0] == "model":
            # model.norm.weight -> "model.norm" (drop trailing param name)
            key = name.removesuffix("." + parts[-1])
        else:
            key = name
        out[key] = out.get(key, 0) + n
    return out


def lora_note(path: Path) -> dict | None:
    """adapter_config.json summary for LoRA runs (None otherwise)."""
    c = load_json(path / "adapter_config.json")
    if not c:
        return None
    return {"r": c.get("r"), "alpha": c.get("lora_alpha"),
            "targets": c.get("target_modules", [])}


def phase_config(phase: str):
    """configs/<phase>.yaml via the repo's own loader (scripts/status.py)."""
    import status  # scripts/ is on sys.path (app.py and tests/_common set it)
    return status.load_phase_cfg(phase)


def run_config(phase: str):
    """Phase config with a dash/underscore fallback (the run is kd-t2p-kd but
    the config file is kd_t2p_kd.yaml)."""
    return phase_config(phase) or phase_config(phase.replace("-", "_"))
