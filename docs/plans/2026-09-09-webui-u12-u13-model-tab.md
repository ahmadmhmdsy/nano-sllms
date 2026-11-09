# Web UI U12+U13 — Model tab (Architecture Explorer + real-run Training Simulator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add one new "Model" tab to `webui/app.py`: an interactive Architecture Explorer (real tensor shapes from safetensors **headers**, two-level explanations, prompt trace) and a Training Simulator that REPLAYS real runs (Pretrain/SFT/LoRA/KD) with animated stages, scrubbing and checkpoint rotation — read-only, CPU-only, zero GPU.

**Architecture:** Four new focused modules under `webui/` (`artifacts.py` = read-only file readers, `explorer.py` = graph/math/info/SVG, `simulator.py` = replay logic, `model_tab.py` = thin Gradio wiring). `app.py` injects its existing `_ckpts`/`_full_curve` helpers (no duplication, no circular import) and grows by ~4 lines. Spec: WEBUI_PRD.md §5 U12/U13 (commit a61ee19).

**Tech Stack:** Gradio 6.26 (already in app), safetensors headers via `safe_open` (installed), matplotlib (already used), transformers tokenizer (local files only), stdlib. **No new dependencies.** Tests are dependency-free assert scripts (venv has NO pytest) — repo convention, canned-self-test style.

---

## Repo facts the executor MUST know

- **Python:** ALWAYS `& .\.venv\Scripts\python.exe` — never bare `python`, never pip (CLAUDE.md §4).
- **Windows/PowerShell.** Working dir = repo root `E:\python_projects\nano_SLMs`.
- **No pytest.** Tests are plain-assert scripts under `tests\`; run directly; exit code 0/1 (repo canned-self-test style).
- **Read-only + CPU-only everywhere** (PRD §1/§5): never load model weights, never call torch CUDA, never write under `runs/` or `configs/`. tensorboards/transformers imports are fine (app already uses them).
- **Never co-run GPU jobs.** These tasks are all CPU file-reads, safe beside a live training run — but do NOT launch any GPU script.
- **Git:** commit after each task with the exact messages given. Do NOT stage unrelated dirty files (pre-existing WIP: GDN/Track work, `scripts/eval.py`, `src/model.py` etc. — leave them alone). `git add <exact paths>` only.
- **Known-good numbers to cross-check:** smoke 12.3M params (4L/256h/4H/2KV/ffn1024/ctx256/vocab32768), pilot **100.7M** (12L/768h/12H/4KV/2048/512), target **226.5M** (16L/1024h/16H/4KV/3072/512-1024). KV-cache = 16 KiB/token for target. All 11 replayable runs verified on 2026-09-09 to have `runs/<name>/logs` tfevents + `final/train_summary.json`.
- **Never create anything under `runs/`.** Tests only READ.

## File structure (locked)

- **Create:** `tests/_common.py` — sys.path setup + PASS/FAIL runner for all test scripts.
- **Create:** `webui/artifacts.py` — read-only readers: JSON loads, HF-config dims, **safetensors header parser** (names/shapes/dtypes only), param aggregation, LoRA note. No Gradio, no torch.
- **Create:** `webui/explorer.py` — dims math (projected params — validated against all 3 real models), block graph builder, two-level info text, SVG renderer, real-tokenizer prompt trace. No Gradio.
- **Create:** `webui/simulator.py` — technique map, RunData loader, stage chains, frame_at (pure), rotation events, VRAM estimates anchored to measured probes, KD delta, HTML renderers. No Gradio.
- **Create:** `webui/model_tab.py` — thin Gradio wiring for the nested tab (Explorer + Simulator), SVG click shim, module-global sim state (single-user app, same pattern as the chat model service `_MODEL`).
- **Modify:** `webui/app.py` — +1 import line, +1 render call (exact anchors in Task 4).
- **Create:** `tests/test_artifacts.py`, `tests/test_explorer.py`, `tests/test_simulator.py`.

---

### Task 0: Preflight (read-only environment verification)

**Files:** none created.

- [ ] **Step 1: Verify venv libs + key run dirs exist**

Run:
```
& .\.venv\Scripts\python.exe -c "import importlib.util as u; print('safetensors', bool(u.find_spec('safetensors'))); print('matplotlib', bool(u.find_spec('matplotlib'))); print('transformers', bool(u.find_spec('transformers'))); print('tensorboard', bool(u.find_spec('tensorboard')))"
```
Expected: all four True.

Run:
```
& .\.venv\Scripts\python.exe -c "from pathlib import Path; rs=['smoke','pilot','target','sft_t1','sft_v2_e1','h2_copy_lora','h2p2_mixed_lora','kd-t2p-kd','kd-s-t1']; print([(r,(Path('runs')/r/'final').is_dir(), any((Path('runs')/r/'logs').glob('*tfevents*'))) for r in rs])"
```
Expected: every tuple = (True, True). If a run dir is missing (machine cleanup), remove that name from `simulator.py` TECHNIQUES in Task 5 and record it in the final report (honest reporting rule).

- [ ] **Step 2: Confirm git state**

Run: `git status --porcelain`
Expected: the pre-existing WIP entries only (GDN/Track work). Branch must be `main`.

---

### Task 1: `tests/_common.py` + `webui/artifacts.py` (TDD)

**Files:**
- Create: `tests/_common.py`
- Create: `tests/test_artifacts.py`
- Create: `webui/artifacts.py`

- [ ] **Step 1: Write the test helper and the failing test**

Create `tests/_common.py` exactly:

```python
"""Path setup + PASS/FAIL runner for the repo's dependency-free test scripts.

No pytest in the venv (repo convention): each tests/test_*.py is run directly
with the venv python, uses plain asserts, prints PASS/FAIL per check and
exits 0/1. Tests add repo root, scripts/ and webui/ to sys.path so they can
import the same modules webui/app.py exposes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "webui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        return False
    except Exception as e:  # noqa: BLE001 - report, don't crash the suite
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False


def run(tests: dict):
    results = [check(n, f) for n, f in tests.items()]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(tests)} checks passed")
    sys.exit(0 if n_pass == len(tests) else 1)
```

Create `tests/test_artifacts.py` exactly:

```python
"""tests for webui/artifacts.py — real files under runs/, headers only."""
import _common  # noqa: F401  (path setup)

import artifacts as A
from _common import run

ROOT = A.ROOT
SMOKE = ROOT / "runs" / "smoke" / "final"
PILOT = ROOT / "runs" / "pilot" / "final"
TARGET = ROOT / "runs" / "target" / "final"
LORA = ROOT / "runs" / "h2_copy_lora" / "final"


def t_header_smoke():
    h = A.safetensors_header(SMOKE)
    assert h, "no safetensors under runs/smoke/final"
    emb = h.get("model.embed_tokens.weight")
    assert emb and emb["shape"] == [32768, 256], f"embed shape {emb}"
    assert emb["dtype"] in ("F16", "F32", "BF16"), emb["dtype"]  # safetensors dtype strings


def t_totals_match_ladder():
    gates = [("smoke", SMOKE, 12_300_000, 700_000),
             ("pilot", PILOT, 100_700_000, 1_200_000),
             ("target", TARGET, 226_500_000, 1_500_000)]
    for name, path, want, tol in gates:
        h = A.safetensors_header(path)
        assert h, f"{name}: no header"
        total = sum(A.tensor_params(h).values())
        assert abs(total - want) <= tol, f"{name}: {total} vs {want} (+/-{tol})"


def t_module_aggregation():
    h = A.safetensors_header(SMOKE)
    mods = A.module_param_totals(h)
    assert mods.get("embed") == 32768 * 256, mods.get("embed")
    assert mods.get("L0.self_attn"), "attn module missing"
    assert mods.get("L3.mlp"), "mlp module missing"
    assert mods.get("model.norm"), "final norm missing (key must be model.norm)"


def t_hf_dims_and_yaml_dims_agree():
    hf = A.hf_config_dims(SMOKE)
    assert hf and hf["layers"] == 4 and hf["hidden"] == 256 and hf["kv_heads"] == 2, hf
    y = A.yaml_config_dims(A.phase_config("smoke"))
    for k in ("layers", "hidden", "heads", "kv_heads", "ffn", "ctx", "vocab", "tie"):
        assert hf[k] == y[k], (k, hf[k], y[k])


def t_lora_note():
    note = A.lora_note(LORA)
    if note is None:
        assert not (LORA / "adapter_config.json").is_file()
        return
    assert note["r"] and note["targets"], note


def t_missing_dir_is_none():
    assert A.safetensors_header(ROOT / "runs" / "definitely-not-here") is None
    assert A.load_json(ROOT / "runs" / "definitely-not-here" / "x.json") is None


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe tests\test_artifacts.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'artifacts'`.

- [ ] **Step 3: Write `webui/artifacts.py`**

Create `webui/artifacts.py` exactly:

```python
r"""Read-only artifact readers for the Model tab (WEBUI_PRD.md §5 U12/U13).

Hard rules (PRD §1): the UI wraps, never reimplements; this module is
read-only + CPU-only — it opens small JSON files and safetensors HEADERS
(tensor names, shapes, dtypes) and never reads tensor data, never imports
torch, never allocates GPU memory, never writes. Safe beside a live run.
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
```

- [ ] **Step 4: Run tests to green**

Run: `& .\.venv\Scripts\python.exe tests\test_artifacts.py`
Expected: `6/6 checks passed`, exit 0.

- [ ] **Step 5: Commit**

```powershell
git add tests/_common.py tests/test_artifacts.py webui/artifacts.py
git commit -m "feat(webui): U12 artifacts - read-only safetensors-header/config readers + dependency-free test runner"
```

---

### Task 2: `webui/explorer.py` — dims math, block graph, two-level info (TDD)

**Files:**
- Create: `webui/explorer.py`
- Create: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_explorer.py` exactly:

```python
"""tests for webui/explorer.py — graph math validated against real models."""
import _common  # noqa: F401

import artifacts as A
import explorer as E
from _common import run

ROOT = A.ROOT
SMOKE = ROOT / "runs" / "smoke" / "final"
TARGET = ROOT / "runs" / "target" / "final"


def _smoke_dims_header():
    return A.hf_config_dims(SMOKE), A.safetensors_header(SMOKE)


def t_projected_formula_matches_all_three():
    gates = [("smoke", 12_323_072), ("pilot", 100_682_496), ("target", 226_526_208)]
    for phase, want in gates:
        dims = A.yaml_config_dims(A.phase_config(phase))
        got = E.projected_params(dims)
        assert got == want, (phase, got, want)


def t_build_graph_smoke_real():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    # tok + embed + 4 layers x 4 blocks + final norm + lm_head + logits
    assert len(blocks) == 2 + 4 * dims["layers"] + 3, len(blocks)
    assert blocks[0]["id"] == "tok" and blocks[-1]["id"] == "logits"
    embed = blocks[1]
    assert embed["params"] == 32768 * 256 and embed["real"] is True
    attn = next(b for b in blocks if b["kind"] == "attn")
    assert attn["params"] > 0 and attn["tensors"], "real attn tensors missing"
    assert all(b["plain"] and b["tech"] for b in blocks), "info text missing"


def t_build_graph_projected_flagged():
    dims = A.yaml_config_dims(A.phase_config("smoke"))
    blocks = E.build_graph(dims, None, label="smoke (config)", lora=None)
    # tok/logits need no header (real=True); tied lm_head + logits are 0-param
    assert [b["id"] for b in blocks if b["real"]] == ["tok", "logits"]
    assert [b["id"] for b in blocks
            if b["params"] == 0] == ["tok", "lm_head", "logits"]
    assert all(b["params"] > 0 for b in blocks
               if b["id"] not in ("tok", "lm_head", "logits"))


def t_kv_cache_token_target_16kib():
    dims = A.hf_config_dims(TARGET)
    kib = E.kv_cache_kib_per_token(dims)
    assert abs(kib - 16.0) < 0.1, kib  # measured on this machine


def t_shape_trace():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    walk = E.shape_walk(blocks)
    assert walk[0][1] == "[text]" and walk[-1][1] == "[1, 64, 32768]"
    mid = next(w for w in walk if w[0].endswith("attn"))
    assert mid[1] == f"[1, 64, {dims['hidden']}]", mid


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe tests\test_explorer.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'explorer'`.

- [ ] **Step 3: Write `webui/explorer.py`**

Create `webui/explorer.py` exactly:

```python
r"""Model Explorer data model: param math, block graph, two-level info.
Pure logic + data — no Gradio here (model_tab.py owns wiring). Every number
is computed live from artifacts (real headers when a checkpoint exists, the
projected formula otherwise, always flagged). Read-only, CPU-only.
"""
from __future__ import annotations

from pathlib import Path

from artifacts import module_param_totals

KIND_COLOR = {"tok": "#7c3aed", "embed": "#d97706", "norm": "#2563eb",
              "attn": "#6d28d9", "ffn": "#059669", "out": "#b45309"}


def head_dim(dims: dict) -> int:
    return dims["hidden"] // max(int(dims["heads"]), 1)


def projected_params(dims: dict) -> int:
    """Param formula; VERIFIED against the three real finals:
    smoke 12,323,072 / pilot 100,682,496 / target 226,526,208.
    (Same math as scripts/status.py estimate_params; kept local so tests
    don't pull TensorBoard.)"""
    h = int(dims["hidden"])
    hd, kv = head_dim(dims), int(dims["kv_heads"])
    q = int(dims["heads"]) * hd * h
    k = v = kv * hd * h
    o = q
    swiglu = 3 * h * int(dims["ffn"])
    per_layer = q + k + v + o + swiglu + 2 * h          # 2 RMSNorms
    embed = int(dims["vocab"]) * h
    lm = 0 if dims["tie"] else int(dims["vocab"]) * h
    return embed + int(dims["layers"]) * per_layer + h + lm


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
    tok = AutoTokenizer.from_pretrained(str(run_path))
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
```

- [ ] **Step 4: Run tests to green**

Run: `& .\.venv\Scripts\python.exe tests\test_explorer.py`
Expected: `5/5 checks passed`. If `t_projected_formula_matches_all_three` is off by a hair, read the header keys and fix the formula — do NOT loosen the gates beyond ±1%.

- [ ] **Step 5: Commit**

```powershell
git add webui/explorer.py tests/test_explorer.py
git commit -m "feat(webui): U12 explorer - param math (validated 12.3M/100.7M/226.5M), block graph, two-level info"
```

---

### Task 3: `webui/explorer.py` — SVG renderer (TDD)

**Files:**
- Modify: `webui/explorer.py` (append)
- Modify: `tests/test_explorer.py` (append one test before the `__main__` block)

- [ ] **Step 1: Add the failing test**

```python
def t_render_svg():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    svg = E.render_svg(blocks, selected_id="L1.attn")
    assert "L1.attn" in svg and "data-bid" in svg, "no clickable ids"
    assert "http" not in svg, "external asset leaked into SVG"
    assert svg.count("<svg") == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe tests\test_explorer.py`
Expected: FAIL — `AttributeError: ... no attribute 'render_svg'`.

- [ ] **Step 3: Implement `render_svg` (append to `webui/explorer.py`)**

```python
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
```

- [ ] **Step 4: Run tests to green**

Run: `& .\.venv\Scripts\python.exe tests\test_explorer.py`
Expected: `6/6 checks passed`.

- [ ] **Step 5: Commit**

```powershell
git add webui/explorer.py tests/test_explorer.py
git commit -m "feat(webui): U12 SVG architecture renderer (self-contained, clickable ids)"
```

---

### Task 4: `webui/model_tab.py` — Explorer wiring + `app.py` integration

**Files:**
- Create: `webui/model_tab.py` (Explorer part; Simulator lands in Task 6)
- Modify: `webui/app.py` (two exact edits below)

NOTE: `model_tab.py` must NOT import `simulator` yet (created in Task 5) —
it uses `artifacts.run_config` instead.

- [ ] **Step 1: Create `webui/model_tab.py` with the Explorer view**

```python
r"""Model tab (WEBUI_PRD.md §5 U12/U13): nested Architecture Explorer +
Training Simulator. THIN Gradio wiring — logic lives in explorer.py /
simulator.py / artifacts.py. Read-only + CPU-only: no torch model loads,
no VRAM. app.py injects its existing _ckpts / _full_curve helpers (no
duplication, no circular import). Single-user app (PRD §2), so the
simulator keeps one module-global replay state — same pattern as _MODEL.
"""
from __future__ import annotations

import threading
import time as _time

import gradio as gr

from artifacts import (ROOT, hf_config_dims, lora_note, phase_config,
                       run_config, safetensors_header, yaml_config_dims)
import explorer

_BANNER = (
    "### SIMULATION - real data, compressed time\n"
    "Curves, params and eval numbers are replayed from this repo's **real** "
    "runs (tfevents + train_summary.json + eval_report.json). Only the clock "
    "is fake. Reads runs/ only, writes nothing, zero GPU/VRAM - safe beside "
    "a live run.")

_SIM: dict = {"rd": None, "base": None, "cur": 0, "stop": threading.Event(),
              "fig": None}


def _dims_for(source, cfg_name, run_name, ckpts_fn):
    """(dims, header, lora, cfg, run_path) for the current selection."""
    if source == "From trained run":
        path = (ckpts_fn() or {}).get(run_name or "")
        if path is None:
            return None, None, None, None, None
        dims = hf_config_dims(path)
        header = safetensors_header(path)
        note = lora_note(path)
        cfg = run_config((run_name or "").split("/")[0])
        return dims, header, note, cfg, path
    cfg = phase_config(cfg_name or "") if cfg_name else None
    if not cfg:
        return None, None, None, None, None
    peft = cfg.get("peft") or {}
    note = ({"r": peft.get("r"), "alpha": peft.get("lora_alpha"),
             "targets": peft.get("target_modules", [])} if peft else None)
    return yaml_config_dims(cfg), None, note, cfg, None


def _detail_md(blocks, bid, tech):
    b = next((x for x in blocks if x["id"] == bid), None)
    if b is None:
        return "*Pick a block.*"
    lines = [f"### {b['title']}", "", b["tech"] if tech else b["plain"], ""]
    if tech:
        if b["config_pairs"]:
            lines.append("**Config feeding this block:** " + ", ".join(
                f"{k} `{v}`" for k, v in b["config_pairs"]))
        lines.append(f"**Params:** {b['params']:,}"
                     + ("" if b["real"] else " *(projected - no checkpoint)*"))
        lines.append(f"**I/O:** `{b['io'][0]}` -> `{b['io'][1]}`")
        if b["tensors"]:
            lines += ["", "**Real tensors (from the safetensors header):**"]
            lines += [f"- `{n}` `{'x'.join(map(str, t['shape']))}` "
                      f"{t['dtype']}"
                      for n, t in sorted(b["tensors"].items())]
    else:
        lines.append(f"*Params: {b['params']:,}*"
                     + ("" if b["real"] else " *(projected)*"))
    return "\n".join(lines)


def _select_block(blocks, tech, evt):
    idx = evt.index if not isinstance(evt.index, (list, tuple)) else evt.index[0]
    bid = blocks[min(int(idx), len(blocks) - 1)]["id"]
    return _detail_md(blocks, bid, bool(tech))


def _explorer_ui(ckpts_fn):
    cfgs = sorted(p.stem for p in (ROOT / "configs").glob("*.yaml"))
    runs = sorted((ckpts_fn() or {}).keys())
    srcs = ["From config"] + (["From trained run"] if runs else [])

    # build the default view FIRST so components get real initial values
    dims0, header0, note0, cfg0, path0 = _dims_for(
        srcs[0], cfgs[0] if cfgs else "", None, ckpts_fn)
    blocks0 = (explorer.build_graph(dims0, header0, label=cfgs[0], lora=note0)
               if dims0 and dims0.get("layers") else [])
    first0 = blocks0[0]["id"] if blocks0 else ""

    gr.Markdown("# Architecture Explorer (read-only)\n"
                "Built live from configs/*.yaml and the safetensors **headers** "
                "of your checkpoints — tensor data is never read, weights never "
                "loaded. Click any block in the diagram or the list.")

    src = gr.Radio(srcs, value=srcs[0], label="Source")
    cfg_dd = gr.Dropdown(choices=cfgs, value=(cfgs[0] if cfgs else None),
                         label="configs/*.yaml", interactive=True)
    run_dd = gr.Dropdown(choices=runs, value=(runs[0] if runs else None),
                         label="trained run (final)", interactive=True,
                         visible=False)

    state_blocks = gr.State(blocks0)
    state_path = gr.State(path0)
    state_sel = gr.State(first0)
    tech = gr.Checkbox(False,
                       label="Technical detail (shapes, tensor names, math)")

    with gr.Row():
        with gr.Column(scale=5):
            diagram = gr.HTML(
                value=explorer.render_svg(blocks0, first0) if blocks0
                else "*No readable model config.*")
            pick_list = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[[b["title"]] for b in blocks0],
                label="Blocks (click)")
        with gr.Column(scale=4):
            detail_md = gr.Markdown(
                value=_detail_md(blocks0, first0, False) if blocks0
                else "*No readable model config.*")

    gr.Markdown("### Trace a prompt (real tokenizer, real token IDs)")
    with gr.Row():
        trace_in = gr.Textbox("def fibonacci(n):\n    return", label="Prompt",
                              lines=2)
        trace_btn = gr.Button("Trace", variant="primary")
    trace_md = gr.Markdown()

    # SVG click shim handles (the shim script lands in Task 7; harmless now)
    shim_ta = gr.Textbox(visible=False, elem_id="model_tab_sel")
    shim_btn = gr.Button(visible=False, elem_id="model_tab_sel_btn")

    def rebuild(source, cfg_name, run_name):
        dims, header, note, cfg, path = _dims_for(
            source, cfg_name, run_name, ckpts_fn)
        if not dims or not dims.get("layers"):
            msg = "*No readable model config for this selection.*"
            return msg, gr.update(samples=[]), msg, [], None, ""
        blocks = explorer.build_graph(dims, header,
                                      label=run_name or cfg_name, lora=note)
        first = blocks[0]["id"]
        return (explorer.render_svg(blocks, first),
                gr.update(samples=[[b["title"]] for b in blocks]),
                _detail_md(blocks, first, False),
                blocks, path, first)

    outputs = [diagram, pick_list, detail_md, state_blocks, state_path,
               state_sel]
    src.change(rebuild, [src, cfg_dd, run_dd], outputs)
    cfg_dd.change(rebuild, [src, cfg_dd, run_dd], outputs)
    run_dd.change(rebuild, [src, cfg_dd, run_dd], outputs)
    src.change(lambda s: (gr.update(visible=s == "From config"),
                          gr.update(visible=s == "From trained run")),
               [src], [cfg_dd, run_dd])

    pick_list.select(_select_block, [state_blocks, tech], detail_md)
    tech.change(lambda t, blocks, sel: _detail_md(blocks, sel, bool(t)),
                [tech, state_blocks, state_sel], detail_md)
    trace_btn.click(explorer.tokenize_trace, [state_path, trace_in], trace_md)
    trace_in.submit(explorer.tokenize_trace, [state_path, trace_in], trace_md)

    def _shim_pick(bid, blocks, tech_val):
        if not bid and blocks:
            bid = blocks[0]["id"]
        return _detail_md(blocks, bid, bool(tech_val)), bid or ""
    shim_ta.change(_shim_pick, [shim_ta, state_blocks, tech],
                   [detail_md, state_sel])
    shim_btn.click(lambda: None, None, None)


def render_model_tab(ckpts_fn, curve_fn):
    """Called by app.py INSIDE the Blocks context (after the Settings tab)."""
    with gr.Tab("Model"):
        with gr.Tabs():
            with gr.Tab("Architecture Explorer"):
                _explorer_ui(ckpts_fn)
            with gr.Tab("Training Simulator"):
                gr.Markdown(_BANNER)
                gr.Markdown("*Replay engine lands in the next milestone "
                            "step — see WEBUI_PRD.md §5 U13.*")
```

- [ ] **Step 2: Integrate into `app.py` (two exact edits)**

Edit 1 — imports. Find (near the top, after the other imports):
```python
import run_custom
import status
```
replace with:
```python
import run_custom
import status
import model_tab
```

Edit 2 — tab render. Find (line ~1501):
```python
    demo.load(fn=None, inputs=None, outputs=None, js=_POLL_JS)
```
replace with:
```python
    demo.load(fn=None, inputs=None, outputs=None, js=_POLL_JS)

    model_tab.render_model_tab(_ckpts, _full_curve)
```
(The call MUST stay inside the `with gr.Blocks` context — after this line
the context has closed and tabs can no longer be added.)

- [ ] **Step 3: Syntax check + headless boot check (NO browser needed)**

```powershell
& .\.venv\Scripts\python.exe -c "import ast; ast.parse(open('webui/model_tab.py', encoding='utf-8').read()); ast.parse(open('webui/app.py', encoding='utf-8').read()); print('syntax OK')"
```
Expected: `syntax OK`.

Headless boot on a PROBE port (never touch a user-launched 7860 instance):
```powershell
& .\.venv\Scripts\python.exe webui\app.py --no-browser --port 7877   # run as a background job
```
Wait ~20 s, then:
```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7877).StatusCode
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7877/config).Content -match 'Architecture Explorer'
```
Expected: `200` and `True`. Then stop the background job YOU started
(never a process you did not start).

- [ ] **Step 4: Manual interaction check (if a browser is available)**

Open http://127.0.0.1:7877 -> Model tab -> verify: smoke diagram renders;
clicking "Layer 0 - GQA attention" in the block list updates the detail
panel; the Technical toggle adds tensor names; "From trained run" +
`target/final` shows 16 layers. If no browser is available, record
`SKIPPED: manual UI interaction - headless config check only` (honest
reporting).

- [ ] **Step 5: Commit**

```powershell
git add webui/model_tab.py webui/app.py
git commit -m "feat(webui): U12 Model tab - Architecture Explorer (read-only SVG + two-level details + real tokenizer trace)"
```

---

### Task 5: `webui/simulator.py` — replay logic (TDD)

**Files:**
- Create: `webui/simulator.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_simulator.py` exactly:

```python
"""tests for webui/simulator.py — replay logic against REAL run data."""
import _common  # noqa: F401

import artifacts as A
import simulator as S
from _common import run


def tb_curve(logs_dir, tag):
    """Real tfevents reader (same semantics as app.py _full_curve)."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    ea = EventAccumulator(str(logs_dir), size_guidance={"scalars": 2000})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    evs = ea.Scalars(tag)
    return [s.step for s in evs], [s.value for s in evs]


def _rd(name, tech="Pretrain"):
    return S.load_run(name, tech, tb_curve)


def t_load_run_smoke_real():
    rd = _rd("smoke")
    assert rd.steps and rd.train, "no train curve parsed"
    assert rd.max_step >= 190, f"smoke replay ends at {rd.max_step}"
    assert rd.eval_s and rd.eval_v, "no eval curve"
    assert rd.save_steps == 50, rd.save_steps
    assert rd.summary and rd.summary.get("best_eval_loss"), rd.summary


def t_rotation_real_schedule():
    rd = _rd("smoke")
    ev = S.rotation(rd)
    assert [e["step"] for e in ev][:1] == [50], ev
    assert ev[0]["dropped"] is None
    assert ev[-1]["dropped"] == 50, "3-slot rotation did not drop the oldest"


def t_frame_stages_and_gauges():
    rd = _rd("smoke")
    f0 = S.frame_at(rd, 0, stage_override=0)
    assert "Stream" in f0["stages_html"] or "Pairs" in f0["stages_html"]
    fmid = S.frame_at(rd, 120)
    assert fmid["gauge_md"] and "120" in fmid["gauge_md"]
    assert "SIMULATION" in fmid["stages_html"], "honesty banner missing"
    fend = S.frame_at(rd, rd.max_step)
    assert ("checkpoint-200" in fend["events_md"]
            or "checkpoint-150" in fend["events_md"]), fend["events_md"]


def t_end_card_real_numbers():
    rd = _rd("smoke")
    card = S.end_card(rd)
    assert card and str(rd.summary.get("best_eval_loss"))[:4] in card, card


def t_kd_pair_delta_positive():
    kd = _rd("kd-t2p-kd", "KD")
    base = _rd("kd-t2p-baseline", "KD")
    deltas = S.kd_delta(kd, base)
    assert deltas, "no matched eval steps"
    last = deltas[-1][1]
    assert last > 0, f"expected KD ahead (baseline-eval - kd-eval > 0), got {last}"


def t_vram_estimates_anchored():
    pilot = _rd("pilot")
    full = S.vram_est_gb(pilot, "Pretrain")
    assert full is not None and abs(full - 1.91) < 0.25, full  # measured 1.91 GB
    rd_t = S.RunData("target", "Pretrain", [0], [0.0], [], [],
                     A.run_config("target"), {"params_m": 226.5}, None)
    assert abs(S.vram_est_gb(rd_t, "Pretrain") - 4.24) < 0.3   # measured 4.24
    kd_est = S.vram_est_gb(_rd("kd-t2p-kd", "KD"), "KD")
    assert kd_est is not None and abs(kd_est - 7.0) < 1.0  # measured peak 7.01


def t_all_technique_runs_load():
    for tech, names in S.TECHNIQUES.items():
        for n in names:
            rd = _rd(n, tech)
            assert rd.max_step > 0, (n, "empty curve")


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe tests\test_simulator.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'simulator'`.

- [ ] **Step 3: Write `webui/simulator.py`**

Create `webui/simulator.py` exactly:

```python
r"""Training Simulator (real-run REPLAY) — WEBUI_PRD.md §5 U13.

Curves, params and eval numbers are REAL, parsed once from runs/ (tfevents +
train_summary.json + eval_report.json); only TIME is compressed. Zero
writes, zero GPU. Pure logic — model_tab.py owns the Gradio wiring.

Every replayable run was verified on disk (2026-09-09): tfevents + summary
for smoke, pilot, target, sft_t1, sft_v2_e1, h2_copy_lora, h2p2_mixed_lora,
kd-t2p-kd, kd-t2p-baseline, kd-s-t1, kd-s-baseline.
"""
from __future__ import annotations

from artifacts import ROOT, run_config

RUNS = ROOT / "runs"

TECHNIQUES = {
    "Pretrain": ["smoke", "pilot", "target"],
    "SFT": ["sft_t1", "sft_v2_e1"],
    "LoRA": ["h2_copy_lora", "h2p2_mixed_lora"],
    "KD": ["kd-t2p-kd", "kd-s-t1"],
}
KD_PAIRS = {"kd-t2p-kd": "kd-t2p-baseline", "kd-s-t1": "kd-s-baseline"}


class RunData:
    """One parsed run: real curves + real summary + real config knobs."""

    def __init__(self, name, technique, steps, train, eval_s, eval_v,
                 cfg, summary, report):
        self.name, self.technique = name, technique
        self.steps, self.train = steps, train
        self.eval_s, self.eval_v = eval_s, eval_v
        self.cfg = cfg or {}
        self.summary = summary or {}
        self.report = report or {}
        self.max_step = int(steps[-1]) if steps else 0
        tr = self.cfg.get("train", {})
        self.save_steps = int(tr.get("save_steps", 500))
        self.eval_every = int(tr.get("eval_steps", 500))


def load_run(name: str, technique: str, curve_fn) -> RunData:
    """curve_fn = app.py's _full_curve(logs_dir, tag) -> (steps, values)."""
    import status as _status
    cfg = run_config(name)
    logs = RUNS / name / "logs"
    s, v = curve_fn(logs, "train/loss")
    es, ev = curve_fn(logs, "eval/loss")
    summary = _status.load_json(RUNS / name / "final" / "train_summary.json")
    report = _status.load_json(RUNS / name / "final" / "eval_report.json")
    return RunData(name, technique, s, v, es, ev, cfg, summary, report)


def rotation(rd: RunData) -> list[dict]:
    """Checkpoint events up to max_step with the REAL 3-slot rotation
    (save_total_limit=3 — exactly what live runs do)."""
    ev, slots = [], []
    limit = int((rd.cfg.get("train", {}) or {}).get("save_total_limit", 3))
    for step in range(rd.save_steps, int(rd.max_step) + 1, rd.save_steps):
        dropped = slots.pop(0) if len(slots) >= limit else None
        slots.append(step)
        ev.append({"step": step, "dropped": dropped, "slots": list(slots)})
    return ev


def vram_est_gb(rd: RunData, technique: str):
    """GB, anchored to this machine's MEASURED probes (TASKS rows 10/11/31):
    pilot full-FT fp32 = 1.91, target full-FT = 4.24, target 8-bit = 1.36,
    pilot-arch LoRA = 0.72. Heuristic ~19 B/param full (fp32 weights + fp32
    Adam m/v + grads/acts), ~6 B/param 8-bit, ~3.7 B/param LoRA — each lands
    within a few percent of the measured number. The UI labels it 'est'."""
    params_m = rd.summary.get("params_m")
    if not params_m:
        return None
    per_b = {"Pretrain": 19.0, "SFT": 19.0, "LoRA": 3.7, "KD": 19.0}[technique]
    if str(rd.cfg.get("train", {}).get("optim", "")).endswith("8bit") \
            and technique in ("Pretrain", "SFT"):
        per_b = 6.0
    est = params_m * per_b / 1024.0
    if technique == "KD":
        # frozen teacher = runs/target/final, 226.5M fp32 (TASKS row 31);
        # measured kd-t2p-kd peak 7.01 GB = student full-FT + teacher + logits
        est += 4.3
    return est


def _stage(title, body):
    return {"title": title, "body": body}


def stage_chains(rd: RunData) -> list[dict]:
    """Stage cards for this run's technique, with REAL config numbers."""
    tr = rd.cfg.get("train", {})
    data = rd.cfg.get("data", {})
    model = rd.cfg.get("model", {})
    tail = [
        _stage("Train loop", f"Each step: forward -> loss -> backward -> clip "
               f"{tr.get('max_grad_norm', 1.0)} -> optimizer. batch "
               f"{tr.get('batch', 1)} x grad-accum {tr.get('accum', 1)} = "
               f"effective {tr.get('batch', 1) * tr.get('accum', 1)}; "
               f"{tr.get('optim')}; lr {tr.get('lr')} ({tr.get('scheduler')}, "
               f"warmup {tr.get('warmup_steps')}); fp16={tr.get('fp16')}, "
               f"grad-ckpt={tr.get('grad_ckpt')}."),
        _stage("Eval", f"Every {rd.eval_every} steps on the held-out split "
               f"(eval_batch {tr.get('eval_batch', 4)}) — the dots on the curve."),
        _stage("Checkpoint + rotation", f"Every {rd.save_steps} steps, "
               f"save_total_limit {tr.get('save_total_limit', 3)}: the oldest "
               "slot is DELETED (watch the disk animation — rotation is why a "
               "crash never loses more than one checkpoint interval)."),
        _stage("Final + eval report", f"runs/{rd.name}/final -> "
               "train_summary.json + eval_report.json (val loss / ppl / AST "
               "pass rates)."),
    ]
    if rd.technique == "Pretrain":
        head = [
            _stage("Stream + filter",
                   f"Stream ~{data.get('rows')} rows from the dataset list, "
                   f"drop rows < min_chars {data.get('min_chars')}, exact-dedupe "
                   f"(sha1), val split {data.get('val_fraction')}. Streaming = "
                   "disk bounded by rows, never by dataset size."),
            _stage("Tokenize + pack",
                   f"CodeLlama 32k tokenizer; packed into fixed shards of "
                   f"{data.get('shard_tokens')} tokens, cut into ctx "
                   f"{model.get('ctx')} blocks; memmap .bin — the resume "
                   "contract: same shards + zero flags = seamless auto-resume."),
        ]
    elif rd.technique in ("SFT", "LoRA"):
        head = [
            _stage("Pairs -> template",
                   "instruction/response JSONL; every pair is wrapped in the "
                   "config's chat template (the instruct format the Chat tab "
                   "reproduces)."),
            _stage("Filter",
                   f"min_chars {data.get('min_chars')} floors the RESPONSE "
                   "(T1 gotcha: a 200 floor here silently dropped 81% of a "
                   "corpus), instruction floor, optional AST filter, dedupe."),
            _stage("Tokenize @ ctx",
                   f"Packed at the SFT ctx "
                   f"{(rd.cfg.get('sft') or {}).get('ctx', 512)} (shorter than "
                   "base ctx) — instruction/response concatenated, loss on the "
                   "response."),
        ]
        if rd.technique == "LoRA":
            peft = rd.cfg.get("peft", {})
            head.append(_stage(
                "Frozen base + LoRA adapters",
                f"r={peft.get('r')}, alpha={peft.get('lora_alpha')}, "
                f"{len(peft.get('target_modules', []))} target modules; base "
                "weights FROZEN — only adapters train (measured on this "
                "machine: 0.72 GB vs 1.91 GB full-FT). Checkpoints save "
                "adapter-only; final = merge_and_unload -> one full "
                "safetensors."))
    else:  # KD
        head = [
            _stage("Teacher (frozen)",
                   "A bigger frozen model runs forward-only on the same batch "
                   "— its full logit distribution is the target."),
            _stage("Student + loss",
                   "Student forward on the same tokens; loss = "
                   "0.5*KL(teacher||student, tau=1) + 0.5*CE — learn the "
                   "teacher's DISTRIBUTION, not just the next token. This is "
                   "why KD beats plain pretraining at equal steps (real A/B "
                   "on disk)."),
        ]
    return head + tail


def frame_at(rd: RunData, step: int, stage_override: int | None = None) -> dict:
    """Everything the UI shows at one virtual step (pure function)."""
    chains = stage_chains(rd)
    step = max(0, min(int(step), rd.max_step))
    if stage_override is not None:
        stage_idx = max(0, min(stage_override, len(chains) - 1))
    else:
        frac = (step / rd.max_step) if rd.max_step else 0.0
        data_stages = max(1, len(chains) - 4)  # pre-train head stages
        if frac <= 0:
            stage_idx = 0
        else:
            span = len(chains) - data_stages
            stage_idx = min(len(chains) - 1,
                            data_stages + int(frac * max(span - 1, 1)))
    rot = rotation(rd)
    by_step = {e["step"]: e for e in rot}
    events = []
    if step in by_step:
        e = by_step[step]
        events.append(f"checkpoint-{e['step']} saved")
        if e["dropped"] is not None:
            events.append(f"checkpoint-{e['dropped']} rotated out (limit 3)")
    if step in set(rd.eval_s):
        i = rd.eval_s.index(step)
        events.append(f"eval {rd.eval_v[i]:.4f}")
    best = [v for s, v in zip(rd.eval_s, rd.eval_v) if s <= step]
    if step in by_step:
        slots = by_step[step]["slots"]
    else:
        slots = next((e["slots"] for e in reversed(rot) if e["step"] <= step), [])
    gauge = (f"step **{step} / {rd.max_step}**"
             + (f" - best eval so far **{min(best):.4f}**" if best else ""))
    vram = vram_est_gb(rd, rd.technique)
    if vram:
        gauge += f" - VRAM ~{vram:.2f} GB *(est, anchored to measured probes)*"
    tk = (rd.cfg.get("train", {}).get("batch", 1)
          * rd.cfg.get("train", {}).get("accum", 1))
    gauge += f" - ~{tk:,} seq/step (batch x accum)"
    return {"step": step, "stage_idx": stage_idx,
            "events_md": "\n".join("- " + e for e in events) or "_no events_",
            "gauge_md": gauge, "slots": slots,
            "stages_html": render_stages(chains, stage_idx, rd)}


def render_stages(chains, stage_idx, rd) -> str:
    cards = []
    for i, c in enumerate(chains):
        state = "PLAYING" if i == stage_idx else ("done" if i < stage_idx
                                                  else "later")
        hi = ("border:2px solid #2563eb;" if i == stage_idx else "opacity:.6;")
        cards.append(
            f'<div style="border:1px solid #d4d4d8;border-radius:8px;'
            f'padding:8px 10px;margin:6px 0;{hi}"><b>{state}: {c["title"]}'
            f'</b><br/><span style="font-size:12.5px">{c["body"]}</span></div>')
    return ('<div style="font-size:12px;color:#6b7280">SIMULATION - real data, '
            'compressed time; reads runs/ only, writes nothing, zero GPU</div>'
            + "".join(cards))


def kd_delta(kd: RunData, base: RunData) -> list[tuple[int, float]]:
    """(step, baseline_eval - kd_eval) at matched eval steps. >0 = KD ahead.
    Real result on disk: KD ahead at every point, -6.71% at 2000."""
    b = dict(zip(base.eval_s, base.eval_v))
    return [(s, b[s] - v) for s, v in zip(kd.eval_s, kd.eval_v) if s in b]


def end_card(rd: RunData) -> str:
    """The run's REAL summary + eval report - 'what this run got you'."""
    s = rd.summary
    lines = [f"### What `{rd.name}` actually got",
             f"- best eval loss: **{s.get('best_eval_loss')}**"
             f" - params: **{s.get('params_m')}M**"]
    r = rd.report or {}
    if r:
        vl = r.get("val_loss")
        vl = f"{vl:.4f}" if isinstance(vl, (int, float)) else vl
        lines.append(f"- val loss **{vl}** - ppl **{r.get('perplexity')}**")
        ie = r.get("instruction_eval")
        if ie:
            lines.append(f"- instruction AST pass: greedy "
                         f"**{ie.get('greedy_ast_pass_rate')}** / sampled "
                         f"**{ie.get('sampled_ast_pass_rate')}**")
    lines.append(f"- weights: runs/{rd.name}/final (chat-able in the Chat tab)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to green**

Run: `& .\.venv\Scripts\python.exe tests\test_simulator.py`
Expected: `7/7 checks passed`. If `t_kd_pair_delta_positive` fails, verify
which arm is which (kd-t2p-kd = distilled student; gate = baseline_eval −
kd_eval > 0 at the LAST matched step).

- [ ] **Step 5: Cross-check curves against the repo's own parser (PASS-gate evidence)**

```powershell
& .\.venv\Scripts\python.exe -c "import sys; sys.path[:0]=['scripts','webui','.']; import status; from pathlib import Path; print('status.py tail:', status.curve_tail(Path('runs/smoke/logs'), 'eval/loss', 3))"
```
Compare with the last values the test's `tb_curve` parsed (same source file,
two parsers agreeing). Record both numbers in the final report.

- [ ] **Step 6: Commit**

```powershell
git add webui/simulator.py tests/test_simulator.py
git commit -m "feat(webui): U13 simulator logic - real-run replay, rotation, stages, KD delta, anchored VRAM estimates"
```

---

### Task 6: Simulator UI wiring in `webui/model_tab.py`

**Files:**
- Modify: `webui/model_tab.py` (add `import simulator`; replace the Task 4 placeholder view)

- [ ] **Step 1: Add the import (top of file, next to `import explorer`)**

```python
import explorer
import simulator
```

- [ ] **Step 2: Replace the placeholder view with the full Simulator**

In `render_model_tab`, replace:
```python
            with gr.Tab("Training Simulator"):
                gr.Markdown(_BANNER)
                gr.Markdown("*Replay engine lands in the next milestone "
                            "step — see WEBUI_PRD.md §5 U13.*")
```
with:
```python
            with gr.Tab("Training Simulator"):
                _simulator_ui(curve_fn)
```

- [ ] **Step 3: Append the Simulator implementation to `webui/model_tab.py`**

```python
def _loss_fig(rd, upto, base=None):
    """Fresh matplotlib fig per frame; previous closed (app.py leak pattern)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prev = _SIM.get("fig")
    if prev is not None:
        plt.close(prev)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if rd.train:
        xs = [s for s in rd.steps if s <= upto]
        ax.plot(xs, rd.train[:len(xs)], lw=1.4, label=f"{rd.name} train/loss")
    ev = [(s, v) for s, v in zip(rd.eval_s, rd.eval_v) if s <= upto]
    if ev:
        ax.plot([s for s, _ in ev], [v for _, v in ev], "o", ms=4,
                label=f"{rd.name} eval/loss")
    if base is not None:
        bev = [(s, v) for s, v in zip(base.eval_s, base.eval_v) if s <= upto]
        if bev:
            ax.plot([s for s, _ in bev], [v for _, v in bev], "--", lw=1.2,
                    color="tab:red", label=f"{base.name} eval/loss (baseline)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _SIM["fig"] = fig
    return fig


def _sim_outputs(rd, frame):
    """(fig, stages_html, gauges_md, events_md) — the 4 replay components."""
    base = _SIM.get("base")
    fig = _loss_fig(rd, frame["step"], base if rd.technique == "KD" else None)
    delta_md = ""
    if rd.technique == "KD" and base is not None:
        d = simulator.kd_delta(rd, base)
        upto = [x for x in d if x[0] <= frame["step"]]
        if upto:
            s, v = upto[-1]
            delta_md = (f"\n\n**KD vs baseline @ step {s}:** baseline - KD = "
                        f"**{v:+.4f}** ({'KD ahead' if v > 0 else 'baseline ahead'})")
    gauges = frame["gauge_md"]
    if frame["slots"]:
        gauges += "\n\nDisk slots: " + " | ".join(
            f"ckpt-{s}" for s in frame["slots"])
    return fig, frame["stages_html"], gauges, frame["events_md"] + delta_md


def _simulator_ui(curve_fn):
    gr.Markdown(_BANNER)
    tech_radio = gr.Radio(list(simulator.TECHNIQUES), value="Pretrain",
                          label="Technique")
    run_dd = gr.Dropdown(choices=simulator.TECHNIQUES["Pretrain"],
                         value="smoke", label="Real run to replay",
                         interactive=True)
    load_btn = gr.Button("Load run", variant="primary")

    with gr.Row():
        play = gr.Button("Play", variant="primary")
        pause = gr.Button("Pause")
        step_b = gr.Button("+1 tick")
        restart = gr.Button("Restart")
        speed = gr.Slider(60, 3600, value=600, step=60,
                          label="Speed (virtual x; 1 step ~ 10 s of real training)")

    scrub = gr.Slider(0, 200, value=0, step=1, label="Scrub (step)")
    fig = gr.Plot(label="loss — real tfevents, replayed")
    stages_html = gr.HTML()
    gauges_md = gr.Markdown()
    events_md = gr.Markdown()
    end_md = gr.Markdown()

    def _load(tech, name):
        rd = simulator.load_run(name, tech, curve_fn)
        _SIM.update(rd=rd, base=None, cur=0, stop=threading.Event())
        if tech == "KD" and name in simulator.KD_PAIRS:
            _SIM["base"] = simulator.load_run(
                simulator.KD_PAIRS[name], "KD", curve_fn)
        frame = simulator.frame_at(rd, 0, stage_override=0)
        o = _sim_outputs(rd, frame)
        end = (simulator.end_card(rd) if rd.max_step
               else "*No curve data for this run.*")
        return (o[0], o[1], o[2], o[3],
                gr.update(value=0, maximum=rd.max_step), end)

    load_btn.click(_load, [tech_radio, run_dd],
                   [fig, stages_html, gauges_md, events_md, scrub, end_md])

    def _render_at(step):
        rd = _SIM.get("rd")
        if rd is None or not rd.max_step:
            return None, "*Load a run first.*", "", ""
        frame = simulator.frame_at(rd, step)
        _SIM["cur"] = frame["step"]
        return _sim_outputs(rd, frame)

    def _play(speed):
        rd = _SIM.get("rd")
        if rd is None or not rd.max_step:
            yield None, "*Load a run first.*", "", ""
            return
        stop: threading.Event = _SIM["stop"]
        stop.clear()
        head_stages = max(1, len(simulator.stage_chains(rd)) - 4)
        for stage in range(head_stages):  # dwell on the data/tokenize stages
            if stop.is_set():
                break
            yield _sim_outputs(rd, simulator.frame_at(rd, 0,
                                                      stage_override=stage))
            _time.sleep(0.9)
        step = _SIM.get("cur", 0)
        while step < rd.max_step and not stop.is_set():
            step = min(step + max(1, int(speed) // 100), rd.max_step)
            _SIM["cur"] = step
            yield _sim_outputs(rd, simulator.frame_at(rd, step))
            _time.sleep(0.1)
        if not stop.is_set():
            yield _sim_outputs(rd, simulator.frame_at(rd, rd.max_step))

    def _pause():
        _SIM["stop"].set()

    def _restart():
        return _render_at(0)

    def _tick(speed):
        rd = _SIM.get("rd")
        if rd is None:
            return None, "*Load a run first.*", "", ""
        return _render_at(min(_SIM.get("cur", 0) + max(1, int(speed) // 100),
                              rd.max_step))

    frame_out = [fig, stages_html, gauges_md, events_md]
    play.click(_play, [speed], frame_out)
    pause.click(_pause, None, None)
    step_b.click(_tick, [speed], frame_out)
    restart.click(_restart, None, frame_out)
    scrub.change(_render_at, [scrub], frame_out)

    tech_radio.change(
        lambda t: gr.update(choices=simulator.TECHNIQUES[t]),
        [tech_radio], run_dd)
```

- [ ] **Step 4: Syntax + headless boot check**

```powershell
& .\.venv\Scripts\python.exe -c "import ast; ast.parse(open('webui/model_tab.py', encoding='utf-8').read()); print('syntax OK')"
```
Boot on probe port 7877 as in Task 4 Step 3; probe /config for
`Training Simulator`. Expected: 200 + True. Stop your background job after.

- [ ] **Step 5: Commit**

```powershell
git add webui/model_tab.py
git commit -m "feat(webui): U13 simulator UI - play/pause/step/speed/scrub replay with KD-pair overlay"
```

---

### Task 7: SVG click shim (enhancement; the Dataset click is the always-working path)

**Files:**
- Modify: `webui/model_tab.py` (append one static `gr.HTML` inside `_explorer_ui`, after `shim_btn`)

- [ ] **Step 1: Append the shim**

```python
    gr.HTML(
        "<script>window.__dshtModelSelect=function(bid){"
        "var ta=document.querySelector('#model_tab_sel textarea');"
        "if(ta){ta.value=bid;"
        "ta.dispatchEvent(new Event('input',{bubbles:true}));}"
        "var btn=document.querySelector('#model_tab_sel_btn button');"
        "if(btn){btn.click();}};</script>", visible=False)
```

- [ ] **Step 2: Manual verify (browser)**

Click a block in the SVG: the detail panel must switch (same as Dataset
clicks). If the shim does not fire in Gradio 6.26 (programmatic input events
can be swallowed by the Svelte binding), REMOVE the shim entirely — the
clickable `gr.Dataset` layer list already satisfies the U12 PASS gate
("click any block -> details"). Record which path shipped in the final
report.

- [ ] **Step 3: Commit (or revert if shim dropped)**

```powershell
git add webui/model_tab.py
git commit -m "feat(webui): U12 SVG click shim (dataset-list fallback retained)"
```

---

### Task 8: Guided tour (spec says "optional" — include; skip only with an explicit note)

**Files:**
- Modify: `webui/model_tab.py` (small generator in `_explorer_ui`)

- [ ] **Step 1: Add a tour button under the block list (after `trace_md`)**

```python
    tour_btn = gr.Button("Guided tour (walks every block)")

    def _tour(tech):
        blocks = state_blocks.value or []
        for b in blocks:
            yield _detail_md(blocks, b["id"], bool(tech))
            _time.sleep(1.6)
    tour_btn.click(_tour, [tech], detail_md)
```

- [ ] **Step 2: Verify manually (or record SKIPPED + reason), then commit**

```powershell
git add webui/model_tab.py
git commit -m "feat(webui): U12 guided tour (auto-walk blocks with explanations)"
```

---

### Task 9: PASS-gate verification + docs (U12/U13 gates from WEBUI_PRD.md §5)

- [ ] **Step 1: Run ALL test scripts (final evidence)**

```powershell
& .\.venv\Scripts\python.exe tests\test_artifacts.py
& .\.venv\Scripts\python.exe tests\test_explorer.py
& .\.venv\Scripts\python.exe tests\test_simulator.py
```
Expected: 6/6, 6/6, 7/7, all exit 0. Any FAIL -> read the full error,
diagnose, fix within scope; report honestly if unresolvable.

- [ ] **Step 2: U12 gate — real shapes + totals + zero GPU**

Covered by tests (totals within gates of the real 12.3M/100.7M/226.5M). For
"works while a training run is live": the feature is CPU file-read only by
construction; if no live run exists during the session, record
`PASS by construction (read-only + CPU-only); live-run coexistence not
exercised (no live run at implementation time)`.

- [ ] **Step 3: U13 gate — replay correctness + zero writes**

```powershell
git status --porcelain runs/ configs/
```
Expected: EMPTY (the feature wrote nothing anywhere). Smoke replay matches
status.py (Task 5 Step 5 evidence). Scrub/pause/speed verified in the boot
session, or honestly marked SKIPPED with reason.

- [ ] **Step 4: Update TASKS.md rows 39/40 -> done** with a PASS summary +
evidence list (style of rows 21-27). Add a short HANDOFF.md entry (date,
what shipped, evidence). Add a README.md line under the web UI section:
"Model tab — explore any model's architecture (real shapes from checkpoint
headers) and replay real runs (Pretrain/SFT/LoRA/KD) as an accelerated
simulation."

- [ ] **Step 5: Final commit**

```powershell
git add TASKS.md HANDOFF.md README.md
git commit -m "docs: U12/U13 Model tab shipped - TASKS/HANDOFF/README updates (PASS gates evidenced)"
```

---

## Self-review (executed while writing the plan)

1. **Spec coverage:** U12 -> Tasks 1-4 (+7 shim, +8 tour): real headers,
two levels, SVG, prompt trace, projected-mode flag, KV-cache. U13 -> Tasks
5-6: technique radio, all 11 verified runs, stage cards, virtual clock +
play/pause/step/speed/scrub, rotation, VRAM anchors, KD pair, end card,
honesty banner, pluggable curve_fn (app injects `_full_curve`; future
synthetic/what-if + real-CPU-toy providers = same signature, zero UI rework).
2. **Placeholders:** none — every code step is complete code.
3. **Type consistency:** `RunData` fields used identically in Tasks 5/6;
`_sim_outputs` returns a 4-tuple consumed by load/play/scrub handlers;
`curve_fn(logs_dir, tag)` matches app.py's `_full_curve` exactly;
`_dims_for` returns a 5-tuple used identically at build + rebuild time.
