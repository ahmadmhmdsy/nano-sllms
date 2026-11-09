"""Track A context probes (TASKS row 29; research/distill_survey/adoption_plan.md A).

Eval-only Dynamic-NTK + StreamingLLM knobs applied to final checkpoints; CSN
val loss @ 2048/4096 vs the 1024 baseline decides what context Track B targets.
No training code is touched: the knobs live in src/model.py + scripts/eval.py
and default OFF (contract safe).

Run:
  .venv/Scripts/python scripts/ctx_probe.py --selftest     # CPU contract tests
  .venv/Scripts/python scripts/ctx_probe.py --quick 3      # VRAM/pace probe
  .venv/Scripts/python scripts/ctx_probe.py                # full matrix

Matrix (per checkpoint): knobs-off 1024 baseline; raw-extrapolation noop arms;
Dynamic-NTK @ 2048/4096; StreamingLLM sink+window arms (W 512/1024, sink 4/0,
absolute vs cache-relative positions). One JSON per point + summary under
runs/ctx_probes/<UTC-stamp>/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _tiny_model():
    import torch  # noqa: F401
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=96, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True, rope_theta=10000.0)
    cfg.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
    return LlamaForCausalLM(cfg)


def selftest() -> None:
    """CPU contract tests (MEMORY 25 lesson: validate new math before GPU)."""
    import torch

    from src.model import (apply_rope_scaling, build_streaming_sink_mask,
                           reset_rope_scaling, streaming_position_ids)

    torch.manual_seed(0)
    model = _tiny_model().eval()
    L = 48
    ids = torch.randint(3, 96, (1, L))

    # 1) knobs-off determinism: two plain forwards are identical
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=ids).logits
    assert torch.equal(a, b), "knobs-off forward not deterministic"

    # 2) our mask with window >= L and sink 0 equals the default causal path
    full_causal = build_streaming_sink_mask(L, window=L, sink_tokens=0)
    with torch.no_grad():
        c = model(input_ids=ids, attention_mask=full_causal).logits
    assert torch.allclose(a, c, atol=1e-5), "full-window mask deviates from causal"
    print("PASS 1-2  knobs-off determinism + mask==causal", flush=True)

    # 3) sink+window mask pattern + sound semantics. NOTE: exact invariance to
    # out-of-window tokens is impossible (their effect propagates into visible
    # keys through the causal chain), so we test the sound properties:
    # pattern equality, causality under the mask, sinks-live, and that the
    # mask actually restricts vs the full causal path.
    W, S = 16, 4
    m = build_streaming_sink_mask(L, window=W, sink_tokens=S)
    expect = torch.zeros(L, dtype=torch.bool)
    expect[:S] = True
    expect[L - W:] = True
    assert torch.equal(m[0, 0, L - 1], expect), "sink+window pattern wrong"
    # masked reference on the original ids (all sub-checks compare vs this)
    with torch.no_grad():
        f = model(input_ids=ids, attention_mask=m).logits
    # causality: under the SAME mask, query 20 must not see tokens 21..47
    ids2 = ids.clone()
    ids2[0, 21:] = torch.randint(3, 96, (1, L - 21))
    with torch.no_grad():
        d = model(input_ids=ids2, attention_mask=m).logits
    assert torch.allclose(f[:, 20], d[:, 20], atol=1e-5), \
        "future tokens leaked into query 20 (causality broken)"
    # sinks are live: garbling them must move the last query
    ids3 = ids.clone()
    ids3[0, :S] = torch.randint(3, 96, (1, S))
    with torch.no_grad():
        e = model(input_ids=ids3, attention_mask=m).logits
    assert not torch.allclose(f[:, L - 1], e[:, L - 1], atol=1e-5), \
        "sink tokens did not influence the last query"
    # the window truly restricts: masked last row differs from the causal one
    assert not torch.allclose(a[:, L - 1], f[:, L - 1], atol=1e-5), \
        "windowed mask had no effect vs full causal"
    print("PASS 3     sink+window semantics (pattern/causal/sinks/restricts)", flush=True)

    # 4) dynamic-NTK rope: scales beyond max_pos, resets below it
    saved = apply_rope_scaling(model, "dynamic-ntk", 1.0)
    assert saved["rope_type"] == "default", "saved rope params wrong"
    base_freq = model.model.rotary_emb.original_inv_freq.clone()
    with torch.no_grad():
        model(input_ids=torch.randint(3, 96, (1, 128)))
    scaled = model.model.rotary_emb.inv_freq.clone()
    assert not torch.allclose(base_freq, scaled), "NTK scaling did not fire"
    # Qwen dynamic-NTK base at seq=128, max_pos=64, dim = tiny head_dim
    dim = model.config.hidden_size // model.config.num_attention_heads
    expect_base = 10000.0 * (128 / 64) ** (dim / (dim - 2))
    expect_last = 1.0 / (expect_base ** ((dim - 2) / dim))
    got_last = scaled[-1].item()
    assert abs(got_last - expect_last) / expect_last < 1e-4, \
        f"NTK inv_freq mismatch: {got_last} vs {expect_last}"
    with torch.no_grad():
        model(input_ids=ids[:, :32])
    assert torch.allclose(base_freq, model.model.rotary_emb.inv_freq,
                          atol=1e-6), "NTK reset below max_pos failed"
    reset_rope_scaling(model, saved)
    assert model.config.rope_parameters["rope_type"] == "default"
    print("PASS 4     dynamic-NTK scaling + auto-reset", flush=True)

    # 5) remapped positions clamp exactly at sink+window (pos_shift)
    pos = streaming_position_ids(48, window=16, sink_tokens=4)
    assert pos.shape == (1, 48)
    assert (pos[0, 19].item() == 19 and pos[0, 20].item() == 20
            and pos[0, -1].item() == 20), "pos cap wrong"
    assert bool((pos[0, :20] == torch.arange(20)).all()), "pos warmup wrong"
    print("PASS 5     streaming position remap", flush=True)
    print("SELFTEST PASS (CPU)", flush=True)


def gpu_free() -> bool:
    """Never-co-run guard (AGENTS 4): refuse to start beside a GPU job."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # nvidia-smi missing: nothing to guard on
        print(f"[guard] nvidia-smi unavailable ({exc}); continuing", flush=True)
        return True
    busy = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or "N/A" in parts[1]:
            continue  # WDDM quirk: desktop contexts list without per-app MiB
        try:
            mem = float(parts[1].split()[0])
        except (ValueError, IndexError):
            continue
        if mem > 500:
            busy.append(line)
    if busy:
        print("[guard] GPU BUSY (compute apps > 500 MiB):", *busy, sep="\n  ",
              flush=True)
        return False
    print("[guard] GPU-FREE-SENTINEL (no compute app above 500 MiB)", flush=True)
    return True

def full_points() -> list[dict]:
    return [
        {"name": "base_1024", "ctx": 1024},
        {"name": "noop_2048", "ctx": 2048},
        {"name": "ntk_2048", "ctx": 2048, "rope": "dynamic-ntk"},
        {"name": "ntk_4096", "ctx": 4096, "rope": "dynamic-ntk"},
        {"name": "noop_4096", "ctx": 4096},
        {"name": "w1024s4abs_2048", "ctx": 2048, "window": 1024, "sink": 4,
         "positions": "absolute"},
        {"name": "w1024s4remap_2048", "ctx": 2048, "window": 1024, "sink": 4,
         "positions": "remapped"},
        {"name": "w512s4remap_2048", "ctx": 2048, "window": 512, "sink": 4,
         "positions": "remapped"},
        {"name": "w1024s0remap_2048", "ctx": 2048, "window": 1024, "sink": 0,
         "positions": "remapped"},
        {"name": "w1024s4remap_4096", "ctx": 4096, "window": 1024, "sink": 4,
         "positions": "remapped"},
    ]


def quick_points() -> list[dict]:
    return [
        {"name": "base_1024", "ctx": 1024},
        {"name": "ntk_2048", "ctx": 2048, "rope": "dynamic-ntk"},
        {"name": "ntk_4096", "ctx": 4096, "rope": "dynamic-ntk"},
        {"name": "w1024s4remap_4096", "ctx": 4096, "window": 1024, "sink": 4,
         "positions": "remapped"},
    ]


def long_points() -> list[dict]:
    """8k-16k extension (row 29 follow-up, 2026-09-09): NTK arms + noop
    reference @8192 + the only working window variant (mask + absolute
    positions). No pos_shift remap arms: proven broken at 2048/4096. No
    noop @12k/16k: the collapse pattern is established; NTK is the question."""
    return [
        {"name": "noop_8192", "ctx": 8192},
        {"name": "ntk_8192", "ctx": 8192, "rope": "dynamic-ntk"},
        {"name": "ntk_12288", "ctx": 12288, "rope": "dynamic-ntk"},
        {"name": "ntk_16384", "ctx": 16384, "rope": "dynamic-ntk"},
        {"name": "w1024s4abs_8192", "ctx": 8192, "window": 1024, "sink": 4,
         "positions": "absolute"},
    ]


def rescue_points() -> list[dict]:
    """e1 rescue set after the 2026-09-09 suspend/CUBLAS crash in the long run."""
    return [
        {"name": "ntk_16384", "ctx": 16384, "rope": "dynamic-ntk"},
        {"name": "w1024s4abs_8192", "ctx": 8192, "window": 1024, "sink": 4,
         "positions": "absolute"},
    ]


def extreme_points() -> list[dict]:
    """Best-effort window+absolute arm at 16384 (mask 16384^2; OOM-skipped)."""
    return [
        {"name": "w1024s4abs_16384", "ctx": 16384, "window": 1024, "sink": 4,
         "positions": "absolute"},
    ]


def load_model(ckpt: Path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, device


def run_point(model, cfg_path: str, point: dict, quick: int | None) -> dict:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from src.data import PackedDataset
    from src.model import (apply_rope_scaling, build_streaming_sink_mask,
                           reset_rope_scaling, streaming_position_ids)

    cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
    device = next(model.parameters()).device
    ctx = int(point["ctx"])
    saved = None
    if point.get("rope"):
        saved = apply_rope_scaling(model, point["rope"],
                                   point.get("factor", 1.0))
    mask = pos = None
    if point.get("window"):
        mask = build_streaming_sink_mask(ctx, int(point["window"]),
                                         int(point.get("sink", 4)), device)
        if point.get("positions") == "remapped":
            bs_hint = point.get("batch") or (8 if ctx <= 1024 else
                                             (4 if ctx <= 2048 else 1))
            pos = streaming_position_ids(ctx, int(point["window"]),
                                         int(point.get("sink", 4)),
                                         batch_size=bs_hint, device=device)
    bs = point.get("batch") or (8 if ctx <= 1024 else (4 if ctx <= 2048 else 1))
    ds = PackedDataset((ROOT / cfg["data"]["tokens_dir"]).glob("val_*.bin"), ctx)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    total_nll, total_tok, n_batches = 0.0, 0, 0
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            fwd = {"input_ids": ids, "labels": labels}
            if mask is not None:
                fwd["attention_mask"] = mask
            if pos is not None:
                fwd["position_ids"] = pos[: ids.shape[0]]
            out = model(**fwd)
            total_nll += float(out.loss) * labels.numel()
            total_tok += labels.numel()
            n_batches += 1
            if quick is not None and n_batches >= quick:
                break
    wall = time.perf_counter() - t0
    loss = total_nll / max(total_tok, 1)
    if saved is not None:
        reset_rope_scaling(model, saved)
    res = {"point": point["name"], "ctx": ctx, "rope": point.get("rope"),
           "factor": point.get("factor", 1.0), "window": point.get("window"),
           "sink": point.get("sink", 4), "positions": point.get("positions"),
           "batch": bs, "blocks": len(ds), "batches": n_batches,
           "tokens": total_tok, "val_loss": round(loss, 6),
           "perplexity": round(math.exp(min(loss, 20)), 4),
           "quick": quick, "wall_s": round(wall, 1),
           "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                            if on_cuda else None)}
    print("POINT " + json.dumps(res), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", action="append", default=None,
                    help="model dir (repeatable); default target/final + sft_v2_e1/final")
    ap.add_argument("--config", action="append", default=None,
                    help="eval config per ckpt, same order (default auto-detect)")
    ap.add_argument("--points", default="full",
                    choices=["full", "quick", "long", "extreme", "rescue"])
    ap.add_argument("--quick", type=int, default=None,
                    help="cap batches per point (VRAM/pace probe)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    import torch

    if not gpu_free():
        raise SystemExit("[guard] refusing to start: GPU busy (AGENTS 4 never-co-run)")
    ckpts = [Path(p) for p in (args.ckpt or
                               ["runs/target/final", "runs/sft_v2_e1/final"])]
    auto_cfg = {"target": "configs/target.yaml",
                "sft_v2_e1": "configs/sft_v2_e1.yaml"}
    cfgs = list(args.config or [])
    while len(cfgs) < len(ckpts):
        cand = ckpts[len(cfgs)]
        name = cand.parent.name if cand.parent.name in auto_cfg else cand.name
        if name not in auto_cfg:
            raise SystemExit(f"no config known for ckpt {cand}; pass --config")
        cfgs.append(auto_cfg[name])
    points = {"full": full_points, "quick": quick_points,
              "long": long_points, "extreme": extreme_points,
              "rescue": rescue_points}[args.points]()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or ROOT / "runs" / "ctx_probes" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"started_utc": stamp, "points_preset": args.points,
               "quick_batches": args.quick, "ckpt_runs": []}
    for ckpt, cfg_path in zip(ckpts, cfgs):
        print(f"[probe] === ckpt={ckpt} config={cfg_path} ===", flush=True)
        model, device = load_model(ckpt)
        hist_path = ckpt / "eval_report.json"
        hist = None
        if hist_path.exists():
            try:
                hist = json.loads(hist_path.read_text(encoding="utf-8"))["val_loss"]
            except (KeyError, json.JSONDecodeError):
                hist = None
        rows = []
        # safe dir name: "runs/target/final" -> "target__final" (ckpt.name
        # alone would collide - two finals once overwrote each other)
        safe = re.sub(r"[^A-Za-z0-9]+", "__", str(ckpt)).strip("_")
        ckpt_dir = out_dir / safe
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        entry = {"ckpt": str(ckpt), "config": cfg_path, "points": rows}
        summary["ckpt_runs"].append(entry)

        def _flush(row: dict) -> None:
            """Persist each point immediately: a crash/suspend mid-matrix
            (nvlddmkm 153, 2026-09-10) must not lose finished points. entry
            references the same rows list, so summary stays current too."""
            (ckpt_dir / (row["point"] + ".json")).write_text(
                json.dumps(row, indent=2), encoding="utf-8")
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")

        for p in points:
            try:
                row = run_point(model, cfg_path, p, args.quick)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                print(f"[OOM] {ckpt.name}/{p['name']}: {exc}", flush=True)
                row = {"point": p["name"], "error": "OOM"}
            except RuntimeError as exc:
                # CUDA context faults (e.g. CUBLAS_STATUS_EXECUTION_FAILED after
                # a suspend/wake) must not kill the whole matrix; record and go on.
                torch.cuda.empty_cache()
                print(f"[CUDA-ERR] {ckpt.name}/{p['name']}: {exc}", flush=True)
                row = {"point": p["name"], "error": "RuntimeError: " + str(exc)[:120]}
            rows.append(row)
            _flush(row)
        base = next((r for r in rows
                     if r.get("point") == "base_1024" and "val_loss" in r), None)
        if base is not None and hist is not None:
            base["historical_eval_report_val_loss"] = hist
            base["delta_vs_history"] = round(base["val_loss"] - hist, 6)
            _flush(base)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"[probe] done; artifacts in {out_dir}", flush=True)


if __name__ == "__main__":
    main()