"""One-command status dashboard over every phase in runs/ (read-only, CPU-only).

Scans runs/<phase>/ for checkpoint trainer states, final summaries, eval
reports and TensorBoard curves; adds GPU + disk lines. Safe to run while a
training run is live (it never writes anything under runs/).

Run: .venv/Scripts/python scripts/status.py [--runs runs] [--curve 5]

Milestone C: adds a per-phase throughput line (tokens/sec from TensorBoard
event wall-times) + an indicative MFU estimate (6*N*tok/s vs hardware peak).
MFU constants are datasheet-derived estimates - A/B decisions should use
tokens/sec, which is hardware-independent.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared param math lives in packages/param-math — the same canonical formulas
# webui/explorer.py uses (and the viz/flow TypeScript package owns). The
# directory name has a hyphen, so it goes on sys.path rather than being a
# package name. Works both as a script (python scripts/status.py) and as a
# module imported by webui/app.py.
sys.path.insert(0, str(ROOT / "packages" / "param-math"))

import param_math  # noqa: E402  (path inserted above)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - mid-write files are expected during runs
        return None


def find_latest_checkpoint(phase_dir: Path):
    best = None
    for p in phase_dir.glob("checkpoint-*"):
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
            except ValueError:
                continue
            if best is None or step > best[0]:
                best = (step, p)
    return best


def curve_tail(logs_dir: Path, tag: str, n: int):
    if not logs_dir.is_dir():
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        ea = EventAccumulator(str(logs_dir), size_guidance={"scalars": 500})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return []
        return [(s.step, round(s.value, 4)) for s in ea.Scalars(tag)][-n:]
    except Exception:  # noqa: BLE001 - dashboard must never crash on telemetry
        return []


def scalar_wall(logs_dir: Path, tag: str):
    """[(step, value, wall_time)] for one scalar tag; [] when unavailable."""
    if not logs_dir.is_dir():
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        ea = EventAccumulator(str(logs_dir), size_guidance={"scalars": 500})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return []
        return [(s.step, s.value, s.wall_time) for s in ea.Scalars(tag)]
    except Exception:  # noqa: BLE001
        return []


def load_phase_cfg(phase_name: str):
    """configs/<phase>.yaml for a run dir (dash/underscore both tried)."""
    import yaml
    for cand in (phase_name, phase_name.replace("-", "_")):
        p = ROOT / "configs" / f"{cand}.yaml"
        if p.is_file():
            try:
                return yaml.safe_load(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
    return None


def estimate_params(model_cfg: dict, vocab: int) -> int:
    """Param estimate matching src/model.py's GQA decoder (tied embeddings).

    Verified against the measured finals: S=12.3M, P=100.7M, T=226.5M.
    Delegates to the shared packages/param-math module (p_total_dense) —
    the same canonical formulas webui/explorer.py uses. Tied embeddings add
    no lm_head; untied add one more vocab*h.
    """
    total = param_math.p_total_dense(
        layers=int(model_cfg["layers"]), d=int(model_cfg["hidden"]),
        heads=int(model_cfg["heads"]), kv=int(model_cfg["kv_heads"]),
        ffn=int(model_cfg["ffn"]), vocab=int(vocab))
    if not model_cfg.get("tie_embeddings", True):
        total += int(vocab) * int(model_cfg["hidden"])
    return total


# Quadro RTX 4000 (Turing TU104): ~7.1 TFLOPS FP32 (2x2304 CUDA cores at
# ~1.55 GHz boost); datasheet "65.4 Tensor TFLOPS" includes 2:4 sparsity,
# dense FP16 tensor peak is ~32.7 TFLOPS. Both are ESTIMATES - MFU is
# indicative only; use tokens/sec for A/B decisions.
PEAK_FP32 = 7.1e12
PEAK_FP16_TENSOR_DENSE = 32.7e12


def throughput_line(phase_dir: Path, cfg) -> str | None:
    if not cfg:
        return None
    t, m = cfg.get("train") or {}, cfg.get("model") or {}
    if not ("batch" in t and "accum" in t and "ctx" in m):
        return None
    evs = scalar_wall(phase_dir / "logs", "train/loss")
    if len(evs) < 2:
        return None
    tokens_per_step = int(t["batch"]) * int(t["accum"]) * int(m["ctx"])

    def rate(events):
        dt = events[-1][2] - events[0][2]
        ds = events[-1][0] - events[0][0]
        if dt <= 0 or ds <= 0:
            return None
        return tokens_per_step * ds / dt

    full = rate(evs)
    recent = rate(evs[-10:]) if len(evs) >= 3 else None
    if not full:
        return None
    vocab = int((cfg.get("tokenizer") or {}).get("vocab_size", 32768))
    summ = load_json(phase_dir / "final" / "train_summary.json")
    if summ and summ.get("params_m"):
        n_params = float(summ["params_m"]) * 1e6
        exact = "measured"
    else:
        n_params = estimate_params(m, vocab)
        exact = "est"
    parts = [f"{full:,.0f} tok/s (full run)"]
    if recent:
        parts.append(f"{recent:,.0f} tok/s (last 10 pts)")
    mfu32 = 6 * n_params * full / PEAK_FP32
    mfu16 = 6 * n_params * full / PEAK_FP16_TENSOR_DENSE
    parts.append(f"est MFU {mfu32 * 100:.1f}% fp32 / {mfu16 * 100:.1f}% fp16-tensor "
                 f"({exact} N={n_params / 1e6:.1f}M)")
    return " | ".join(parts)


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        used, total, util, temp = (x.strip() for x in out.split(","))
        return f"{used}/{total} MiB, util {util}%, {temp} C"
    except Exception:  # noqa: BLE001
        return "nvidia-smi unavailable"


def main() -> None:
    ap = argparse.ArgumentParser(description="nano_SLMs phase status dashboard")
    ap.add_argument("--runs", default="runs", help="runs dir (default runs/)")
    ap.add_argument("--curve", type=int, default=5,
                    help="tail points of train/loss + eval/loss to show")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs dir at {runs_dir}")

    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print(f"GPU: {gpu_line()}")
    print(f"Disk free (repo drive): {free_gb:.1f} GB")
    print("=" * 78)

    for phase_dir in sorted(runs_dir.iterdir()):
        if not phase_dir.is_dir() or phase_dir.name == "logs":
            continue
        print(f"[{phase_dir.name}]")
        cfg = load_phase_cfg(phase_dir.name)

        latest = find_latest_checkpoint(phase_dir)
        if latest is not None:
            step, ckpt = latest
            ts = load_json(ckpt / "trainer_state.json")
            if ts:
                epoch = ts.get("epoch", 0)
                best = ts.get("best_metric")
                lr = ""
                for entry in reversed(ts.get("log_history", [])):
                    if "learning_rate" in entry:
                        lr = f"{entry['learning_rate']:.3e}"
                        break
                evals = [(e.get("step"), round(e["eval_loss"], 4))
                         for e in ts.get("log_history", []) if "eval_loss" in e]
                eval_txt = " ".join(f"{s}:{v}" for s, v in evals[-6:])
                print(f"  latest checkpoint : {ckpt.name} (step {step}/{ts.get('max_steps', '?')}, "
                      f"epoch {epoch:.2f}, best_eval {best}, lr {lr})")
                if eval_txt:
                    print(f"  eval curve (state): {eval_txt}")
            else:
                print(f"  latest checkpoint : {ckpt.name} (trainer_state unreadable - writing?)")
        else:
            print("  latest checkpoint : none")

        tlogs = sorted((phase_dir / "logs").glob("events.*")) if (phase_dir / "logs").is_dir() else []
        if tlogs:
            tl = curve_tail(phase_dir / "logs", "train/loss", args.curve)
            el = curve_tail(phase_dir / "logs", "eval/loss", args.curve)
            if tl:
                print("  train/loss (tb)   : " + " ".join(f"{s}:{v}" for s, v in tl))
            if el:
                print("  eval/loss  (tb)   : " + " ".join(f"{s}:{v}" for s, v in el))
            thr = throughput_line(phase_dir, cfg)
            if thr:
                print("  throughput (tb)   : " + thr)

        final = phase_dir / "final"
        if final.is_dir():
            summ = load_json(final / "train_summary.json")
            if summ:
                print(f"  final             : params {summ.get('params_m', '?')}M, "
                      f"best_eval {summ.get('best_eval_loss')}, "
                      f"best_ckpt {Path(str(summ.get('best_checkpoint', '?'))).name}")
            rep = load_json(final / "eval_report.json")
            if rep:
                ppl = rep.get("perplexity")
                line = f"  eval_report       : val_loss {rep.get('val_loss'):.4f}, ppl {ppl}"
                ie = rep.get("instruction_eval")
                if ie:
                    line += (f", ast greedy {ie.get('greedy_ast_pass_rate'):.2f}, "
                             f"sampled {ie.get('sampled_ast_pass_rate'):.2f}")
                print(line)
        print("-" * 78)

    print("Tip: eval_report + train_summary are written by eval.py / the trainers;")
    print("     trainer_state lives in the newest checkpoint (mid-write reads are handled).")


if __name__ == "__main__":
    main()
