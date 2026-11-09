r"""Web UI dashboard + chat - WEBUI_PRD.md milestones U1-U6.

Read-only over runs/ except two places: the chat model service (the only
component allowed to allocate GPU VRAM, under the VRAM policy of WEBUI_PRD.md
§2: GPU by default when idle, warn + CPU while a training run is live, reject
+ CPU when free VRAM is too small) and the Monitor tab's delete flow
(checkpoint folders, confirmation-gated, refused while that run is live).

Run: & .\.venv\Scripts\python.exe webui\app.py   (http://127.0.0.1:7860)
LAN: & .\.venv\Scripts\python.exe webui\app.py --lan --auth user:pass
     [--webhook URL] [--port N] [--no-browser]
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gradio as gr

import run_custom
import status
import model_tab
import v2_reader
import v2_jobs

logger = logging.getLogger(__name__)

RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
CURVE_N = 12
CHAT_PORT = 7860


# ---------------------------------------------------------------- M4.4: TTL cache

# One tiny shared TTL cache backs the three expensive reads (_full_curve,
# _ckpts, _free_vram_mib). A cached value is returned only while it is
# younger than `ttl` seconds AND the cached `version` still matches;
# `version` lets a caller bind invalidation to something other than wall
# time (the curve cache uses the metrics file's mtime, so an edit
# invalidates the entry the moment the file is rewritten). Keys are few
# and bounded, so the dict is never evicted; a racing thread at worst
# reloads once.
_TTL_CACHE: dict[object, tuple[object, float, object]] = {}


def _ttl_cached(key, ttl: float, loader, version=None):
    """Return the cached value for `key`, loading and caching it on a miss.

    `loader` runs only on a miss; its result is stored with `version`
    and a deadline of now + `ttl` (monotonic seconds).
    """
    now = _time.monotonic()
    hit = _TTL_CACHE.get(key)
    if hit is not None and version == hit[0] and now < hit[1]:
        return hit[2]
    value = loader()
    _TTL_CACHE[key] = (version, now + ttl, value)
    return value


_CKPTS_TTL_S = 30.0    # the run/checkpoint set does not change while the app is up
_FREE_VRAM_TTL_S = 5.0  # GPU memory drifts constantly; 5 s bounds staleness
_CURVE_TTL_S = 30.0    # safety net only - the mtime version is the real invalidation


def phases() -> list[str]:
    if not RUNS.is_dir():
        return []
    return sorted(p.name for p in RUNS.iterdir()
                  if p.is_dir() and p.name != "logs")


def curve(logs_dir: Path, tag: str, n: int | None = None):
    vals = status.curve_tail(logs_dir, tag, n or CURVE_N)
    return " ".join(f"{s}:{v}" for s, v in vals) if vals else "-"


def snapshot():
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    head = f"GPU: {status.gpu_line()} | Disk free: {free_gb:.1f} GB"
    head += _runs_footprint()  # U8: runs/ footprint on the Dashboard

    rows = []
    for name in phases():
        d = RUNS / name
        latest = status.find_latest_checkpoint(d)
        ckpt, step, max_s, best = "-", "-", "-", "-"
        if latest:
            ts = status.load_json(latest[1] / "trainer_state.json")
            ckpt = latest[1].name
            if ts:
                step = ts.get("global_step", "?")
                max_s = ts.get("max_steps", "?")
                best = ts.get("best_metric", "-")
        final = d / "final"
        summary = status.load_json(final / "train_summary.json")
        params = f"{summary.get('params_m')}M" if summary else "-"
        rep = status.load_json(final / "eval_report.json")
        if rep:
            ie = rep.get("instruction_eval")
            vl = rep.get("val_loss")
            ppl = rep.get("perplexity")
            eval_txt = (f"val {vl:.4f}, ppl {ppl:.2f}"
                        if isinstance(vl, (int, float)) else f"val {vl}, ppl {ppl}")
            if ie:
                eval_txt += (f", ast {ie.get('greedy_ast_pass_rate')}"
                             f"/{ie.get('sampled_ast_pass_rate')}")
        else:
            eval_txt = "-"
        rows.append([name, ckpt, f"{step}/{max_s}", best, params,
                     curve(d / "logs", "eval/loss"), eval_txt])

    table = [[head]] if not rows else rows
    return (head, table, phases())


def detail(name: str):
    if not name:
        return "*No run selected.*"
    d = RUNS / name
    cfg = status.load_phase_cfg(name)
    lines = [f"### {name}"]
    if cfg:
        m = cfg.get("model", {})
        if m:
            lines.append(f"model: {m.get('layers')}L h{m.get('hidden')} "
                         f"heads {m.get('heads')}/{m.get('kv_heads')} "
                         f"ctx {m.get('ctx')} (est ~"
                         f"{status.estimate_params(m, cfg.get('tokenizer', {}).get('vocab_size', 32768)) / 1e6:.1f}M)")
    latest = status.find_latest_checkpoint(d)
    if latest:
        ts = status.load_json(latest[1] / "trainer_state.json")
        lines.append(f"checkpoint: {latest[1].name}")
        if ts:
            evals = [(e.get("step"), round(e["eval_loss"], 4))
                     for e in ts.get("log_history", []) if "eval_loss" in e]
            lines.append("eval curve (state): "
                         + " ".join(f"{s}:{v}" for s, v in evals))
    tl = curve(d / "logs", "train/loss")
    el = curve(d / "logs", "eval/loss")
    thr = status.throughput_line(d, cfg) if cfg else None
    lines.append(f"train/loss (tb): {tl}")
    lines.append(f"eval/loss  (tb): {el}")
    if thr:
        lines.append(f"throughput: {thr}")
    final = d / "final"
    summ = status.load_json(final / "train_summary.json")
    if summ:
        lines.append(f"final: params {summ.get('params_m')}M, "
                     f"best_eval {summ.get('best_eval_loss')}")
    rep = status.load_json(final / "eval_report.json")
    if rep:
        vl = rep.get("val_loss")
        vl_txt = f"{vl:.4f}" if isinstance(vl, (int, float)) else str(vl)
        lines.append(f"eval_report: val_loss {vl_txt}, "
                     f"ppl {rep.get('perplexity')}")
    ie = (rep or {}).get("instruction_eval")
    if ie:
        lines.append(f"instruction eval: greedy ast {ie.get('greedy_ast_pass_rate')}"
                     f", sampled ast {ie.get('sampled_ast_pass_rate')}")
    ckpts = sorted(p.name for p in d.glob("checkpoint-*") if p.is_dir())
    if ckpts:
        lines.append(f"checkpoints on disk: {', '.join(ckpts)}")
    cfgs = sorted(p.name for p in CONFIGS.glob("*.yaml")) if CONFIGS.is_dir() else []
    if cfgs:
        lines.append(f"\nconfigs/: {', '.join(cfgs)}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------- U2: chat

_MODEL = None  # {"label", "path", "tok", "model", "device", "note"}
# M4.1: every read and write of _MODEL goes through _MODEL_LOCK. The lock
# is never held across I/O, model loading, or generator yields: chat_stream
# copies the references it needs under the lock and then generates from the
# locals, so a concurrent unload_model setting _MODEL = None mid-generation
# can neither crash the stream nor block on the lock.
_MODEL_LOCK = threading.Lock()


def _model_ref():
    """The current _MODEL dict, or None - the reference is copied under
    _MODEL_LOCK, so a concurrent unload cannot tear the read.

    The dict is only ever mutated by unload_model (which nulls its "model"
    entry), so reading stable fields like "device"/"path" from the returned
    reference stays safe even after a concurrent unload.
    """
    with _MODEL_LOCK:
        return _MODEL


def _model_entry():
    """(cfg, tok, model, device, ctx) for the loaded model, or None.

    Copies every reference chat_stream needs under _MODEL_LOCK in one
    critical section, so generation can run on the locals after the lock is
    released - the lock must never be held across the generation yields.
    """
    with _MODEL_LOCK:
        if _MODEL is None:
            return None
        return (_MODEL["cfg"], _MODEL["tok"], _MODEL["model"],
                _MODEL["device"], _MODEL["ctx"])


def _has_weights(p: Path) -> bool:
    return any(f.is_file() and f.suffix in (".safetensors", ".bin")
               and f.name != "training_args.bin" for f in p.iterdir())


def _list_ckpts() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not d.is_dir() or d.name == "logs":
            continue
        if (d / "final").is_dir() and _has_weights(d / "final"):
            out[f"{d.name}/final"] = d / "final"
        for ck in d.glob("checkpoint-*"):
            if ck.is_dir() and _has_weights(ck):
                out[f"{d.name}/{ck.name}"] = ck
    return out


def _ckpts() -> dict[str, Path]:
    """M4.4: the listing is cached for _CKPTS_TTL_S - runs and checkpoints
    do not change while the app is up, so every Chat/SFT load reuses one
    walk of runs/ instead of re-stat-ing every folder."""
    return _ttl_cached("ckpts", _CKPTS_TTL_S, _list_ckpts)


def _query_free_vram_mib() -> int | None:
    """M4.4: the raw nvidia-smi probe, split out from _free_vram_mib so the
    TTL cache wraps it and tests can monkeypatch it without a GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:  # noqa: BLE001
        logger.warning("nvidia-smi free-VRAM probe failed; falling back to CPU",
                       exc_info=True)
        return None


def _free_vram_mib() -> int | None:
    """M4.4: the probe is cached for _FREE_VRAM_TTL_S - nvidia-smi is a
    subprocess spawn per call and the value only drifts slowly."""
    return _ttl_cached("free_vram_mib", _FREE_VRAM_TTL_S, _query_free_vram_mib)


def _weights_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.iterdir()
               if f.is_file() and f.suffix in (".safetensors", ".bin")) / 2**20


def unload_model():
    global _MODEL
    import torch
    with _MODEL_LOCK:
        if _MODEL is not None:
            logger.info("chat model unloaded: %s", _MODEL.get("label"))
            _MODEL["model"] = None
            _MODEL = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _model_status_md()


def load_model(label: str):
    global _MODEL
    if not label:
        return "*Pick a checkpoint first.*"
    path = _ckpts().get(label)
    if path is None:
        return f"*Unknown checkpoint: {label}*"
    entry = _model_ref()
    if entry is not None and entry["label"] == label:
        logger.info("chat model %s already loaded; skipping reload", label)
        return _model_status_md()
    unload_model()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    phase = label.split("/", 1)[0]
    cfg = status.load_phase_cfg(phase) or {}
    ctx = int(cfg.get("model", {}).get("ctx", 1024))
    need_mib = _weights_mb(path) * 2.048 + 600  # fp32 on GPU + activation headroom

    # our own process shows up in nvidia-smi once it has ever held a CUDA
    # context (it lingers after unload) - it is not a co-running trainer
    others = [h for h in run_custom.gpu_compute_pids()
              if not h.startswith(f"{os.getpid()},")]
    if others:
        device, note = "cpu", ("⚠️ A training run is live on the GPU — chat "
                               "loads on CPU (slower) so the run is never touched.")
    else:
        free = _free_vram_mib()
        if free is None:
            device, note = "cpu", "nvidia-smi unavailable — falling back to CPU."
        elif need_mib < free * 0.9:
            device, note = "cuda", f"On GPU ({need_mib:.0f} MiB needed, {free} free)."
        else:
            device, note = "cpu", (f"Rejected GPU: needs ~{need_mib:.0f} MiB, only "
                                   f"{free} free — using CPU instead.")

    tok = AutoTokenizer.from_pretrained(str(path))
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path))
    except OSError as e:
        return f"*Could not load {label}: {e}*"
    model.to(device).eval()
    with _MODEL_LOCK:
        _MODEL = {"label": label, "path": path, "tok": tok, "model": model,
                  "device": device, "ctx": ctx, "cfg": cfg, "note": note}
    logger.info("chat model loaded: %s on %s", label, device)
    return _model_status_md()


def _model_status_md():
    with _MODEL_LOCK:
        if _MODEL is None:
            return "*No model loaded.*"
        m = _MODEL["cfg"].get("model", {})
        size = m.get("layers")
        return (f"**{_MODEL['label']}** on **{_MODEL['device'].upper()}** "
                f"(ctx {_MODEL['ctx']}, {size}L) — {_MODEL['note']}")


def _default_instruct(label: str) -> bool:
    phase = label.split("/", 1)[0]
    cfg = status.load_phase_cfg(phase) or {}
    return "template" in cfg.get("data", {})


_STOP_EVENT = threading.Event()


class _StopOnFlag:
    """U9: generation-level stop - generate() halts at the next token step;
    no process kill, the model stays usable."""

    def __call__(self, input_ids, scores, **_):
        return _STOP_EVENT.is_set()


def _stop_gen():
    _STOP_EVENT.set()
    return "Stop requested."


# M4.2: chat parameter bounds. Out-of-range values are REJECTED with a
# clear error message, never silently clamped: a clamped value would hide
# the user's mistake and run a generation they did not ask for.
CHAT_MAX_NEW_TOKENS = (1, 4096)
CHAT_TEMPERATURE = (0.01, 2.0)
CHAT_TOP_P = (0.01, 1.0)
CHAT_TOP_K = (1, 200)


def validate_chat_params(message, max_new_tokens, temperature, top_p, top_k):
    """M4.2: validate the chat inputs; None when valid, else an error string.

    Pure function (no I/O, no model access) so it is unit-testable without
    a GPU or a checkpoint. chat_stream calls it before any model call.
    """
    errors = []
    if not isinstance(message, str) or message.strip() == "":
        errors.append("message must be a non-empty string.")
    for name, value, (lo, hi) in (
            ("max_new_tokens", max_new_tokens, CHAT_MAX_NEW_TOKENS),
            ("temperature", temperature, CHAT_TEMPERATURE),
            ("top_p", top_p, CHAT_TOP_P),
            ("top_k", top_k, CHAT_TOP_K)):
        try:
            in_range = lo <= float(value) <= hi
        except (TypeError, ValueError, OverflowError):
            in_range = False
        if not in_range:
            errors.append(f"{name} must be between {lo} and {hi} "
                          f"(got {value!r}).")
    if errors:
        logger.info("chat params rejected: %s", " ".join(errors))
    return " ".join(errors) if errors else None


def chat_stream(message, history, instruct, sample, max_new, temp, top_p,
                top_k):
    """U9: TextIteratorStreamer progressive output; yields partial text."""
    # M4.2: a blank box keeps its long-standing placeholder (pinned by the
    # M4.1 tests); every other invalid parameter is rejected here, BEFORE
    # any model call, as a clear line in the chat.
    if isinstance(message, str) and message.strip() == "":
        yield "_(load a checkpoint above, then send a code prefix)_"
        return
    error = validate_chat_params(message, max_new, temp, top_p, top_k)
    if error is not None:
        yield f"❌ {error}"
        return
    # M4.1: copy every _MODEL reference this generation needs (cfg is read
    # for the instruct template below) under the lock, release it, then
    # generate from the locals - a concurrent unload_model setting
    # _MODEL = None mid-generation can neither crash this stream nor block
    # on the lock, which is never held across the yields below.
    entry = _model_entry()
    if entry is None:
        yield "_(load a checkpoint above, then send a code prefix)_"
        return
    cfg, tok, model, device, ctx = entry
    import torch
    from transformers import StoppingCriteriaList, TextIteratorStreamer

    prompt = message
    if instruct:
        tpl = cfg.get("data", {}).get("template")
        if tpl:
            prompt = tpl.format(instruction=message)
        else:
            prompt = message + "\n"  # no template in this config: raw prefix
    room = max(ctx - int(max_new), 16)
    ids_full = tok(prompt, return_tensors="pt").input_ids
    warn = ""
    if ids_full.shape[1] > room:  # U6: degrade gracefully, never crash
        warn = (f"?? input was {ids_full.shape[1]} tokens - truncated to the "
                f"last {room} (ctx {ctx} - max_new {int(max_new)}).\n\n")
    ids = ids_full[:, -room:].to(device)
    kwargs = {"max_new_tokens": int(max_new)}
    if sample:
        kwargs.update(do_sample=True, temperature=float(temp),
                      top_p=float(top_p), top_k=int(top_k))

    _STOP_EVENT.clear()
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    gen_kwargs = dict(input_ids=ids, pad_token_id=tok.eos_token_id,
                      streamer=streamer, stopping_criteria=StoppingCriteriaList(
                          [_StopOnFlag()]), **kwargs)

    def _run():
        try:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        except Exception as e:  # noqa: BLE001 - keep the UI from hanging
            logger.warning("chat generation failed: %s", e, exc_info=True)
            streamer.end()

    threading.Thread(target=_run, daemon=True).start()
    out = warn
    try:
        for piece in streamer:
            out += piece
            yield out
    finally:
        # normal end OR client-cancel (GeneratorExit): make sure the
        # generate thread stops so the model is immediately reusable
        _STOP_EVENT.set()


def pick_checkpoint(label):
    """When the dropdown changes: reset the instruct checkbox to the smart default."""
    return gr.update(value=_default_instruct(label))


# ------------------------------------------------------------- U3: monitor

_LAST_FIG = None


def _event_mtime(logs_dir: Path) -> float:
    """Newest event-file mtime in logs_dir (0.0 when there are none) - the
    cache version for _full_curve: any rewrite of the metrics file changes
    it and invalidates the cached curve."""
    mtimes = [p.stat().st_mtime
              for p in logs_dir.glob("events.out.tfevents.*") if p.is_file()]
    return max(mtimes) if mtimes else 0.0


def _read_full_curve(logs_dir: Path, tag: str):
    """M4.4: the tensorboard read itself, split out from _full_curve so the
    TTL cache wraps it and tests can monkeypatch it."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        ea = EventAccumulator(str(logs_dir), size_guidance={"scalars": 2000})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return [], []
        evs = ea.Scalars(tag)
        return [s.step for s in evs], [s.value for s in evs]
    except Exception:  # noqa: BLE001 - monitor must never crash on telemetry
        logger.warning("tensorboard curve read failed for %s", tag,
                       exc_info=True)
        return [], []


def _full_curve(logs_dir: Path, tag: str):
    """M4.4: the (steps, values) tuple is cached on (logs_dir, tag,
    event-file mtime) - the tensorboard reload is the most expensive read
    on every Monitor tick, and the mtime in the cache key invalidates it
    the moment the metrics file is rewritten."""
    if not logs_dir.is_dir():
        return [], []
    return _ttl_cached(("curve", logs_dir, tag), _CURVE_TTL_S,
                      lambda: _read_full_curve(logs_dir, tag),
                      version=_event_mtime(logs_dir))


def _eta_line(phase: str):
    d = RUNS / phase
    if (d / "final" / "train_summary.json").is_file():
        summ = status.load_json(d / "final" / "train_summary.json")
        return f"**✅ complete** — best_eval {summ.get('best_eval_loss')}"
    latest = status.find_latest_checkpoint(d)
    ts = status.load_json(latest[1] / "trainer_state.json") if latest else None
    if not ts:
        return "No checkpoint yet."
    step, max_s = int(ts.get("global_step", 0)), ts.get("max_steps")
    if not max_s:
        return f"step {step} (max unknown)"
    evs = status.scalar_wall(d / "logs", "train/loss")
    eta = "?"
    if evs:
        # resumed runs append duplicate steps across event files: dedupe by
        # step (last wall wins) so the rate math sees strictly increasing steps
        by_step = {s: t for s, _, t in evs}
        seq = sorted(by_step.items())
        if len(seq) >= 3:
            w = seq[-11:] if len(seq) >= 11 else seq
            dt = w[-1][1] - w[0][1]
            ds = w[-1][0] - w[0][0]
            if dt > 0 and ds > 0:
                rem = (int(max_s) - step) * dt / ds
                eta = (f"~{rem / 3600:.1f} h" if rem >= 3600
                       else f"~{rem / 60:.0f} min")
    pct = 100 * step / int(max_s)
    bar = "█" * int(pct // 4) + "░" * (25 - int(pct // 4))
    return f"**{bar}** {step}/{max_s} ({pct:.0f}%) — ETA {eta}"


def _latest_log(phase: str) -> str:
    d = RUNS / phase
    if not d.is_dir():
        return ""
    logs = [p for p in d.glob("*.log") if p.is_file()]
    if not logs:
        return "(no .log files for this run)"
    log = max(logs, key=lambda p: p.stat().st_mtime)
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unreadable)"
    return f"tail of `{log.name}`:\n```\n" + "\n".join(lines[-40:]) + "\n```"


def monitor_data(phase: str):
    if not phase:
        return None, "Pick a run.", "", ""
    d = RUNS / phase
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        global _LAST_FIG
        if _LAST_FIG is not None:  # timer creates a fig every tick: close the old
            plt.close(_LAST_FIG)
            _LAST_FIG = None
        ts_steps, ts_vals = _full_curve(d / "logs", "train/loss")
        ev_steps, ev_vals = _full_curve(d / "logs", "eval/loss")
        fig, ax = plt.subplots(figsize=(8, 4))
        if ts_steps:
            ax.plot(ts_steps, ts_vals, label="train/loss", linewidth=1)
        if ev_steps:
            ax.plot(ev_steps, ev_vals, label="eval/loss", marker="o",
                    linewidth=1.5)
        if ts_steps or ev_steps:
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no tensorboard curves yet", ha="center")
        fig.tight_layout()
        _LAST_FIG = fig
    except Exception as e:  # noqa: BLE001
        import matplotlib.pyplot as plt
        plt.close("all")
        return None, f"(plot failed: {e})\n\nGPU: {status.gpu_line()}", \
            _latest_log(phase), eval_card(phase)
    prog = _eta_line(phase) + f"\n\nGPU: {status.gpu_line()}"
    cfg = status.load_phase_cfg(phase)
    thr = status.throughput_line(d, cfg) if cfg else None
    if thr:  # U6: tok/s + MFU in Monitor (status.py already computes it)
        prog += f"\n\nThroughput: {thr}"
    chain = _chain_line(phase)
    if chain:
        prog += f"\n\n{chain}"
    return fig, prog, _latest_log(phase), eval_card(phase)


# ------------------------------------------------------------- U5: polish

WEBHOOK_URL = os.environ.get("WEBUI_WEBHOOK_URL")
_NOTIFIED: set[str] = set()


def eval_card(phase: str) -> str:
    """U5: eval report card (val/ppl, ast rates, sample generations)."""
    if not phase:
        return "*No run selected.*"
    final = RUNS / phase / "final"
    rep = status.load_json(final / "eval_report.json") \
        or status.load_json(final / "mini_eval_report.json")
    if not rep:
        return (f"*No eval report for `{phase}` yet — it appears here when "
                "the run finishes and eval completes.*")
    lines = [f"### Eval report — {phase}"]
    vl, ppl = rep.get("val_loss"), rep.get("perplexity")
    if isinstance(vl, (int, float)):
        lines.append(f"**val_loss {vl:.4f} · ppl {ppl:.2f}**")
    ie = rep.get("instruction_eval")
    if ie:
        lines.append(f"instruction eval: greedy ast **{ie.get('greedy_ast_pass_rate')}"
                     f"** · sampled ast **{ie.get('sampled_ast_pass_rate')}**")
    for i, (prompt, gen) in enumerate((rep.get("samples") or {}).items()):
        if i >= 3:
            break
        g = (gen or "").strip()
        if len(g) > 220:
            g = g[:220] + " …"
        lines.append(f"**prompt:** `{prompt.strip()[:80]}`\n```\n{g}\n```")
    return "\n\n".join(lines)


def _notify(msg: str, event: str = "run_finished"):
    try:
        gr.Info(msg)
    except Exception:  # noqa: BLE001 - toast is best-effort (headless runs)
        logger.warning("gr.Info toast failed (headless run?)", exc_info=True)
        pass
    if WEBHOOK_URL:
        try:
            import urllib.request
            req = urllib.request.Request(
                WEBHOOK_URL,
                data=json.dumps({"event": event, "message": msg,
                                 "job": _job()}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:  # noqa: BLE001 - webhook must never break polling
            logger.warning("webhook notify failed (event=%s)", event,
                           exc_info=True)
            pass


def _check_done():
    """U5: fire a done/crashed notification once per launched job."""
    job = _job()
    if not job or not job.get("phase"):
        return
    key = f"{job.get('phase')}@{job.get('started_utc', '')}"
    if key in _NOTIFIED:
        return
    phase = job["phase"]
    summ = status.load_json(RUNS / phase / "final" / "train_summary.json")
    if summ:
        _NOTIFIED.add(key)
        _notify(f"✅ Run `{phase}` finished — best_eval "
                f"{summ.get('best_eval_loss', '?')}. Eval report is in the "
                "Monitor tab.")
    elif job.get("pid") and not _job_alive(job):
        _NOTIFIED.add(key)
        _notify(f"⚠️ Run `{phase}` is no longer alive and has no final "
                "summary — if it crashed, re-launching the same config "
                "auto-resumes from the last checkpoint.",
                event="run_crashed")


def _ckpt_choices(phase: str) -> list[tuple[str, str]]:
    d = RUNS / phase if phase else None
    if not d or not d.is_dir():
        return []
    # U8: per-checkpoint MB on the label (value stays the bare folder name)
    return sorted((f"{p.name} ({_dir_mb(p):.0f} MB)", p.name)
                  for p in d.glob("checkpoint-*") if p.is_dir())


def _delete_ckpt(phase, ckpt, confirm):
    """U5: delete-with-confirmation. Refuses while that run is live, while
    any python GPU job is live if the folder is the live job's, and when the
    chat service still has the folder loaded."""
    remaining = gr.update(choices=_ckpt_choices(phase), value=None)
    if not phase or not ckpt:
        return "Pick a run and a checkpoint first.", remaining, gr.update()
    path = RUNS / phase / ckpt
    if (not ckpt.startswith("checkpoint-")
            or path.resolve().parent != (RUNS / phase).resolve()
            or not path.is_dir()):
        return f"Invalid checkpoint: {ckpt}", remaining, gr.update()
    job = _job()
    if job and job.get("phase") == phase and (_job_alive(job)
                                              or run_custom.gpu_compute_pids()):
        return (f"❌ Refused: run `{phase}` is live — no deletes while it "
                "runs.", remaining, gr.update())
    entry = _model_ref()
    if entry is not None and Path(entry["path"]).resolve() == path.resolve():
        return "❌ Refused: unload this checkpoint in Chat first.", \
            remaining, gr.update()
    if not confirm:
        return "❌ Tick the confirmation box first — deletion is irreversible.", \
            remaining, gr.update()
    mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20
    shutil.rmtree(path)
    logger.info("checkpoint deleted: %s/%s (freed ~%.0f MB)", phase, ckpt, mb)
    return (f"🗑️ Deleted `{phase}/{ckpt}` (freed ~{mb:.0f} MB).",
            remaining, gr.update(value=False))


# ---------------------------------------------------------------- U4: train

PRESETS = {  # dims copied verbatim from the shipped configs
    "Small (~12M, smoke)": dict(layers=4, hidden=256, heads=4, kv_heads=2,
                                ffn=1024, accum=32, ctx_choices=[256, 512]),
    "Medium (~101M, pilot)": dict(layers=12, hidden=768, heads=12, kv_heads=4,
                                  ffn=2048, accum=32, ctx_choices=[512]),
    "Large (~226M, target)": dict(layers=16, hidden=1024, heads=16, kv_heads=4,
                                  ffn=4096, accum=16, ctx_choices=[512, 1024]),
}
LR_PRESETS = {"Pretrain 4e-4": 4.0e-4, "Conservative 2e-4": 2.0e-4,
              "Fine-tune 3e-5": 3.0e-5}
MIN_DISK_GB_BLOCK = 5.0
MIN_DISK_GB_WARN = 15.0
JOB_STATE = ROOT / "webui" / "job_state.json"
ENV_PATH = ROOT / ".env"  # U7: gitignored secrets (verified .gitignore L8)
KNOWN_KEYS = ("HF_TOKEN", "EXA_API_KEY")
MAX_ROWS = 500_000  # U7 hard cap: bounds net time + raw/tokens footprint


def _env_lines() -> list[str]:
    if not ENV_PATH.is_file():
        return []
    return ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()


def _env_get(key: str) -> str:
    for line in _env_lines():
        s = line.strip()
        if s.startswith(f"{key}=") or s.startswith(f"{key} ="):
            v = s.partition("=")[2].strip()
            return v.strip('"').strip("'")
    return ""


def _mask(v: str) -> str:
    if not v:
        return "—"
    return f"••••••{v[-4:]}"


def keys_status_md() -> str:
    known = {k: _env_get(k) for k in KNOWN_KEYS}
    rows = "\n".join(f"| `{k}` | {_mask(v)} |"
                     f" {'set' if v else '**missing**'} |"
                     for k, v in known.items())
    ignored = any(line.strip() == ".env"
                  for line in (ROOT / ".gitignore").read_text(
                      encoding="utf-8", errors="replace").splitlines())
    return (f"`.env` at `{ENV_PATH}` — gitignored: "
            f"{'✅ yes' if ignored else '❌ NO — fix .gitignore!'}\n\n"
            f"| key | value | status |\n|---|---|---|\n{rows}\n\n"
            "Values are never displayed, logged, or written into configs "
            "(masked = last 4 chars). `prepare_data.py` auto-loads this "
            "file (real env vars win).")


def set_key(key: str, value: str):
    if key not in KNOWN_KEYS:
        return keys_status_md(), f"❌ Unknown key: {key}"
    value = (value or "").strip().strip('"').strip("'")
    if not value:
        return keys_status_md(), "❌ Empty value — nothing written."
    out, replaced = [], False
    for line in _env_lines():
        s = line.strip()
        if "=" in s and s.partition("=")[0].strip() == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    return keys_status_md(), f"✅ `{key}` stored ({_mask(value)}); never shown again."


def del_key(key: str):
    if key not in KNOWN_KEYS:
        return keys_status_md(), f"❌ Unknown key: {key}"
    had = bool(_env_get(key))
    out = [line for line in _env_lines()
           if not (line.strip().partition("=")[0].strip() == key
                   and "=" in line)]
    ENV_PATH.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return keys_status_md(), (f"🗑️ `{key}` deleted." if had
                              else f"`{key}` was not set.")


def _job() -> dict | None:
    return status.load_json(JOB_STATE)


def _job_alive(job: dict) -> bool:
    pid = int(job.get("pid", 0))
    if not pid:
        return False
    try:  # Windows: OpenProcess + close - no psutil needed (was an honest gap)
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            k32.CloseHandle(h)
            return True
        return False
    except Exception:  # noqa: BLE001 - can't tell, assume not live
        logger.warning("liveness probe failed for pid %s; assuming not live",
                       pid, exc_info=True)
        return False


def build_config(name, preset, ctx, dataset_mode, hf_name, hf_config,
                 local_path, rows, val_fraction, min_chars, max_steps,
                 lr_label, optim, data_mode):
    p = PRESETS[preset]
    name = (name or "").strip().lower().replace(" ", "-")
    import re as _re
    if not _re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,30}", name):
        return None, "Invalid run name (a-z, 0-9, '-', '_')."
    if name in ("smoke", "pilot", "target", "sft_t1", "custom_example"):
        return None, f"'{name}' is a reserved shipped-config name."
    if dataset_mode == "HF dataset" and not hf_name.strip():
        return None, "HF dataset name is empty."
    if dataset_mode == "Local file" and not (ROOT / local_path.strip()).is_file():
        return None, f"Local file not found: {local_path}"
    if optim not in ("adamw_bnb_8bit", "adamw_torch"):
        return None, f"Unsupported optimizer: {optim}"
    if data_mode not in ("stream", "download"):
        return None, f"Unsupported data mode: {data_mode}"
    if int(rows) > MAX_ROWS:
        return None, f"Rows above the hard cap {MAX_ROWS} (net time + disk)."

    data_cand = ([{"name": hf_name.strip(),
                   **({"config": hf_config.strip()} if hf_config.strip() else {})}]
                 if dataset_mode == "HF dataset"
                 else [{"path": local_path.strip()}])
    lr = LR_PRESETS[lr_label]
    cfg = {
        "name": name,
        "tokenizer": {"name": "codellama/CodeLlama-7b-hf", "vocab_size": 32768},
        "model": {"layers": p["layers"], "hidden": p["hidden"],
                  "heads": p["heads"], "kv_heads": p["kv_heads"],
                  "ffn": p["ffn"], "ctx": int(ctx), "dropout": 0.0,
                  "tie_embeddings": True},
        "data": {"dataset_candidates": data_cand, "rows": int(rows),
                 "val_fraction": float(val_fraction), "dedupe": True,
                 "min_chars": int(min_chars), "shard_tokens": 8000000,
                 "data_mode": data_mode,
                 "raw_dir": f"data/{name}/raw", "tokens_dir": f"data/{name}/tokens"},
        "train": {"output_dir": f"runs/{name}", "final_dir": f"runs/{name}/final",
                  "max_steps": int(max_steps), "batch": 1, "eval_batch": 4,
                  "accum": p["accum"], "lr": lr, "scheduler": "cosine",
                  "warmup_steps": max(10, int(max_steps) // 20),
                  "weight_decay": 0.1, "max_grad_norm": 1.0,
                  "logging_steps": 50, "eval_steps": 500, "save_steps": 500,
                  "save_total_limit": 3, "fp16": True, "grad_ckpt": True,
                  "optim": optim, "dataloader_num_workers": 0,
                  "seed": 42},
        "eval": {"max_new_tokens": 64,
                 "prompts": ["def fibonacci(n):", "class Stack:"]},
    }
    out = CONFIGS / f"webui_{name}.yaml"
    import yaml as _yaml
    if out.is_file():
        # U6 resume-collision guard: a mismatched regenerated config must
        # never auto-resume over an old checkpoint of the same run name.
        old = _yaml.safe_load(out.read_text(encoding="utf-8"))
        if old != cfg:
            return None, (f"❌ Resume-collision: `configs/{out.name}` exists "
                          "with DIFFERENT settings than this form. Re-use a "
                          "fresh run name, or set the form back to the exact "
                          "original values to resume.")
        return out, f"Config unchanged (resume-safe): `{out.name}`"
    out.write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out, f"Config written: `{out.name}`"


def preflight(name, preset, ctx, dataset_mode, hf_name, hf_config,
              local_path, rows, val_fraction, min_chars, max_steps, lr_label,
              optim, data_mode):
    cfg_path, msg = build_config(name, preset, ctx, dataset_mode, hf_name,
                                 hf_config, local_path, rows, val_fraction,
                                 min_chars, max_steps, lr_label, optim,
                                 data_mode)
    if cfg_path is None:
        return f"❌ {msg}", None
    lines = [f"✅ {msg}", f"GPU: {status.gpu_line()}"]
    # U7 disk estimate for the chosen mode (raw ≈ rows × ~0.8 KB avg chars;
    # tokens ≈ rows × ~200 tok × 4 B uint32 — same order), + rotated
    # checkpoint footprint by preset (save_total_limit 3).
    rows_i = int(rows)
    if dataset_mode == "Local file":
        raw_mb = (ROOT / local_path.strip()).stat().st_size / 2**20
    else:
        raw_mb = rows_i * 0.8 / 2**10
        raw_mb *= {"download": 3.0, "stream": 1.0}[data_mode]  # dl keeps full cache
    tok_mb = rows_i * 0.8 / 2**10
    # per-ckpt GB measured on this machine (SFT 226M ckpt = 2.7 GB: ~906 MB
    # weights + 1.8 GB optimizer); save_total_limit caps at 3 rotated
    est_gb = {"Small (~12M, smoke)": 0.14, "Medium (~101M, pilot)": 1.2,
              "Large (~226M, target)": 2.7}[preset]
    lines.append(f"Disk estimate: raw ~{raw_mb:.0f} MB + tokens ~{tok_mb:.0f} MB"
                 f" + ~3 rotated ckpts ~{est_gb * 1024:.0f} MB"
                 + (" (download mode keeps the full HF cache — stream keeps "
                    "only your rows)" if data_mode == "download" else ""))
    blockers = []
    # our own lingering CUDA context (chat loaded+unloaded earlier) is not a
    # co-running trainer - but a chat model STILL on the GPU blocks the launch
    others = [h for h in run_custom.gpu_compute_pids()
              if not h.startswith(f"{os.getpid()},")]
    entry = _model_ref()
    if others:
        blockers.append("a python GPU compute process is live — the chain "
                        "would refuse at the train step")
    elif entry is not None and entry["device"] == "cuda":
        blockers.append("the chat model is loaded on the GPU — unload it in "
                        "the Chat tab first (single-job rule)")
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    lines.append(f"Disk free: {free_gb:.1f} GB")
    if free_gb < MIN_DISK_GB_BLOCK:
        blockers.append(f"only {free_gb:.1f} GB free (< {MIN_DISK_GB_BLOCK} GB)")
    elif free_gb < MIN_DISK_GB_WARN:
        lines.append(f"⚠️ disk below {MIN_DISK_GB_WARN} GB — checkpoints "
                     "rotate (limit 3) but watch it")
    job = _job()
    if job and _job_alive(job):
        blockers.append(f"webui job already live: {job.get('phase')} (pid "
                        f"{job.get('pid')})")
    lines.append("Chain: sanity_check → prepare_data → tokenize_data → "
                 "train → eval (run_custom.py, stop-on-fail, auto-resume).")
    if blockers:
        lines.append("❌ BLOCKED: " + "; ".join(blockers))
    else:
        lines.append("✅ Ready to launch.")
    return "\n\n".join(lines), cfg_path


def launch_job(name, preset, ctx, dataset_mode, hf_name, hf_config,
               local_path, rows, val_fraction, min_chars, max_steps, lr_label,
               optim, data_mode):
    """Preflight, then start the chain as a detached subprocess. NO kill button:
    crash/kill is recoverable by re-running (auto-resume), never by the UI."""
    report, cfg_path = preflight(name, preset, ctx, dataset_mode, hf_name,
                                 hf_config, local_path, rows, val_fraction,
                                 min_chars, max_steps, lr_label, optim,
                                 data_mode)
    if cfg_path is None or "❌ BLOCKED" in report:
        return report
    name = cfg_path.stem.removeprefix("webui_")
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    import time as _time
    log_o = (out_dir / "chain_out.log").open("w", encoding="utf-8")
    log_e = (out_dir / "chain_err.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"),
         str(ROOT / "scripts" / "run_custom.py"), "--config", str(cfg_path)],
        cwd=str(ROOT), stdout=log_o, stderr=log_e)
    JOB_STATE.write_text(
        json.dumps({"phase": name, "config": cfg_path.name, "pid": proc.pid,
                    "started_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  _time.gmtime())}),
        encoding="utf-8")
    logger.info("run launched: %s (pid %d, config %s)", name, proc.pid,
                cfg_path.name)
    return report + (f"\n\n🚀 **Launched** `{name}` (pid {proc.pid}). Watch "
                     "it in the Monitor tab; you'll get a toast when it "
                     "finishes.")


# ------------------------------------------------------- U10: SFT launch path

SFT_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"
SFT_DATASET = "nickrosh/Evol-Instruct-Code-80k-v1"  # ungated (sft_t1 uses it)


def build_sft_config(name, base_label, lora):
    """U10: SFT config from a picked checkpoint (base, template, FT-LR,
    forgetting-guard eval kept); LoRA checkbox -> peft block (row 10 hook)."""
    name = (name or "").strip().lower().replace(" ", "-")
    import re as _re
    if not _re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,30}", name):
        return None, "Invalid run name (a-z, 0-9, '-', '_')."
    if name in ("smoke", "pilot", "target", "sft_t1", "custom_example",
                "lora_example"):
        return None, f"'{name}' is a reserved shipped-config name."
    base = _ckpts().get(base_label)
    if base is None or not (base / "config.json").is_file():
        return None, (f"Base checkpoint not found or has no config.json: "
                      f"{base_label}")
    base_rel = base.relative_to(ROOT).as_posix()          # runs/<r>/<ckpt>
    base_run = base_rel.split("/")[1]
    if not (ROOT / "data" / base_run / "tokens").is_dir():
        # the forgetting-guard regression check packs these val shards
        return None, (f"Forgetting-guard shards missing: data/{base_run}/tokens "
                      "- pick a checkpoint from a run that was tokenized.")
    import json as _json
    base_ctx = _json.loads((base / "config.json").read_text(
        encoding="utf-8")).get("max_position_embeddings", 1024)
    cfg = {
        "name": name,
        "base_model": base_rel,
        "tokenizer": {"name": "codellama/CodeLlama-7b-hf", "vocab_size": 32768},
        "model": {"ctx": base_ctx},
        "data": {"dataset": SFT_DATASET, "raw_dir": f"data/{name}/raw",
                 "dataset_dir": f"data/{name}/ds",
                 "tokens_dir": f"data/{base_run}/tokens", "rows": 20000,
                 "val_fraction": 0.02, "min_chars": 200,
                 "min_instruction_chars": 20, "dedupe": True,
                 "ast_filter": True, "template": SFT_TEMPLATE},
        "sft": {"ctx": 512, "pilot_rows": 5000},
        "train": {"output_dir": f"runs/{name}",
                  "final_dir": f"runs/{name}/final", "epochs": 2, "batch": 1,
                  "eval_batch": 4, "accum": 16, "lr": 3.0e-5,
                  "scheduler": "cosine", "warmup_steps": 100,
                  "weight_decay": 0.1, "max_grad_norm": 1.0,
                  "logging_steps": 50, "eval_steps": 250, "save_steps": 250,
                  "save_total_limit": 3, "fp16": True, "grad_ckpt": True,
                  "optim": "adamw_torch", "dataloader_num_workers": 0,
                  "seed": 42},
        "eval": {"max_new_tokens": 256,
                 "instructions_file":
                     f"data/{name}/raw/val_instructions.jsonl",
                 "n_instructions": 50, "sample_temperature": 0.8,
                 "prompts": [
                     "### Instruction:\nWrite a Python function that returns "
                     "the nth Fibonacci number.\n### Response:\n",
                     "### Instruction:\nWrite a Python class that implements a "
                     "stack with push, pop and peek.\n### Response:\n"]},
    }
    if lora:  # maybe_wrap_peft: adapter-only checkpoints, merged final
        cfg["peft"] = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                       "target_modules": ["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"],
                       "bias": "none"}
    out = CONFIGS / f"{name}.yaml"
    import yaml as _yaml
    if out.is_file():  # same resume-collision guard as build_config
        old = _yaml.safe_load(out.read_text(encoding="utf-8"))
        if old != cfg:
            return None, (f"⚠️ Resume-collision: `configs/{out.name}` exists "
                          "with DIFFERENT settings than this form. Re-use a "
                          "fresh run name, or set the form back to the exact "
                          "original values to resume.")
        return out, f"Config unchanged (resume-safe): `{out.name}`"
    out.write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # chat-lookup stub: the pilot final lands in runs/<name>_pilot, and
    # load_phase_cfg reads configs/<phase>.yaml - without the stub the pilot
    # chat load finds no data.template (instruct mode would degrade to raw)
    n_layers = _json.loads((base / "config.json").read_text(
        encoding="utf-8")).get("num_hidden_layers", 4)
    stub = {"name": f"{name}_pilot",
            "model": {"ctx": base_ctx, "layers": n_layers},
            "data": {"template": SFT_TEMPLATE}, "sft": {"ctx": 512}}
    (CONFIGS / f"{name}_pilot.yaml").write_text(
        _yaml.safe_dump(stub, sort_keys=False), encoding="utf-8")
    return out, (f"Config written: `{out.name}` (+ `{name}_pilot.yaml` "
                 "chat stub)")


def launch_sft(name, base_label, lora, pilot):
    """U10: preflight, then chain sft_data (CPU) -> sft (GPU guard) detached."""
    blockers = []
    job = _job()
    if job and _job_alive(job):
        blockers.append(f"webui job already live: {job.get('phase')} (pid "
                        f"{job.get('pid')})")
    others = [h for h in run_custom.gpu_compute_pids()
              if not h.startswith(f"{os.getpid()},")]
    entry = _model_ref()
    if others:
        blockers.append("a python GPU compute process is live")
    elif entry is not None and entry["device"] == "cuda":
        blockers.append("the chat model is loaded on the GPU - unload it in "
                        "the Chat tab first (single-job rule)")
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < MIN_DISK_GB_BLOCK:
        blockers.append(f"only {free_gb:.1f} GB free")
    cfg_path, msg = build_sft_config(name, base_label, bool(lora))
    if cfg_path is None:
        blockers.append(msg)
    lines = [f"✅ {msg}", f"GPU: {status.gpu_line()}",
             f"Disk free: {free_gb:.1f} GB"]
    if cfg_path is not None:
        w_mb = _weights_mb(_ckpts()[base_label])
        # full-FT ckpt ~= fp32 weights + 2x fp32 Adam moments (~12 B/param vs
        # the fp16 file's ~2 B/param -> ~6x); LoRA checkpoints are adapter-only
        mult = 0.05 if lora else 6.0
        lines.append(f"Disk estimate: base weights {w_mb:.0f} MB -> ~3 rotated "
                     f"ckpts ~{w_mb * mult * 3 / 1024:.1f} GB"
                     + (", LoRA saves adapter-only" if lora else ""))
        lines.append("Chain: sft_data (CPU, co-run safe) -> sft (GPU, "
                     "never-co-run guard, auto-resume) via run_custom.py --sft")
    if blockers:
        lines.append("⛔ BLOCKED: " + "; ".join(blockers))
        return "\n\n".join(lines), gr.update(visible=False)
    import time as _time
    out_dir = RUNS / cfg_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_o = (out_dir / "chain_out.log").open("w", encoding="utf-8")
    log_e = (out_dir / "chain_err.log").open("w", encoding="utf-8")
    cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"),
           str(ROOT / "scripts" / "run_custom.py"), "--config", str(cfg_path),
           "--sft"] + (["--pilot"] if pilot else [])
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_o, stderr=log_e)
    JOB_STATE.write_text(
        json.dumps({"phase": cfg_path.stem, "config": cfg_path.name,
                    "pid": proc.pid, "sft": True, "pilot": bool(pilot),
                    "started_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  _time.gmtime())}),
        encoding="utf-8")
    logger.info("SFT run launched: %s (pid %d, pilot=%s)", cfg_path.stem,
                proc.pid, bool(pilot))
    base_rel = _ckpts()[base_label].relative_to(ROOT).as_posix()
    base_run = base_rel.split("/")[1]
    report = (f"🚀 **Launched** SFT `{cfg_path.stem}`"
              + (" (pilot)" if pilot else "") + f" (pid {proc.pid}).\n"
              f"Base = `{base_rel}`; forgetting-guard eval kept (tokens_dir "
              f"-> `data/{base_run}/tokens`).\nWeights land in "
              f"`runs/{cfg_path.stem}{'_pilot' if pilot else ''}/final`; "
              "after the run, chat-able in instruct mode.")
    return report, gr.update(value=cfg_path.read_text(encoding="utf-8"),
                             visible=True)



# ----------------------------------------------------------------- U6: resume

def _stop_requested(phase: str) -> bool:
    return (RUNS / phase / "STOP").is_file() or \
        (RUNS / f"{phase}_pilot" / "STOP").is_file()


def request_stop():
    """U11 cooperative stop: drop a flag the trainer sees at the next save.
    NEVER a process kill - the trainer exits cleanly with a valid
    checkpoint; the exact same command relaunches to resume."""
    job = _job()
    if not job or not _job_alive(job):
        return "⛔ No live job to stop."
    phase = job.get("phase", "?")
    out = RUNS / (f"{phase}_pilot"
                  if job.get("sft") and job.get("pilot") else phase)
    (out / "STOP").write_text(
        _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()), encoding="utf-8")
    return (f"⏹ STOP flag set in `{out.name}/` — the trainer checks it at "
            "the next checkpoint save, exits cleanly (valid checkpoint on "
            "disk), and the exact same zero-flag command relaunches to "
            "resume.")


def _job_banner() -> str:
    """U6: current-job banner for Dashboard + Train (phase, pid, started, ETA)."""
    job = _job()
    if not job:
        return "_No webui-launched job._"
    phase = job.get("phase", "?")
    alive = _job_alive(job)
    started = job.get("started_utc", "?")
    if alive:
        note = ("\n\n⏹ **Stop requested** — the trainer will exit cleanly "
                "at the next checkpoint save." if _stop_requested(phase) else "")
        return (f"?? **Live job: `{phase}`** - pid {job.get('pid')}, "
                f"started {started} UTC\n\n{_eta_line(phase)}" + note)
    # U10: SFT pilots write final to runs/<phase>_pilot - the completed
    # chain json is the one honest "finished" signal for both pipelines
    chain = status.load_json(RUNS / phase / "custom_chain.json")
    if (chain and chain.get("status") == "completed") or \
            (RUNS / phase / "final" / "train_summary.json").is_file() or \
            (RUNS / f"{phase}_pilot" / "final" / "train_summary.json").is_file():
        return (f"? **Last job: `{phase}`** finished "
                f"(started {started} UTC). No job running.")
    return (f"⚠️ **Job `{phase}` is not alive** (pid {job.get('pid')}, "
            f"started {started} UTC) and has no final summary — resume it "
            "from the Monitor tab.")


def _chain_line(phase: str) -> str:
    """U6: chain-step indicator from runs/<phase>/custom_chain.json, so the
    pre-checkpoint phase is visible instead of only 'No checkpoint yet.'"""
    chain = status.load_json(RUNS / phase / "custom_chain.json")
    if not chain:
        return ""
    marks = []
    done = {s["step"]: s["exit"] for s in chain.get("steps", [])}
    for step in ["sanity_check", "prepare_data", "tokenize_data", "train",
                 "eval"]:
        if step in done:
            marks.append(f"{'✅' if done[step] == 0 else '❌'} {step}")
        elif chain.get("status") == f"failed_at_{step}":
            marks.append(f"❌ {step}")
        else:
            marks.append(f"⏳ {step}")
    st = chain.get("status", "?")
    txt = " → ".join(marks) + f"  (chain status: `{st}`)"
    log = RUNS / phase / "chain_out.log"
    if log.is_file():
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")\
                .splitlines()[-6:]
            txt += "\n\nchain_out.log tail:\n```\n" + "\n".join(tail) + "\n```"
        except OSError:
            logger.warning("chain_out.log tail read failed for %s", phase,
                           exc_info=True)
            pass
    return txt


# ------------------------------------------------------- U8: transparency

def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20


def _runs_footprint() -> str:
    """U8: runs/ disk footprint on the Dashboard."""
    if not RUNS.is_dir():
        return ""
    rows = sorted(((_dir_mb(p), p.name) for p in RUNS.iterdir() if p.is_dir()),
                  reverse=True)
    if not rows:
        return ""
    tops = ", ".join(f"`{n}` {mb:.0f} MB" for mb, n in rows[:5])
    return (f"\n\n**runs/ footprint:** {sum(m for m, _ in rows) / 1024:.2f} GB "
            f"across {len(rows)} runs - largest: {tops}")


def preview_data(ds_mode, hf_name, hf_config, local_path, rows, min_chars):
    """U8: first rows + drop counts BEFORE committing, using the exact
    prepare_data rules (text_of + min_chars + sha1-dedupe) on a bounded
    window - so preview counts match a small prepare run 1:1."""
    from prepare_data import iter_local, text_of
    try:
        rows, min_chars = max(int(rows or 0), 10), max(int(min_chars or 0), 0)
    except (TypeError, ValueError):
        return "Rows / min-chars must be numbers."
    cap = min(rows, 2000)  # ponytail: preview cap; unbounded scans stay in prepare_data
    scanned = kept = dmin = ddupe = 0
    seen, samples = set(), []
    it = None
    try:
        if ds_mode == "Local file":
            p = ROOT / str(local_path or "").strip()
            if not p.is_file():
                return f"? Local file not found: `{local_path}`"
            it = iter_local(p)
        else:
            name = str(hf_name or "").strip()
            if not name:
                return "? Give an HF dataset name first."
            from datasets import load_dataset
            it = iter(load_dataset(name, str(hf_config or "").strip() or None,
                                   split="train", streaming=True))
        for ex in it:
            if kept >= cap:
                break
            scanned += 1
            text = text_of(ex)
            if len(text) < min_chars:
                dmin += 1
                continue
            h = __import__("hashlib").sha1(text.encode("utf-8")).hexdigest()
            if h in seen:
                ddupe += 1
                continue
            seen.add(h)
            if len(samples) < 3:
                samples.append(text[:120].replace("\n", " "))
            kept += 1
    except Exception as e:  # noqa: BLE001 - network/format errors are data
        return f"? Preview failed after {scanned} rows: {e}"
    n_val = max(1, round(rows * 0.02))
    lines = [f"**Preview** (window: {scanned} scanned, kept {kept} / target "
             f"{rows}):",
             f"- dropped by min_chars ({min_chars}): {dmin}",
             f"- dropped as duplicates: {ddupe}",
             f"- projected split at target rows: ~{max(rows - n_val, 0)} train "
             f"+ {n_val} val (prepare's formula)"]
    lines += [f"- sample {i+1}: `{s}`" for i, s in enumerate(samples)]
    if scanned < rows:
        lines.append(f"- ? stream ended at {scanned} rows - a full run would "
                     "fail prepare's row check")
    return "\n".join(lines)


_LAST_OVERLAY_FIG = None


def overlay_fig(selected):
    """U8: eval/loss overlay across runs (compare phases)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global _LAST_OVERLAY_FIG
    if _LAST_OVERLAY_FIG is not None:
        plt.close(_LAST_OVERLAY_FIG)
    fig, ax = plt.subplots(figsize=(8, 4))
    found = False
    for name in selected or []:
        d = RUNS / name
        if not d.is_dir():
            continue
        s, v = _full_curve(d / "logs", "eval/loss")
        if s:
            ax.plot(s, v, marker="o", markersize=3, linewidth=1.2, label=name)
            found = True
    if not found:
        ax.text(0.5, 0.5, "no eval curves in the selected runs", ha="center")
        ax.set_xticks([]), ax.set_yticks([])
    else:
        ax.set_xlabel("step"), ax.set_ylabel("eval/loss")
        ax.legend(), ax.grid(alpha=0.3)
    fig.tight_layout()
    _LAST_OVERLAY_FIG = fig
    return fig


def resume_job():
    """U6: one-click Resume — re-launch the exact same config with ZERO
    flags (the auto-resume contract); never edits the config."""
    job = _job()
    if not job:
        return _job_banner(), "Nothing to resume — no webui job recorded."
    if _job_alive(job):
        return _job_banner(), f"Job `{job.get('phase')}` is still live (pid "\
            f"{job.get('pid')}) — nothing to resume."
    cfg_name = job.get("config", "")
    cfg_path = CONFIGS / cfg_name
    if not cfg_name or not cfg_path.is_file():
        return _job_banner(), (f"❌ Resume refused: `{cfg_name}` is missing "
                               "from configs/ — regenerate it in the Train "
                               "tab (identical form inputs) and launch.")
    phase = job.get("phase", cfg_path.stem.removeprefix("webui_"))
    import time as _time
    log_o = (RUNS / phase / "chain_out.log").open("a", encoding="utf-8")
    log_e = (RUNS / phase / "chain_err.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"),
         str(ROOT / "scripts" / "run_custom.py"), "--config", str(cfg_path)],
        cwd=str(ROOT), stdout=log_o, stderr=log_e)
    JOB_STATE.write_text(
        json.dumps({"phase": phase, "config": cfg_name, "pid": proc.pid,
                    "started_utc": job.get("started_utc", ""),
                    "resumed_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  _time.gmtime())}),
        encoding="utf-8")
    logger.info("run resumed: %s (new pid %d, config %s)", phase, proc.pid,
                cfg_name)
    _NOTIFIED.discard(f"{phase}@{job.get('started_utc', '')}")
    return _job_banner(), (f"🚀 **Resumed** `{phase}` (new pid {proc.pid}) — "
                           "same config, zero flags; auto-resume picks up "
                           "from the last checkpoint.")


# ------------------------------------------------------------------ U11: JS

_POLL_JS = """() => {
  let last = null;
  const tick = async () => {
    try {
      if (localStorage.getItem('u11_notif') !== '1') { last = null; return; }
      const j = await (await fetch('/api/job_done')).json();
      if (j.finished && last && !last.finished && window.Notification &&
          Notification.permission === 'granted')
        new Notification('nano_SLMs',
                         {body: 'Run ' + (j.phase || '?') + ' finished'});
      last = j;
    } catch (e) {}
  };
  setInterval(tick, 20000);
  tick();
}"""

_ENABLE_NOTIF_JS = """async () => {
  if (!window.Notification) return '⚠️ Notifications not supported here';
  const p = await Notification.requestPermission();
  localStorage.setItem('u11_notif', p === 'granted' ? '1' : '0');
  return p === 'granted' ? '✅ Browser notifications enabled for run completion'
      : '⚠️ Permission: ' + p;
}"""

# ------------------------------------------------------------- V2: pipeline_v2
# The views below read V2 through webui/v2_reader.py and drive V2 through
# webui/v2_jobs.py. Neither module writes into runs/v2, and neither imports the
# V1 readers, so a V1 defect can never surface as a V2 number.

V2_MODE = "V2 (pipeline_v2)"
V1_MODE = "V1 (legacy)"


def _v2_fig(series, title, xlabel="step", ylabel="loss"):
    """One figure for the given [(label, xs, ys)] series; None when empty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not any(ys for _, _, ys in series):
        plt.close("all")
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, xs, ys in series:
        if xs:
            ax.plot(xs, ys, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def v2_snapshot():
    """The V2 Dashboard: one row per run, straight from the engine labels."""
    choices = v2_reader.run_choices()
    head = "GPU: " + status.gpu_line() + " | runs/v2: " + str(len(choices)) + " run(s)"
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    total = v2_reader.footprint(v2_reader.V2_ROOT)
    head += (f" | Disk free: {free_gb:.1f} GB"
             f" | runs/v2 footprint: {total / 2**30:.1f} GiB")
    table = [[head]] if not choices else [v2_reader.summary_row(c) for c in choices]
    return head, table, choices


def v2_eval_card(label: str) -> str:
    """The eval report card for a V2 run, from the published reports only."""
    root = v2_reader.run_root(label)
    if root is None:
        return "*Pick a V2 run.*"
    rows = v2_reader.metric_rows(root)
    gates = v2_reader.external_gates(root)
    lines = ["### Eval " + v2_reader.code(label)]
    if not rows:
        lines.append("")
        lines.append("*The internal evaluation stage published no metric "
                     "report for this run.*")
    else:
        lines += ["", "| evaluator | metric | value | gate | status |",
                  "|---|---|---|---|---|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
    if gates:
        lines += ["", "**external gates: " + str(gates.get("status", "?"))
                  + "**"]
        for name, gate in sorted((gates.get("gates") or {}).items()):
            lines.append("- " + name + ": " + str(gate.get("status", "?")))
    else:
        lines += ["", "*No external gate record for this run.*"]
    return "\n".join(lines)


def v2_monitor(label: str):
    """The V2 Monitor payload: curve figure, progress, log, eval card."""
    if not label:
        choices = v2_reader.run_choices()
        label = choices[-1] if choices else ""
    if not label:
        return None, "*No V2 runs yet.*", "", "*Pick a V2 run.*"
    root = v2_reader.run_root(label)
    if root is None:
        return None, "*Unknown V2 run.*", "", "*Pick a V2 run.*"

    points = v2_reader.loss_points(root)
    val = v2_reader.scores(root).get("val_loss") or {}
    val_steps = sorted(val, key=lambda k: int(k))
    fig = _v2_fig([("train/loss", [p[0] for p in points], [p[1] for p in points]),
                   ("val_loss @ checkpoint",
                    [int(s) for s in val_steps], [val[s] for s in val_steps])],
                  label)

    state = v2_reader.run_label(root)
    record = v2_reader.checkpoint_record(root) or {}
    pre = v2_reader.preflight(root) or {}
    observation = pre.get("observation") or {}

    bar = ""
    if record.get("final_step") and record.get("max_steps"):
        done = int(record["final_step"])
        total_steps = int(record["max_steps"])
        pct = 100 * done / total_steps
        bar = (chr(0x2588) * int(pct // 4) + chr(0x2591) * (25 - int(pct // 4))
               + f" {done}/{total_steps} ({pct:.0f}%)")
    elif points:
        bar = f"step {points[-1][0]} (max steps not recorded)"

    lines = ["**" + state["status"] + "** "
             + str(len(state["completed_stages"])) + "/"
             + str(len(v2_reader.R1_STAGES)) + " stages"]
    if bar:
        lines += ["", bar]
    if points:
        lines += ["", f"latest train/loss **{points[-1][1]:.4f}** at step "
                  + str(points[-1][0]) + ", from "
                  + v2_reader.code("runs/v2/" + label + "/train/logs/metrics/")]
    if state["reason"]:
        lines += ["", "reason: " + state["reason"]]
    if state["failed_stage"]:
        lines += ["", "failed stage: " + v2_reader.code(state["failed_stage"])]
    if observation.get("name"):
        lines += ["", "GPU: " + status.gpu_line() + " | preflight recorded "
                  + str(observation.get("name"))]
    else:
        lines += ["", "GPU: " + status.gpu_line()]
    lines += ["", v2_jobs.job_banner()]

    events = v2_reader.monitor_events(root, limit=40)
    log = ""
    if events:
        body = "\n".join(
            str(r.get("record")) + ": " + str(r.get("event")) + " "
            + json.dumps(r.get("payload") or {}, sort_keys=True)
            for r in events[-25:])
        log = ("tail of the train stage monitor stream ("
               + v2_reader.code("runs/v2/" + label + "/train/logs/monitor/")
               + "):\n\u0060\u0060\u0060\n" + body + "\n\u0060\u0060\u0060")
    return fig, "\n".join(lines), log, v2_eval_card(label)


def v2_overlay(labels):
    """Overlay several V2 runs train/loss on one figure."""
    series = []
    for label in labels or []:
        root = v2_reader.run_root(label)
        if root is None:
            continue
        points = v2_reader.loss_points(root, limit=300)
        if points:
            series.append((label, [p[0] for p in points],
                           [p[1] for p in points]))
    if not series:
        return None
    return _v2_fig(series, "V2 train/loss overlay")


def v2_model_card(label: str) -> str:
    """The Model tab in V2 mode: the declared architecture, honestly scoped.

    V2 publishes no weight files, so there is no safetensors header and no
    per-tensor inventory to show. This states what the run declared and names
    the limitation instead of rendering an empty parameter explorer that would
    read as a failure of the run rather than a property of the pipeline.
    """
    root = v2_reader.run_root(label)
    if root is None:
        return "*Pick a V2 run.*"
    model = v2_reader.model_identity(root)
    published = v2_reader.published_model_config(root)
    if not model:
        return ("*The model validation stage published no model identity for "
                "this run.*")
    # The architecture fields, the parameter count, the precision and the
    # source hash all live inside capability_metadata, not at the top level.
    # Reading them from the top level renders a 213M model as 0 parameters.
    meta = model.get("capability_metadata") or {}
    rows = [
        ["architecture", model.get("architecture", "-")],
        ["layers", meta.get("layers", "-")],
        ["hidden", meta.get("hidden", "-")],
        ["heads", meta.get("heads", "-")],
        ["kv_heads", meta.get("kv_heads", "-")],
        ["intermediate", meta.get("intermediate", "-")],
        ["context", meta.get("context", "-")],
        ["vocab_size", meta.get("vocab_size", "-")],
        ["tied_embeddings", meta.get("tied_embeddings", "-")],
        ["precision", meta.get("precision", "-")],
        ["attention", meta.get("attention", "-")],
        ["parameters", f"{meta.get('num_parameters', 0):,}"],
        ["initialization", model.get("initialization", "-")],
        ["config_hash", v2_reader.code(model.get("config_hash", "-"))],
        ["source_hash", v2_reader.code(meta.get("source_hash", "-"))],
    ]
    lines = ["### Declared model " + v2_reader.code(label), "",
             "| field | value |", "|---|---|"]
    lines += ["| " + str(k) + " | " + str(v) + " |" for k, v in rows]
    lines += ["", "**Scope of this view.** These are the fields the run "
              "declared and the engine validated. V2 publishes no weight "
              "files, so there is no safetensors header and therefore no "
              "per-tensor inventory here."]
    if published:
        lines += ["", "Published model config: " + v2_reader.code(
            "runs/v2/" + label + "/publish_final/final/model/model_config.json")]
    lines += ["", "Model identity: " + v2_reader.code(
        "runs/v2/" + label + "/model_validation/manifests/model.json")]
    return "\n".join(lines)


def v2_delete_refused(label: str) -> str:
    """Why the checkpoint delete control is refused in V2 mode.

    A V2 checkpoint is hash-chained evidence: its manifest certifies
    model.pt, optimizer.pt and the RNG states by sha256, and the stage marker
    binds those hashes. Deleting one folder would leave a published manifest
    certifying files that are gone, which is precisely the broken evidence the
    V2 artifact rules exist to prevent.
    """
    root = v2_reader.run_root(label)
    if root is None:
        return "Pick a run first."
    rel = root.relative_to(ROOT).as_posix()
    return ("**Refused.** A V2 checkpoint is hash-chained evidence, not a "
            "disposable folder.\n\n"
            + v2_reader.code(rel + "/train/checkpoints/step-*/manifest.json")
            + " certifies every state file by sha256, and the stage marker "
              "binds those hashes. Removing a folder would leave a published "
              "manifest certifying files that no longer exist.\n\nV2 manages "
              "retention itself (best plus last N, recorded in "
            + v2_reader.code(rel + "/train/checkpoints/COMPLETE.json")
            + "). Disk reclamation is a pipeline decision, not a UI one.")


with gr.Blocks(title="nano_SLMs") as demo:
    gr.Markdown("# nano_SLMs")
    mode = gr.Radio([V1_MODE, V2_MODE], value=V2_MODE, label="Pipeline",
                    info="**V2** reads \u0060runs/v2/\u0060 from the engine's own "
                         "published artifacts. **V1** reads \u0060runs/\u0060 with "
                         "the legacy readers. The two never mix: a V1 number is "
                         "never rendered in a V2 view, or the reverse.")

    with gr.Tab("Dashboard"):
        gr.Markdown("## Run dashboard (read-only)")
        head = gr.Markdown()
        dash_job = gr.Markdown(value=v2_jobs.job_banner())
        table = gr.Dataframe(
            headers=["run", "status", "stages", "checkpoint", "best val_loss",
                     "size"],
            interactive=False)
        pick = gr.Dropdown(choices=v2_reader.run_choices(), label="Run detail",
                           allow_custom_value=True)
        info = gr.Markdown()

        V1_HEADERS = ["phase", "latest ckpt", "step", "best_eval", "params",
                      "eval/loss tail", "eval report"]
        V2_HEADERS = ["run", "status", "stages", "checkpoint", "best val_loss",
                      "size"]

        def _dash(mode_value):
            """One place that decides which pipeline a Dashboard view reads."""
            if mode_value == V2_MODE:
                head_md, rows, choices = v2_snapshot()
                return (head_md, gr.update(headers=V2_HEADERS, value=rows),
                        choices, v2_reader.detail_markdown, v2_jobs.job_banner)
            head_md, rows, choices = snapshot()
            return (head_md, gr.update(headers=V1_HEADERS, value=rows),
                    choices, detail, _job_banner)

        def refresh_and_keep(mode_value, name):
            head_md, table_update, choices, detail_fn, banner = _dash(mode_value)
            if name not in choices and choices:
                name = choices[-1]
            return (head_md, table_update,
                    gr.update(choices=choices, value=name), detail_fn(name),
                    banner())

        demo.load(refresh_and_keep, inputs=[mode, pick],
                  outputs=[head, table, pick, info, dash_job])
        timer = gr.Timer(30)
        timer.tick(refresh_and_keep, inputs=[mode, pick],
                   outputs=[head, table, pick, info, dash_job])
        pick.change(lambda m, n: refresh_and_keep(m, n)[3], inputs=[mode, pick],
                    outputs=info)
        mode.change(refresh_and_keep, inputs=[mode, pick],
                    outputs=[head, table, pick, info, dash_job])

    with gr.Tab("Monitor"):
        gr.Markdown("## Live monitor (read-only)")
        mon_job_md = gr.Markdown(value=v2_jobs.job_banner())
        with gr.Row():
            stop_btn = gr.Button("\u23f9 Request stop (cooperative: exits at "
                                 "next checkpoint save \u2014 never a kill)",
                                 variant="stop", interactive=False)
            resume_btn = gr.Button("\u267b\ufe0f Resume dead job (same config, "
                                   "zero flags)")
        stop_md = gr.Markdown()
        resume_md = gr.Markdown()
        mon_dd = gr.Dropdown(choices=v2_reader.run_choices(), label="Run",
                             allow_custom_value=True)
        mon_plot = gr.Plot(label="Loss curves")
        mon_prog = gr.Markdown()
        mon_log = gr.Markdown()
        gr.Markdown("### Eval report card")
        mon_eval = gr.Markdown()
        with gr.Accordion("U8: multi-run loss overlay", open=False):
            ov_pick = gr.Dropdown(choices=v2_reader.run_choices(),
                                  multiselect=True,
                                  label="Runs to overlay (train/loss)")
            ov_btn = gr.Button("Render overlay")
            ov_plot = gr.Plot()
        with gr.Accordion("Danger zone \u2014 delete a checkpoint", open=False):
            danger_note = gr.Markdown()
            del_dd = gr.Dropdown(choices=[], label="Checkpoint (under the selected run)",
                                 interactive=True)
            del_confirm = gr.Checkbox(
                label="I understand this permanently deletes the folder")
            del_btn = gr.Button("Delete checkpoint", variant="stop")
            del_md = gr.Markdown()

        def mon_all(mode_value, name):
            """One place that decides which pipeline a Monitor view reads."""
            if mode_value == V2_MODE:
                if not name:
                    choices = v2_reader.run_choices()
                    name = choices[-1] if choices else ""
                fig, prog, log, ev = v2_monitor(name)
                return (fig, prog, log, ev,
                        gr.update(choices=v2_reader.run_choices(), value=name),
                        gr.update(choices=[], value=None),
                        v2_jobs.job_banner())
            _check_done()
            if not name:
                ps = phases()
                if not ps:
                    return (None, "No runs yet.", "", "",
                            gr.update(), gr.update(choices=[]), _job_banner())
                name = ps[-1]
            fig, prog, log, ev = monitor_data(name)
            return (fig, prog, log, ev,
                    gr.update(choices=phases(), value=name),
                    gr.update(choices=_ckpt_choices(name), value=None),
                    _job_banner())

        def stop_click(mode_value, name):
            if mode_value == V2_MODE:
                return v2_jobs.stop_unavailable_markdown()
            return request_stop()

        def resume_click(mode_value):
            if mode_value == V2_MODE:
                return v2_jobs.resume()
            return resume_job()

        def del_click(mode_value, name, ckpt, confirm):
            if mode_value == V2_MODE:
                return (v2_delete_refused(name), gr.update(choices=[]),
                        gr.update(value=False))
            return _delete_ckpt(name, ckpt, confirm)

        def overlay_click(mode_value, labels):
            if mode_value == V2_MODE:
                return v2_overlay(labels)
            return overlay_fig(labels)

        def sync_mode(mode_value):
            """Grey out the controls V2 has no honest equivalent for.

            The stop control and the checkpoint delete are not hidden in V2
            mode, they are disabled and explained, because a control that
            silently does nothing is worse than one that says it cannot run.
            """
            v1 = mode_value != V2_MODE
            note = ("" if v1 else v2_delete_refused(
                v2_reader.run_choices()[-1] if v2_reader.run_choices() else ""))
            return (gr.update(interactive=v1),
                    gr.update(interactive=v1),
                    gr.update(interactive=v1, value=False),
                    gr.update(choices=phases() if v1 else v2_reader.run_choices()),
                    gr.update(choices=phases() if v1 else v2_reader.run_choices(),
                              multiselect=v1),
                    note)

        stop_btn.click(stop_click, inputs=[mode, mon_dd], outputs=[stop_md])
        resume_btn.click(resume_click, inputs=[mode],
                         outputs=[mon_job_md, resume_md])
        del_btn.click(del_click, inputs=[mode, mon_dd, del_dd, del_confirm],
                      outputs=[del_md, del_dd, del_confirm])
        ov_btn.click(overlay_click, inputs=[mode, ov_pick], outputs=ov_plot)
        demo.load(mon_all, inputs=[mode, mon_dd],
                  outputs=[mon_plot, mon_prog, mon_log, mon_eval, mon_dd,
                           del_dd, mon_job_md])
        mon_timer = gr.Timer(10)
        mon_timer.tick(mon_all, inputs=[mode, mon_dd],
                       outputs=[mon_plot, mon_prog, mon_log, mon_eval,
                                mon_dd, del_dd, mon_job_md])
        mon_dd.change(mon_all, inputs=[mode, mon_dd],
                      outputs=[mon_plot, mon_prog, mon_log, mon_eval,
                               mon_dd, del_dd, mon_job_md])
        mode.change(mon_all, inputs=[mode, mon_dd],
                    outputs=[mon_plot, mon_prog, mon_log, mon_eval,
                             mon_dd, del_dd, mon_job_md])
        mode.change(sync_mode, inputs=[mode],
                    outputs=[stop_btn, del_btn, del_confirm, mon_dd, ov_pick,
                             danger_note])
        demo.load(sync_mode, inputs=[mode],
                  outputs=[stop_btn, del_btn, del_confirm, mon_dd, ov_pick,
                           danger_note])

    with gr.Tab("Train"):
        gr.Markdown(
            "# Configure + launch a training run\n"
            "Generates `configs/webui_<name>.yaml` and starts the standard "
            "chain via `run_custom.py` (sanity → data → tokenize → train → "
            "eval, stop-on-fail, auto-resume). There is **no stop button** — "
            "a crashed/killed run resumes via the one-click Resume button in "
            "the Monitor tab (same config, zero flags).")
        with gr.Row():
            tr_name = gr.Textbox(label="Run name", placeholder="my_first_run")
            tr_preset = gr.Dropdown(choices=list(PRESETS), value=list(PRESETS)[0],
                                    label="Model size preset")
            tr_ctx = gr.Dropdown(choices=["256", "512", "1024"], value="512",
                                 label="Context length")
        with gr.Row():
            ds_mode = gr.Radio(["HF dataset", "Local file"], value="HF dataset",
                               label="Data source")
            hf_name = gr.Textbox(label="HF dataset name",
                                 value="TheGamingMahi/TinyCode")
            hf_cfg = gr.Textbox(label="HF config (optional)", value="")
            local_path = gr.Textbox(label="Local file (repo-relative)",
                                    visible=False)
        # U7: stream-pack default (disk bounded by rows); download caches the
        # full dataset first. Per-step net-feeding was REJECTED in the PRD.
        ds_data_mode = gr.Radio(
            ["Stream (low disk)", "Download full local cache"],
            value="Stream (low disk)", label="Data mode")
        with gr.Row():
            tr_rows = gr.Number(value=20000, label="Rows", precision=0)
            tr_val = gr.Number(value=0.02, label="Val fraction")
            tr_min = gr.Number(value=200, label="Min chars", precision=0)
        with gr.Row():
            tr_steps = gr.Slider(100, 20000, value=3000, step=100,
                                 label="Max steps")
            tr_lr = gr.Dropdown(choices=list(LR_PRESETS), value=list(LR_PRESETS)[0],
                                label="Learning-rate preset")
            # U6: 8-bit is the Milestone B default (TASKS rows 11/18); fp32
            # kept as the manual fallback.
            tr_optim = gr.Dropdown(choices=["adamw_bnb_8bit", "adamw_torch"],
                                   value="adamw_bnb_8bit",
                                   label="Optimizer (8-bit = Milestone B default)")
        train_job_md = gr.Markdown(value=_job_banner())
        gr.Timer(30).tick(_job_banner, inputs=[], outputs=train_job_md)

        def ds_toggle(mode):
            return gr.update(visible=mode == "HF dataset"), \
                gr.update(visible=mode == "HF dataset"), \
                gr.update(visible=mode == "Local file")

        ds_mode.change(ds_toggle, inputs=ds_mode,
                       outputs=[hf_name, hf_cfg, local_path])
        pre_md = gr.Markdown()
        # U8: read-only disclosure of the generated config before Start
        yaml_code = gr.Code(label="Generated config (read-only)",
                            interactive=False, language="yaml",
                            visible=False)

        def pre_and_yaml(*a):
            report, cfg_path = preflight(*a)
            txt = cfg_path.read_text(encoding="utf-8") if cfg_path else ""
            return report, gr.update(value=txt, visible=bool(txt))

        # U8: dataset preview before committing (same rules as prepare_data)
        pre_data_md = gr.Markdown()
        prev_btn = gr.Button("Preview dataset (first rows + drop counts)")
        prev_args = [ds_mode, hf_name, hf_cfg, local_path, tr_rows, tr_min]
        with gr.Row():
            pre_btn = gr.Button("Preflight (writes config)")
            go_btn = gr.Button("?? Start training", variant="primary")
        args = [tr_name, tr_preset, tr_ctx, ds_mode, hf_name, hf_cfg,
                local_path, tr_rows, tr_val, tr_min, tr_steps, tr_lr,
                tr_optim, ds_data_mode]
        pre_btn.click(lambda m, *a: legacy_guard(
            m, lambda: pre_and_yaml(*a)), inputs=[mode] + args,
            outputs=[pre_md, yaml_code])
        go_btn.click(lambda m, *a: legacy_guard(
            m, lambda: launch_job(*a)), inputs=[mode] + args, outputs=pre_md)
        prev_btn.click(lambda m, *a: legacy_guard(
            m, lambda: preview_data(*a)), inputs=[mode] + prev_args,
            outputs=pre_data_md)

        # ------- U10: SFT from checkpoint (sft_data -> sft chain, pilot-first)
        with gr.Accordion("SFT from checkpoint (fine-tune, U10)", open=False):
            gr.Markdown(
                "Fine-tunes a **picked checkpoint** on Evol-Instruct pairs "
                "(instruct template, Fine-tune LR 3e-5, forgetting-guard eval "
                "kept). LoRA = adapter-only checkpoints + merged final. "
                "Chain: `sft_data.py` (CPU, co-run safe) -> `sft.py` under "
                "the single-job lock.")
            sft_name = gr.Textbox(label="SFT run name", placeholder="sft_fib")
            with gr.Row():
                sft_base = gr.Dropdown(choices=list(_ckpts()),
                                       label="Base checkpoint")
                with gr.Column():
                    sft_lora = gr.Checkbox(label="LoRA adapter (peft, r=16)")
                    sft_pilot = gr.Checkbox(
                        label="Pilot (5k pairs, 1 epoch)", value=True)
            sft_btn = gr.Button("Preflight + launch SFT", variant="primary")
        sft_md = gr.Markdown()
        sft_yaml = gr.Code(label="Generated SFT config (read-only)",
                           interactive=False, language="yaml", visible=False)
        sft_btn.click(lambda m, *a: legacy_guard(
            m, lambda: launch_sft(*a)),
            inputs=[mode, sft_name, sft_base, sft_lora, sft_pilot],
            outputs=[sft_md, sft_yaml])

        # ---------------------------------------------- V2 launcher panel
        v2_train_md = gr.Markdown()
        v2_cfg = gr.Dropdown(choices=v2_jobs.experiment_configs(),
                             label="V2 experiment config (shipped)",
                             value=(v2_jobs.experiment_configs() or [None])[0])
        v2_gpu_box = gr.Checkbox(
            label="Probe the device (runs \u0060preflight --gpu\u0060, the only "
                  "V2 command that touches a GPU)")
        with gr.Row():
            v2_validate_btn = gr.Button("Validate")
            v2_plan_btn = gr.Button("Plan (writes nothing)")
            v2_explain_btn = gr.Button("Explain")
            v2_preflight_btn = gr.Button("Preflight")
        with gr.Row():
            v2_go_btn = gr.Button("\U0001f680 Start V2 run", variant="primary")
            v2_resume_btn = gr.Button("\u267b\ufe0f Resume (exact command, "
                                       "zero flags)")
        v2_out = gr.Markdown()
        _v2_validated = {"config": None}

        def v2_validate(cfg):
            """Validate records which config passed, so a launch can be gated
            on this exact config rather than on a stale earlier result."""
            report = v2_jobs.validate_report(cfg)
            _, payload = v2_jobs.run_cli(["validate", "--config", cfg]) if cfg \
                else (1, None)
            ok = bool(cfg) and isinstance(payload, dict) \
                and payload.get("status") == "PASS"
            _v2_validated["config"] = cfg if ok else None
            return report

        def v2_launch(cfg):
            if not cfg:
                return "Pick a config first."
            # The gate is per-config on purpose: validating one config must
            # never authorise a launch of a different one.
            return v2_jobs.launch(cfg, _v2_validated["config"] == cfg)

        v2_validate_btn.click(v2_validate, inputs=[v2_cfg], outputs=[v2_out])
        v2_plan_btn.click(lambda c: v2_jobs.plan_report(c), inputs=[v2_cfg],
                          outputs=[v2_out])
        v2_explain_btn.click(lambda c: v2_jobs.explain_report(c),
                             inputs=[v2_cfg], outputs=[v2_out])
        v2_preflight_btn.click(
            lambda c, g: v2_jobs.preflight_report(c, g), inputs=[v2_cfg, v2_gpu_box],
            outputs=[v2_out])
        v2_go_btn.click(v2_launch, inputs=[v2_cfg], outputs=[v2_out])
        v2_resume_btn.click(lambda: v2_jobs.resume(), outputs=[v2_out])
        demo.load(lambda: v2_jobs.validate_report(v2_cfg.value)
                  if v2_cfg.value else "Pick a V2 config.",
                  outputs=[v2_out])

        def legacy_guard(mode_value, action):
            """V1 launch controls refuse to fire while V2 mode is selected.

            The V1 controls stay on screen rather than being removed, so the
            V1 path remains reachable, but they cannot start a V1 GPU job from
            a screen the operator believes is showing V2.
            """
            if mode_value == V2_MODE:
                return ("**Refused.** The V1 launcher is below, and this is "
                        "the V2 surface. Switch the Pipeline selector to V1 to "
                        "use it. A V1 launch from here would start a V1 GPU "
                        "job on a screen labelled V2, and would not appear in "
                        "any V2 view.")
            return action()

        def train_mode_banner(mode_value):
            if mode_value != V2_MODE:
                return ""
            return ("---\n\n## V2 launcher (active)\n\n"
                    "V2 treats YAML as the adaptation surface, so this panel "
                    "launches a **shipped** config rather than generating one "
                    "from form fields. The exact command is recorded and "
                    "replayed verbatim by Resume; V2 treats any flag change as "
                    "a new run identity, not a resume. The V1 controls lower "
                    "on this tab refuse to fire while V2 is selected.")

        mode.change(train_mode_banner, inputs=[mode], outputs=[v2_train_md])

    with gr.Tab("Chat"):
        gr.Markdown(
            "## Chat with a trained checkpoint\n"
            "Base models are **code completion** models: send a code prefix, "
            "not a question. SFT models: tick **Instruct** to apply the "
            "training template.")
        v2_chat_md = gr.Markdown()
        v2_chat_dd = gr.Dropdown(choices=v2_reader.run_choices(),
                                 label="V2 run")
        model_md = gr.Markdown(value=_model_status_md())
        ckpt_dd = gr.Dropdown(choices=list(_ckpts()), label="Checkpoint",
                              allow_custom_value=True)
        with gr.Row():
            load_btn = gr.Button("Load", variant="primary")
            unload_btn = gr.Button("Unload")
        with gr.Row():
            instruct_box = gr.Checkbox(label="Instruct (apply SFT template)",
                                       value=False)
            sample_box = gr.Checkbox(label="Sample (else greedy)", value=False)
        with gr.Row():
            n_new = gr.Slider(16, 512, value=128, step=16,
                              label="Max new tokens")
            temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        with gr.Row():
            topp = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            topk = gr.Slider(1, 200, value=50, step=1, label="Top-k")

        # U9: generation-level stop (token-level; the model stays usable)
        with gr.Row():
            stop_btn = gr.Button("Stop generation", variant="stop", scale=0)
            stop_md = gr.Markdown()
        stop_btn.click(_stop_gen, inputs=[], outputs=stop_md)

        chat = gr.ChatInterface(
            chat_stream,
            additional_inputs=[instruct_box, sample_box, n_new, temp,
                               topp, topk],
            title=None,
            examples=[["def fibonacci(n):"],
                      ["import numpy as np\n\ndef mean_absolute_error(y_true, y_pred):"],
                      ["class Stack:"]],
        )

        def chat_load(mode_value, v2_label, ckpt_label):
            """Load refuses in V2 mode rather than quietly loading a V1 model.

            The V1 loader scans runs/ for safetensors. Handing it a V2 run
            would either find nothing or, worse, appear to succeed against a
            different run than the one the operator selected.
            """
            if mode_value == V2_MODE:
                return v2_reader.chat_blocked_markdown(v2_label)
            return load_model(ckpt_label)

        load_btn.click(chat_load, inputs=[mode, v2_chat_dd, ckpt_dd],
                       outputs=model_md)
        unload_btn.click(unload_model, outputs=model_md)
        ckpt_dd.change(pick_checkpoint, inputs=ckpt_dd, outputs=instruct_box)
        v2_chat_dd.change(lambda l: v2_reader.chat_blocked_markdown(l),
                          inputs=[v2_chat_dd], outputs=[v2_chat_md])
        mode.change(lambda m: v2_reader.chat_blocked_markdown(v2_chat_dd.value)
                    if m == V2_MODE else "", inputs=[mode],
                    outputs=[v2_chat_md])
        demo.load(lambda: v2_reader.chat_blocked_markdown(
            v2_chat_dd.value) if v2_chat_dd.value else "", outputs=[v2_chat_md])

        def refresh_ckpts(current):
            return gr.update(choices=list(_ckpts()), value=current)

        # U6: a fresh run's checkpoints appear in the picker without a reload
        gr.Timer(15).tick(refresh_ckpts, inputs=ckpt_dd, outputs=ckpt_dd)

    with gr.Tab("Settings"):
        gr.Markdown(
            "# Settings — API keys (U7)\n"
            "Secrets live in the gitignored project-root `.env` "
            "(`prepare_data.py` auto-loads it for gated HF datasets; "
            "`exa_search.py` reads `EXA_API_KEY` from it). Values are "
            "**never displayed, logged, or written into configs** — the "
            "table below shows only the last 4 chars.")
        keys_md = gr.Markdown(value=keys_status_md())
        key_dd = gr.Dropdown(choices=list(KNOWN_KEYS), value="HF_TOKEN",
                             label="Key")
        key_val = gr.Textbox(label="Value (write-only; stored to .env)",
                             type="password")
        with gr.Row():
            key_set_btn = gr.Button("Add / override", variant="primary")
            key_del_btn = gr.Button("Delete key", variant="stop")
        key_md = gr.Markdown()
        key_set_btn.click(set_key, inputs=[key_dd, key_val],
                          outputs=[keys_md, key_md])
        key_del_btn.click(del_key, inputs=[key_dd], outputs=[keys_md, key_md])

        with gr.Accordion("U11: browser notifications (run finished)",
                          open=False):
            gr.Markdown("Asks the browser for permission (stored locally); a "
                        "desktop notification fires when a webui job "
                        "finishes — beside the in-page toast + webhook.")
            notif_btn = gr.Button("Enable browser notifications")
            notif_md = gr.Markdown()
        notif_btn.click(fn=None, inputs=None, outputs=[notif_md],
                        js=_ENABLE_NOTIF_JS)
    demo.load(fn=None, inputs=None, outputs=None, js=_POLL_JS)

    model_tab.render_model_tab(_ckpts, _full_curve, mode=mode,
                                v2_reader=v2_reader)

# ------------------------------------------------------------------ U11: API

def _api_job_done() -> dict:
    """Client-side poller endpoint for browser notifications."""
    job = _job() or {}
    alive = bool(job) and _job_alive(job)
    phase = job.get("phase")
    finished = False
    if phase and not alive:
        chain = status.load_json(RUNS / phase / "custom_chain.json")
        finished = bool(chain and chain.get("status") == "completed") or \
            (RUNS / phase / "final" / "train_summary.json").is_file() or \
            (RUNS / f"{phase}_pilot" / "final" / "train_summary.json").is_file()
    return {"phase": phase, "live": alive, "finished": finished}


demo.app.add_api_route("/api/job_done", _api_job_done, methods=["GET"])

if __name__ == "__main__":
    import argparse
    # M4.3: the module loggers emit at INFO; unconfigured, the root logger
    # stays at WARNING and every info record is dropped on the floor.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # gradio's telemetry polls would otherwise spam one httpx INFO line per
    # request and drown this app's own events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="nano_SLMs web UI (WEBUI_PRD U1-U5)")
    ap.add_argument("--lan", action="store_true",
                    help="listen on 0.0.0.0 (LAN) instead of localhost")
    ap.add_argument("--auth", metavar="USER:PASS",
                    help="require basic auth (recommended with --lan)")
    ap.add_argument("--webhook", default=None, metavar="URL",
                    help="POST run-finished/crashed notifications to this URL "
                         "(else $WEBUI_WEBHOOK_URL)")
    ap.add_argument("--port", type=int, default=CHAT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--ssl-certfile", default=None, metavar="PEM",
                    help="U11: optional HTTPS for LAN (with --ssl-keyfile)")
    ap.add_argument("--ssl-keyfile", default=None, metavar="PEM")
    a = ap.parse_args()
    if a.lan and not a.auth:  # U6: delete + launch must never be LAN-open
        sys.exit("--lan requires --auth USER:PASS (a LAN-exposed unauthenticated "
                 "UI could delete checkpoints or launch GPU jobs).")
    if a.webhook:
        WEBHOOK_URL = a.webhook
    auth = None
    if a.auth:
        user, _, pw = a.auth.partition(":")
        if not user or not pw:
            sys.exit("--auth expects USER:PASS")
        auth = (user, pw)
    # M5.4: a startup failure (port in use, ...) must surface as a
    # friendly, actionable message - never a raw traceback crash.
    try:
        demo.launch(server_name="0.0.0.0" if a.lan else "127.0.0.1",
                    server_port=a.port, auth=auth, inbrowser=not a.no_browser,
                    ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)
    except OSError as e:
        logger.exception("webui failed to start")
        # errno 48/98 = EADDRINUSE (macOS/Linux), 10048 = WSAEADDRINUSE
        # (Windows). Gradio wraps the same failure in its own OSError
        # ("Cannot find empty port ...") with errno None - match both.
        if e.errno in (48, 98, 10048) or "empty port" in str(e).lower():
            sys.exit(f"❌ webui could not start: port {a.port} is already in "
                     f"use ({e}). Stop the process holding it, or pick another "
                     f"port: python webui/app.py --port {a.port + 1}")
        sys.exit(f"❌ webui could not start: {e}")
    except Exception as e:  # noqa: BLE001 - a startup crash must stay readable
        logger.exception("webui failed to start")
        sys.exit(f"❌ webui could not start: {e!r}\n"
                 "The full traceback is logged above. Common causes: a port "
                 "already in use (try --port N) or a missing dependency.")
