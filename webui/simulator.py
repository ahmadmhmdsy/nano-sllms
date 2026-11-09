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
    # distill summaries record student_params_m/teacher_params_m, no params_m
    params_m = rd.summary.get("params_m") or rd.summary.get("student_params_m")
    if not params_m:
        return None
    per_b = {"Pretrain": 19.0, "SFT": 19.0, "LoRA": 3.7, "KD": 19.0}[technique]
    if str(rd.cfg.get("train", {}).get("optim", "")).endswith("8bit") \
            and technique in ("Pretrain", "SFT"):
        per_b = 6.0
    est = params_m * per_b / 1024.0
    if technique == "KD":
        # frozen teacher: summaries carry teacher_params_m (kd-t2p -> target
        # 226.5M, kd-s -> pilot 100.7M); fallback assumes the target teacher.
        # Measured kd-t2p-kd peak 7.01 GB vs this linear 6.07 — the residual
        # is the full-vocab logits/optimizer transient (UI labels it 'est').
        est += (rd.summary.get("teacher_params_m") or 226.5) * 19.0 / 1024.0
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
             f" - params: **{s.get('params_m') or s.get('student_params_m')}M**"]
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
