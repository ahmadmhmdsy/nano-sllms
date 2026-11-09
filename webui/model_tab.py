r"""Model tab (WEBUI_PRD.md §5 U12/U13): nested Architecture Explorer +
Training Simulator. THIN Gradio wiring — logic lives in explorer.py /
simulator.py / artifacts.py. Read-only + CPU-only: no torch model loads,
no VRAM. app.py injects its existing _ckpts / _full_curve helpers (no
duplication, no circular import). Single-user app (PRD §2), so the
simulator keeps one module-global replay state — same pattern as _MODEL.
"""
from __future__ import annotations

import logging
import threading
import time as _time

import gradio as gr

from artifacts import (ROOT, hf_config_dims, lora_note, phase_config,
                       run_config, safetensors_header, yaml_config_dims)
import explorer
import simulator

logger = logging.getLogger(__name__)

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


def _select_block(blocks, tech, evt: gr.SelectData):
    idx = evt.index if not isinstance(evt.index, (list, tuple)) else evt.index[0]
    bid = blocks[min(int(idx), len(blocks) - 1)]["id"]
    # (detail, bid): the bid output persists the selection into state_sel so
    # the Technical toggle re-renders the CLICKED block, not the first one
    # (final-review major 1: state_sel was only written by rebuild before).
    return _detail_md(blocks, bid, bool(tech)), bid


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

    tour_btn = gr.Button("Guided tour (walks every block)")

    def _tour(tech, blocks):
        # blocks arrives as an EVENT INPUT (live session graph); reading
        # state_blocks.value here would return the build-time init graph
        # (custom_example) after any source/config/run change (major 2).
        for b in (blocks or []):
            yield _detail_md(blocks, b["id"], bool(tech))
            _time.sleep(1.6)
    tour_btn.click(_tour, [tech, state_blocks], detail_md)

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

    pick_list.select(_select_block, [state_blocks, tech],
                     [detail_md, state_sel])
    tech.change(lambda t, blocks, sel: _detail_md(blocks, sel, bool(t)),
                [tech, state_blocks, state_sel], detail_md)
    trace_btn.click(explorer.tokenize_trace, [state_path, trace_in], trace_md)
    trace_in.submit(explorer.tokenize_trace, [state_path, trace_in], trace_md)


def render_model_tab(ckpts_fn, curve_fn, mode=None, v2_reader=None):
    """Called by app.py INSIDE the Blocks context (after the Settings tab).

    mode and v2_reader are optional. When app.py supplies them the tab gains a
    third sub-tab that reads the V2 declared model; without them the tab is
    exactly the V1 explorer plus simulator it always was, so this function
    stays backward compatible with its original two-argument call.
    """
    with gr.Tab("Model"):
        with gr.Tabs():
            with gr.Tab("Architecture Explorer"):
                _explorer_ui(ckpts_fn)
            with gr.Tab("Training Simulator"):
                _simulator_ui(curve_fn)
            if v2_reader is not None:
                with gr.Tab("V2 Declared Model"):
                    _v2_declared_ui(v2_reader, mode)


def _v2_declared_ui(v2_reader, mode):
    """The V2 model view: what the run declared, with its limits named.

    V2 publishes no weight files, so this cannot show a per-tensor inventory.
    It shows the validated architecture and says plainly that the tensor view
    is unavailable, rather than rendering an empty explorer that would read as
    a broken run.
    """
    from app import V2_MODE, v2_model_card

    gr.Markdown(
        "### V2 declared model\n"
        "Read from the artifacts the V2 engine published: the validated model "
        "identity and the published model config. Read-only, CPU-only, no "
        "weights loaded.")
    choices = v2_reader.run_choices()
    initial = choices[-1] if choices else ""
    picker = gr.Dropdown(choices=choices, label="V2 run", value=initial or None)
    # Rendered eagerly rather than through a load event: gr.load was removed
    # in Gradio 4 and this tab has no handle on the Blocks object.
    out = gr.Markdown(value=v2_model_card(initial) if initial else "")
    picker.change(v2_model_card, inputs=picker, outputs=out)
    if mode is not None:
        def _show(current_mode, label):
            if current_mode != V2_MODE:
                return ""
            return v2_model_card(label or (v2_reader.run_choices() or [""])[-1])

        mode.change(_show, inputs=[mode, picker], outputs=out)
    gr.Markdown(
        "_Scope: V2 has no safetensors header for this run, so the parameter "
        "breakdown of the Architecture Explorer has no V2 equivalent._")


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
        logger.info("simulator run loaded: %s (technique=%s)", name, tech)
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
        lambda t: gr.update(choices=simulator.TECHNIQUES[t],
                            value=(simulator.TECHNIQUES[t] or [None])[0]),
        [tech_radio], run_dd)
