"""Post-run evaluation: val loss + perplexity + generation samples.

Run: .venv/Scripts/python scripts/eval.py --config configs/smoke.yaml [--ckpt DIR]

With an SFT config (eval.instructions_file + data.template present, e.g.
configs/sft_t1.yaml) it additionally runs the C12 Tier-1 instruction eval:
greedy + sampled generations over held-out instructions and the ast.parse
pass-rate (plan §3.3 success criterion). The forgetting guard is the same
script twice on the same config: --ckpt runs/target/final (before SFT) vs
--ckpt runs/sft_t1/final (after); the CSN val-loss delta is the regression gate.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="model dir; default runs/<name>/final")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="overrides eval.max_new_tokens in the config (default 64)")
    # --- Track A context-probe knobs (TASKS row 29; CLI > config > off).
    # All default OFF: without them this script behaves exactly as before.
    ap.add_argument("--ctx", type=int, default=None,
                    help="re-pack CSN val shards at this context length (eval.ctx_override)")
    ap.add_argument("--rope-scaling", default=None,
                    help="rope knob: dynamic-ntk swaps RoPE to HF dynamic rope (eval.rope_scaling)")
    ap.add_argument("--rope-factor", type=float, default=None,
                    help="dynamic-NTK factor (default 1.0 = Qwen dynamic-NTK; eval.rope_factor)")
    ap.add_argument("--stream-window", type=int, default=None,
                    help="StreamingLLM windowed attention W (eval.stream_window)")
    ap.add_argument("--sink-tokens", type=int, default=None,
                    help="attention-sink tokens kept beside the window (default 4; eval.sink_tokens)")
    ap.add_argument("--stream-positions", default=None, choices=["absolute", "remapped"],
                    help="absolute = mask only; remapped = cache-relative pos_shift ids (eval.stream_positions)")
    ap.add_argument("--batch", type=int, default=None,
                    help="val batch size (default 8 legacy / auto 8|4|1 by ctx when probing)")
    ap.add_argument("--quick", type=int, default=None,
                    help="stop after N val batches (VRAM/pace probe only - partial loss)")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip generation samples + instruction eval (pure val-loss probes)")
    ap.add_argument("--report-out", default=None,
                    help="write eval_report.json here instead of <ckpt>/eval_report.json")
    ap.add_argument("--label", default=None, help="probe label stored in the report")
    args = ap.parse_args()

    import torch
    import yaml
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data import PackedDataset
    from src.model import (apply_rope_scaling, build_streaming_sink_mask,
                           streaming_position_ids)

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seq_len = int(cfg["model"]["ctx"])
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / cfg["train"]["final_dir"]
    if not ckpt.is_dir():
        raise SystemExit(f"model dir not found: {ckpt} (train first)")

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[eval] model={ckpt} device={device}", flush=True)

    # --- Track A knobs (TASKS row 29): CLI > config > off. With all off the
    # eval path below is byte-identical to the pre-knob script.
    ev = cfg.get("eval", {}) or {}
    eval_ctx = args.ctx or ev.get("ctx_override") or seq_len
    rope_type = args.rope_scaling or ev.get("rope_scaling")
    rope_factor = (args.rope_factor if args.rope_factor is not None
                   else float(ev.get("rope_factor", 1.0)))
    stream_window = (args.stream_window if args.stream_window is not None
                     else ev.get("stream_window"))
    sink_tokens = (args.sink_tokens if args.sink_tokens is not None
                   else int(ev.get("sink_tokens", 4)))
    stream_positions = (args.stream_positions if args.stream_positions is not None
                        else ev.get("stream_positions", "absolute"))
    probing = any([rope_type, stream_window is not None, eval_ctx != seq_len])
    if rope_type:
        apply_rope_scaling(model, rope_type, rope_factor)
        print(f"[eval] rope_scaling={rope_type} factor={rope_factor} (NTK fires only beyond max_position_embeddings={model.config.max_position_embeddings})", flush=True)
    if eval_ctx > seq_len and stream_window is None and not rope_type:
        print(f"[eval] WARNING: ctx={eval_ctx} > trained ctx={seq_len} with no rope/window knob - raw extrapolation probe", flush=True)
    if args.batch is not None:
        batch_size = args.batch
    elif ev.get("probe_batch"):
        batch_size = int(ev["probe_batch"])
    elif probing:
        batch_size = 8 if eval_ctx <= 1024 else (4 if eval_ctx <= 2048 else 1)
    else:
        batch_size = 8
    stream_mask = stream_pos = None
    if stream_window is not None:
        stream_mask = build_streaming_sink_mask(eval_ctx, int(stream_window),
                                                sink_tokens, device)
        if stream_positions == "remapped":
            stream_pos = streaming_position_ids(eval_ctx, int(stream_window),
                                                sink_tokens, device=device)
        print(f"[eval] streaming window={stream_window} sink={sink_tokens} positions={stream_positions}", flush=True)
    print(f"[eval] val ctx={eval_ctx} batch={batch_size}", flush=True)

    # CLI flag > config > legacy default (old configs all set 64, so behavior
    # for smoke/pilot/target is unchanged).
    gen_len = (args.max_new_tokens if args.max_new_tokens is not None
               else int(cfg["eval"].get("max_new_tokens", 64)))

    val_ds = PackedDataset((ROOT / cfg["data"]["tokens_dir"]).glob("val_*.bin"), eval_ctx)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            fwd = {"input_ids": ids, "labels": labels}
            if stream_mask is not None:
                fwd["attention_mask"] = stream_mask
            if stream_pos is not None:
                fwd["position_ids"] = stream_pos[: ids.shape[0]]
            out = model(**fwd)
            n = labels.numel()
            total_nll += float(out.loss) * n
            total_tok += n
            if args.quick is not None and i + 1 >= args.quick:
                print(f"[eval] --quick {args.quick}: partial probe only, stopped after {i + 1} batches", flush=True)
                break
    avg_nll = total_nll / max(total_tok, 1)
    ppl_raw = math.exp(avg_nll) if avg_nll < 20 else float('inf')
    ppl = math.exp(min(avg_nll, 20))
    print(f"[eval] val_loss={avg_nll:.4f} perplexity={ppl:.2f} (raw={ppl_raw})",
          flush=True)

    samples = {}
    for prompt in (cfg.get("eval", {}).get("prompts") or []):
        if args.skip_gen:
            break
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=gen_len,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        samples[prompt] = tok.decode(out[0], skip_special_tokens=True)
        print(f"--- prompt: {prompt!r}\n{samples[prompt]}\n", flush=True)

    # C12 Tier-1 instruction eval (only for SFT configs that declare it)
    instruction_eval = None
    instr_rel = cfg["eval"].get("instructions_file")
    template = cfg["data"].get("template")
    if not args.skip_gen and instr_rel and template:
        from sft_data import extract_code, parses_as_python
        instr_path = ROOT / instr_rel
        if not instr_path.exists():
            print(f"[eval] instructions file missing, skipping instruction eval: "
                  f"{instr_path}", flush=True)
        else:
            n_max = int(cfg["eval"].get("n_instructions", 50))
            temperature = float(cfg["eval"].get("sample_temperature", 0.8))
            instructions = []
            with instr_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        instructions.append(json.loads(line)["instruction"])
            instructions = instructions[:n_max]
            print(f"[eval] instruction eval: {len(instructions)} held-out "
                  f"instructions, greedy + temp={temperature}, "
                  f"max_new_tokens={gen_len}", flush=True)
            room = max(seq_len - gen_len, 16)
            greedy_ok, sample_ok, fails = [], [], []
            for i, instr in enumerate(instructions):
                prompt = template.format(instruction=instr)
                inputs = tok(prompt, return_tensors="pt", truncation=True,
                             max_length=room).to(model.device)
                base_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    g = model.generate(**inputs, max_new_tokens=gen_len,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    s = model.generate(**inputs, max_new_tokens=gen_len,
                                       do_sample=True, temperature=temperature,
                                       top_p=0.95,
                                       pad_token_id=tok.eos_token_id)
                g_text = tok.decode(g[0][base_len:], skip_special_tokens=True)
                s_text = tok.decode(s[0][base_len:], skip_special_tokens=True)
                g_ok = parses_as_python(extract_code(g_text))
                s_ok = parses_as_python(extract_code(s_text))
                greedy_ok.append(g_ok)
                sample_ok.append(s_ok)
                if not (g_ok and s_ok):
                    fails.append({"instruction": instr,
                                  "greedy": g_text, "sampled": s_text})
                if (i + 1) % 10 == 0:
                    print(f"[eval] instructions {i + 1}/{len(instructions)}: "
                          f"greedy {sum(greedy_ok)}/{i + 1}, "
                          f"sampled {sum(sample_ok)}/{i + 1}", flush=True)
            instruction_eval = {
                "n": len(instructions),
                "greedy_ast_pass_rate": sum(greedy_ok) / max(len(instructions), 1),
                "sampled_ast_pass_rate": sum(sample_ok) / max(len(instructions), 1),
                "temperature": temperature,
                "failed_examples": fails[:5],
            }
            print(f"[eval] ast.parse pass-rate: greedy "
                  f"{instruction_eval['greedy_ast_pass_rate']:.2f}, sampled "
                  f"{instruction_eval['sampled_ast_pass_rate']:.2f}", flush=True)

    report = {"ckpt": str(ckpt), "val_loss": avg_nll, "perplexity": ppl,
              "samples": samples}
    if instruction_eval is not None:
        report["instruction_eval"] = instruction_eval
    if probing or args.quick is not None:
        report["probe"] = {
            "label": args.label,
            "ctx": eval_ctx, "trained_ctx": seq_len,
            "rope_scaling": rope_type, "rope_factor": rope_factor,
            "stream_window": stream_window, "sink_tokens": sink_tokens,
            "stream_positions": (stream_positions
                                 if stream_window is not None else None),
            "batch": batch_size,
            "quick_batches": args.quick,
            "val_tokens": total_tok,
        }
    # Default destination unchanged (<ckpt>/eval_report.json); Track A probes
    # pass --report-out so they never overwrite the historical report.
    out_path = Path(args.report_out) if args.report_out else Path(ckpt) / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[eval] report written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
