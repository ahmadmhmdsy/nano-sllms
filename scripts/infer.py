"""Interactive / one-off prompting of a trained model (completion-style LM).

Run: .venv/Scripts/python scripts/infer.py --config configs/pilot.yaml [--ckpt DIR]
     [--prompt TEXT ...] [--max_new_tokens N] [--sample] [--temperature T]
     [--top_p P] [--top_k K] [--sft]

Greedy decode by default (deterministic); --sample adds temperature/top-p/top-k.
With no --prompt, enters a REPL: type a code prefix per line, Enter generates,
empty line or Ctrl+Z+Enter quits. Prompts are truncated so that prompt +
generation stays inside the training context window (ctx from the config).
The model is a plain completion LM (no chat template): give it a code prefix,
not a question.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="model dir; default runs/<name>/final")
    ap.add_argument("--sft", action="store_true",
                    help="wrap each prompt in the config's data.template "
                         "(instruct-style prompting for SFT models)")
    ap.add_argument("--prompt", action="append", default=[],
                    help="prompt text; repeatable; omit for the REPL")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of greedy decode")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seq_len = int(cfg["model"]["ctx"])
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / cfg["train"]["final_dir"]
    if not ckpt.is_dir():
        raise SystemExit(f"model dir not found: {ckpt} (train first)")

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[infer] model={ckpt} device={device} ctx={seq_len}", flush=True)

    # keep prompt + new tokens inside the trained context window
    room = max(seq_len - args.max_new_tokens, 16)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens,
                  "pad_token_id": tok.eos_token_id}
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k)
    else:
        gen_kwargs.update(do_sample=False)

    def ask(prompt: str) -> None:
        if args.sft:
            tmpl = (cfg.get("data") or {}).get("template")
            if not tmpl:
                raise SystemExit("[infer] --sft needs data.template in the config")
            prompt = tmpl.format(instruction=prompt)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=room)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"--- prompt: {prompt!r}\n{text}\n", flush=True)

    if args.prompt:
        for p in args.prompt:
            ask(p)
        return

    print("[infer] REPL: type a code prefix, Enter generates; empty line or "
          "Ctrl+Z+Enter quits", flush=True)
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line.strip():
            break
        ask(line)


if __name__ == "__main__":
    main()
