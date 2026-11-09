"""Interactive diacritizer for the D-line GOLD model.

Usage (arbitrary text, no benchmarks involved):
  & .venv/Scripts/python.exe -X utf8 diacritizer/scripts/diacritize.py --text <Arabic text>
  (or --file <path>, or pipe lines via stdin)
Defaults load the GOLD: runs/diac/stage2b2500/final/model.pt + config.yaml.
To test a gate-probe snapshot instead: point --ckpt/--config at
  .../gate_probe/best_gate_weights.pt + .../gate_probe/probe_config.yaml.
Prints diacritized text (1 line in -> 1 line out). GPU fp16; never co-run with
a live training job (single-GPU rule).
"""
import argparse
import sys
import unicodedata
from pathlib import Path

import torch
import yaml


def main():
    REPO = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO / "diacritizer" / "src"))
    from model import build_from_config  # noqa: E402
    import tokenizer as TK  # noqa: E402
    from labels import marks_for_label  # noqa: E402
    from passthrough import is_arabic_base  # noqa: E402
    import bench as B  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"runs\diac\stage2b2500\final\model.pt")
    ap.add_argument("--config", default=r"runs\diac\stage2b2500\final\config.yaml")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    ck = Path(args.ckpt)
    snap = torch.load(ck, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    device = "cuda"
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    sd = snap["model"] if "model" in snap else snap
    model.load_state_dict(sd)
    model.eval()
    print(f"[loaded] {ck.name} step={snap.get('step', '?')} params={sum(p.numel() for p in model.parameters())}", file=sys.stderr)

    # REPL whenever no explicit input source is given. Do NOT trust
    # sys.stdin.isatty() - it reports False in some double-click console
    # sessions (that bug made the bat look like it did nothing).
    repl = not (args.text or args.file)
    if repl:
        print("Type/paste Arabic (one sentence per line). 'q' or Ctrl+Z+Enter quits.", file=sys.stderr)
    for line in iter_lines(args, repl):
        bare = unicodedata.normalize("NFC", B.strip_marks(line))
        if not bare.strip():
            continue
        if line.strip().lower() == "q":
            break
        print(B.predict_bare(model, bare, ctx, device))
        if repl and sys.stdin.isatty():
            print("", file=sys.stderr)  # fresh line after each answer


def iter_lines(args, repl):
    if args.text:
        yield from args.text.splitlines()
    elif args.file:
        yield from Path(args.file).read_text(encoding="utf-8").splitlines()
    elif repl:
        # input() reads the console prompt AND works piped (piped line first,
        # then EOFError ends the stream) - so keep one code path for both.
        while True:
            try:
                line = input("arabic> ")
            except EOFError:
                break
            yield line
    else:
        yield from sys.stdin.read().splitlines()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("[diacritize.py crashed - read the traceback above]", file=sys.stderr)
        input("press Enter to close...")
