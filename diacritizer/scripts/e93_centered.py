"""E-093 - is the sliding-window VOTE doing anything beyond picking the most
central covering window?

E-092's amendment showed 91.8 pct of the vote's error reduction is words within
8 chars of the NON-OVERLAPPING 256-char chunk boundary that bench.predict_bares
creates, and that the repair is done by the middle-context tie-break. That leaves
a direct question: the vote does MAJORITY first and centering only on TIES, so
does the majority step contribute anything at all?

This driver emits THREE arms from ONE loaded model per run (E-075's control
property), differing ONLY in how a position's mark is chosen:
  single    hard-chunked non-overlapping 256-char pieces (bench.predict_bares)
  vote      majority vote, ties -> most central window      (E-075 mechanism)
  centered  NO VOTE: every position takes the mark from the window in which it
            is most central - the same _tie_break rule applied to ALL candidates

predict_marks / window_starts / _tie_break are IMPORTED from e67_p6_vote.py, not
reimplemented, so the comparison cannot drift from the deployed mechanism.

GPU inference - never run beside a live train.py (single-GPU rule).

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e93_centered.py \
      --ckpt runs/diac/e73_soup/c4b_uniform \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src abdou_test --limit 400 --window 256 --stride 128 --out-dir runs/diac/e93_centered
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from model import build_from_config  # noqa: E402
from passthrough import is_arabic_base  # noqa: E402
import bench as B  # noqa: E402
import e67_p6_vote as V  # noqa: E402


def centered_window_predict(model, bare, ctx, device, window, stride):
    """Overlapping decode with NO vote.

    Each position takes the mark from the window in which it is most central.
    This is exactly the vote's tie-break rule, applied to ALL candidates instead
    of only the tied ones, so the arm isolates the majority step.
    """
    n = len(bare)
    votes = [[] for _ in range(n)]
    for s in V.window_starts(n, window, stride):
        e = min(s + window, n)
        marks = V.predict_marks(model, bare[s:e], device)
        center = (s + e - 1) / 2.0
        for wi, m in enumerate(marks):
            votes[s + wi].append((m, center))
        if e == n:
            break
    out_chars = []
    for i, ch in enumerate(bare):
        out_chars.append(ch)
        if is_arabic_base(ch):
            cands = votes[i]
            if not cands:
                out_chars.append("")
                continue
            out_chars.append(V._tie_break(cands, i))
    return "".join(out_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True,
                    choices=["fadel_test", "sadeed25", "wikinews2014",
                             "wikinews2024", "abdou_test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ck = Path(args.ckpt)
    state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(state_p, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    device = "cuda"
    model = build_from_config(cfg, vocab_size=V.TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    arms = {"single": [], "vote": [], "centered": []}
    for bare, _ in pairs:
        arms["single"].append(B.predict_bares(model, [(bare, None)], ctx, device)[0])
        arms["vote"].append(V.sliding_window_predict(model, bare, ctx, device,
                                                     args.window, args.stride))
        arms["centered"].append(centered_window_predict(model, bare, ctx, device,
                                                        args.window, args.stride))

    refs_all = ["\t".join(refs) for _, refs in pairs]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, preds in arms.items():
        (outdir / ("%s.%s.txt" % (args.src, name))).write_text(
            "\n".join(preds) + "\n", encoding="utf-8")
    (outdir / ("%s.ref.txt" % args.src)).write_text(
        "\n".join(refs_all) + "\n", encoding="utf-8")
    print(json.dumps({"src": args.src, "pairs": len(pairs), "window": args.window,
                      "stride": args.stride, "arms": sorted(arms),
                      "out_dir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
