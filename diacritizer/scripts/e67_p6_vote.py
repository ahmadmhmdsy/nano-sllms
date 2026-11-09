"""E-067 P6 - sliding-window voting (inference-time review, fixes L4).

The model is bidirectional with ctx 128, so a single forward pass sees at most
128 chars. Sliding-window voting gives each Arabic-base position MULTIPLE
contexts: the sentence is split into overlapping windows, each window is
diacritized separately, and each position's mark is decided by majority vote
across the windows that cover it. Ties favor the window where the position is
most central (the "middle context" rule from Mubarak et al. N19-1248).

This is the cheapest L4 lever: no retraining, no new data, deterministic.

GPU inference - never run beside a live train.py (single-GPU rule).

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e67_p6_vote.py \
      --ckpt runs/diac/e63s_s7_18l64_adamw_ext40k/final \
      --config runs/diac/e63s_s7_18l64/optimized_config_adamw_ext40k.yaml \
      --src fadel_test --window 128 --stride 64
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
from labels import marks_for_label  # noqa: E402
from passthrough import is_arabic_base  # noqa: E402
import bench as B  # noqa: E402


def window_starts(n, window, stride):
    """Start offsets for overlapping windows covering [0, n)."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_marks(model, piece, device):
    """One forward pass on a window; return one mark string per char in piece.

    The tokenizer is char-level, so pred[wi] is the label for piece[wi].
    Non-base chars get "" (they are passthrough, never diacritized).
    """
    ids = TK.encode(piece)
    if not ids:
        return []
    x = torch.tensor([ids], dtype=torch.long, device=device)
    # Bool attn_mask convention: True = KEEP, False = BLOCK
    # (model.py:170-171 does masked_fill(~attn_mask, -inf)). An all-False mask
    # blocks EVERY position, so each query row becomes all -inf and the argmax
    # collapses to class 0 - the degenerate fatha-everywhere output measured on
    # 2026-10-06 (vote der_anyalt 0.7554 / macro_f1 0.0953 vs single 0.3712 /
    # 0.6422). Mirror the reference batcher, which marks the real positions
    # (bench.py:84-88), so a window with no padding is fully visible.
    mask = torch.ones(1, 1, 1, len(ids), dtype=torch.bool, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x, attn_mask=mask)["logits"]
    arg = logits.argmax(-1)[0].tolist()
    return [marks_for_label(a) for a in arg]


def _tie_break(cands, pos):
    """cands: list of (mark, window_center) already filtered to the tied marks.
    Pick the mark whose window center is closest to pos (the middle-context
    rule)."""
    return min(cands, key=lambda cm: (abs(pos - cm[1]), cm[0]))[0]


def sliding_window_predict(model, bare, ctx, device, window, stride):
    """Majority-vote diacritization of one bare sentence."""
    n = len(bare)
    # votes[i] = list of (mark, window_center) for position i
    votes = [[] for _ in range(n)]
    for s in window_starts(n, window, stride):
        e = min(s + window, n)
        piece = bare[s:e]
        marks = predict_marks(model, piece, device)
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
            counts = Counter(m for m, _ in cands)
            top = max(counts.values())
            tied = [m for m, c in counts.items() if c == top]
            if len(tied) == 1:
                out_chars.append(tied[0])
            else:
                tied_cands = [(m, c) for m, c in cands if m in tied]
                out_chars.append(_tie_break(tied_cands, i))
    return "".join(out_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True,
                    choices=["fadel_test", "sadeed25", "wikinews2014",
                             "wikinews2024", "abdou_test"])
    ap.add_argument("--out-single", required=True)
    ap.add_argument("--out-vote", required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ck = Path(args.ckpt)
    state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(state_p, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    device = "cuda"
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    single_preds = []
    vote_preds = []
    for bare, _ in pairs:
        single_preds.append(B.predict_bares(model, [(bare, None)], ctx, device)[0])
        vote_preds.append(sliding_window_predict(model, bare, ctx, device,
                                                args.window, args.stride))

    refs_all = ["\t".join(refs) for _, refs in pairs]
    for path, preds in ((args.out_single, single_preds),
                        (args.out_vote, vote_preds)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(preds) + "\n", encoding="utf-8")
        p.with_suffix(".ref.txt").write_text("\n".join(refs_all) + "\n",
                                            encoding="utf-8")
    print(json.dumps({
        "ckpt": str(ck), "src": args.src, "sentences": len(pairs),
        "params": n_params, "window": args.window, "stride": args.stride,
        "out_single": args.out_single, "out_vote": args.out_vote,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
