"""E-099 batched decode - the SAME three arms as diacritizer/scripts/e93_centered.py,
but with the forward passes batched instead of one sentence at a time.

WHY THIS EXISTS: e93_centered.py calls B.predict_bares(model, [(bare, None)], ...) and
V.predict_marks(...) ONE SENTENCE PER FORWARD PASS. Peak VRAM for the whole decode is
0.44 GiB of 6 GiB (recorded in runs/diac/e73_soup/e73_run.log), so the GPU is almost
entirely idle and a 15091-sentence corpus takes ~90 minutes. This script collects every
sliding window from every sentence, sorts them by length, and runs them in batches,
which is the same batching convention bench.predict_bares already uses for 'single'.

SEMANTICS ARE UNCHANGED. The windowing (V.window_starts), the majority vote
(V.sliding_window_predict's loop), the centrality tie-break (V._tie_break) and the
mark rendering (marks_for_label / is_arabic_base) are all the ORIGINAL functions,
imported, not reimplemented. The only difference is batch composition, which changes
fp16 padding numerics and flips near-tie argmaxes; that offset was measured in E-098 at
about 1e-4 absolute (~0.07 pct relative) by diffing e96_baseonly.py against
e93_centered.py, so it is 60x below the 2 pct significance bar.

GPU inference - never run beside a live train.py (single-GPU rule).

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e99_batch_decode.py \
      --ckpt runs/diac/e73_soup/c4b_uniform \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src fadel_test --window 256 --stride 128 --batch 256 \
      --out-dir runs/diac/e99_batch/fadel_test
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
from labels import marks_for_label  # noqa: E402
from passthrough import is_arabic_base  # noqa: E402
import bench as B  # noqa: E402
import e67_p6_vote as V  # noqa: E402


def batched_window_marks(model, windows, device, batch):
    """windows: list of (si, s, e, center, piece). Returns votes[si][pos] = [(mark, center)].

    One forward pass per batch of length-sorted windows, with a bool padding mask
    (True = KEEP) matching bench.py:84-88, so a padded row still sees its real chars.
    """
    metas = []
    for wi, (si, s, e, center, piece) in enumerate(windows):
        metas.append((wi, si, s, e, center, TK.encode(piece)))
    order = sorted(range(len(metas)), key=lambda i: len(metas[i][5]))
    out = [None] * len(metas)

    for k in range(0, len(order), batch):
        group = order[k:k + batch]
        T = max(1, max(len(metas[i][5]) for i in group))
        x = torch.zeros(len(group), T, dtype=torch.long, device=device)
        mask = torch.zeros(len(group), 1, 1, T, dtype=torch.bool, device=device)
        for r0, i in enumerate(group):
            L0 = len(metas[i][5])
            if L0:
                x[r0, :L0] = torch.tensor(metas[i][5], dtype=torch.long)
                mask[r0, 0, 0, :L0] = True
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, attn_mask=mask)["logits"]
        arg = logits.argmax(-1).tolist()
        for r0, i in enumerate(group):
            L0 = len(metas[i][5])
            out[i] = [marks_for_label(a) for a in arg[r0][:L0]]
    return metas, out


def assemble(pairs, metas, out, arm):
    """Rebuild per-sentence predictions from per-window marks, using the ORIGINAL
    tie-break / vote rules from e67_p6_vote.py."""
    n_pairs = len(pairs)
    votes = [[[] for _ in range(len(pairs[si][0]))] for si in range(n_pairs)]
    for i, (wi, si, s, e, center, enc) in enumerate(metas):
        for off, m in enumerate(out[i]):
            votes[si][s + off].append((m, center))
    preds = []
    for si, (bare, _) in enumerate(pairs):
        out_chars = []
        for pos, ch in enumerate(bare):
            out_chars.append(ch)
            if not is_arabic_base(ch):
                continue
            cands = votes[si][pos]
            if not cands:
                out_chars.append("")
                continue
            if arm == "centered":
                out_chars.append(V._tie_break(cands, pos))
            else:
                counts = Counter(m for m, _ in cands)
                top = max(counts.values())
                tied = [m for m, c in counts.items() if c == top]
                if len(tied) == 1:
                    out_chars.append(tied[0])
                else:
                    out_chars.append(V._tie_break([(m, c) for m, c in cands if m in tied], pos))
        preds.append("".join(out_chars))
    return preds


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
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="single,vote,centered")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    t0 = time.time()
    ck = Path(args.ckpt)
    state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(state_p, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    device = "cuda"
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print("[load] %.1fs params=%d" % (time.time() - t0,
          sum(p.numel() for p in model.parameters())), flush=True)

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    want = [a for a in args.arms.split(",") if a in ("single", "vote", "centered")]
    arms = {}
    if "single" in want:
        t1 = time.time()
        arms["single"] = B.predict_bares(model, [(b, None) for b, _ in pairs], ctx, device)
        print("[single] batched %.1fs" % (time.time() - t1), flush=True)
    if "vote" in want or "centered" in want:
        windows = []
        for si, (bare, _) in enumerate(pairs):
            n = len(bare)
            for s in V.window_starts(n, args.window, args.stride):
                e = min(s + args.window, n)
                windows.append((si, s, e, (s + e - 1) / 2.0, bare[s:e]))
                if e == n:
                    break
        print("[windows] %d windows over %d sentences" % (len(windows), len(pairs)), flush=True)
        t1 = time.time()
        metas, out = batched_window_marks(model, windows, device, args.batch)
        print("[windows] forward %.1fs" % (time.time() - t1), flush=True)
        for arm in ("vote", "centered"):
            if arm in want:
                t2 = time.time()
                arms[arm] = assemble(pairs, metas, out, arm)
                print("[%s] assemble %.1fs" % (arm, time.time() - t2), flush=True)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, preds in arms.items():
        (outdir / ("%s.%s.txt" % (args.src, name))).write_text(
            "\n".join(preds) + "\n", encoding="utf-8")
    (outdir / ("%s.ref.txt" % args.src)).write_text(
        "\n".join("\t".join(refs) for _, refs in pairs) + "\n", encoding="utf-8")
    print(json.dumps({"src": args.src, "pairs": len(pairs), "window": args.window,
                      "stride": args.stride, "batch": args.batch,
                      "arms": sorted(arms), "out_dir": str(outdir),
                      "total_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
