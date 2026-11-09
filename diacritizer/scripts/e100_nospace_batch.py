"""E-100 batched whitespace-mismatch decode - e94_nospace.py's 2x2 factorial
(single / nospace / centered / nospace_centered) with batched forward passes.

WHY THIS EXISTS. E-094 established the whitespace mismatch on v3ctx256 (0 space
tokens in the training stream, so the space embedding row is never trained).
That finding was re-measured here for the PUBLISHED model's own bed:

    data/diac/v3/tokens/train_ids.npy       space_id 77 -> 0 tokens in 200 rows
    data/diac/v3ctx256/tokens/train_ids.npy space_id 77 -> 0 tokens in 200 rows
    data/diac/v3full/tokens/train_ids.npy   space_id 77 -> 7823 / 51200 (15.28 pct)

So the published model (runs/diac/e63s_s4_12l160, tokens_phase v3) was trained
with NO spaces and IS subject to the mismatch, while E-097 arm B (v3full) was
trained WITH spaces, which is exactly why E-097 measured the nospace fix
INVERTING on bed B (+334 pct on abdou). A fair published-vs-bed-B comparison
therefore needs the published model decoded in BOTH conditions.

e94_nospace.py is one sentence per forward pass, so this is the same 34-84x
speedup as e99_batch_decode.py, sharing its batching convention (length-sorted
groups, bool padding mask True = KEEP) and its vote/centered assembly, which is
IMPORTED from e99_batch_decode rather than reimplemented.

GPU inference - never run beside a live train.py (single-GPU rule).

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e100_nospace_batch.py \
      --ckpt runs/diac/e63s_s4_12l160/final \
      --config runs/diac/e63s_s4_12l160/final/config.yaml \
      --src fadel_test --window 128 --stride 64 --batch 256 \
      --out-dir runs/diac/e100_published_nospace/fadel_test
"""
import argparse
import contextlib
import json
import sys
import time
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
from e99_batch_decode import assemble  # noqa: E402

WS = set(" \t\n\r\x0b\x0c")


def _strip(piece):
    return "".join(c for c in piece if c not in WS)


def _amp(device):
    """fp16 autocast on CUDA; a no-op context on CPU so this script can be
    smoke-tested without touching the GPU (single-GPU rule)."""
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def diacritize_piece_ws(piece, pred, strip_ws):
    """bench.diacritize_piece, optionally with the encoder's whitespace removed.

    'pred' is one label per ENCODED character. With strip_ws the whitespace
    positions are not encoded, so they consume no label and emit no mark; every
    other character still consumes exactly one, matching bench.py:55-63.
    """
    out = []
    k = 0
    for ch in piece:
        out.append(ch)
        if strip_ws and ch in WS:
            continue
        if is_arabic_base(ch):
            out.append(marks_for_label(pred[k]))
        k += 1
    return "".join(out)


def predict_bares_ws(model, pairs, ctx, device, batch, strip_ws):
    """bench.predict_bares (bench.py:65-104) with optional whitespace stripping."""
    metas, keep = [], []
    for si, (bare, _) in enumerate(pairs):
        pieces = B.chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            metas.append((si, pi, TK.encode(_strip(pc) if strip_ws else pc)))
    order = sorted(range(len(metas)), key=lambda i: len(metas[i][2]))
    out_pieces = [None] * len(metas)

    def flush(group):
        if not group:
            return
        T = max(1, max(len(metas[i][2]) for i in group))
        x = torch.zeros(len(group), T, dtype=torch.long, device=device)
        mask = torch.zeros(len(group), 1, 1, T, dtype=torch.bool, device=device)
        for r0, i in enumerate(group):
            L0 = len(metas[i][2])
            if L0:
                x[r0, :L0] = torch.tensor(metas[i][2], dtype=torch.long)
                mask[r0, 0, 0, :L0] = True
        with torch.no_grad(), _amp(device):
            logits = model(x, attn_mask=mask)["logits"]
        arg = logits.argmax(-1).tolist()
        for r0, i in enumerate(group):
            si, pi, _ = metas[i]
            out_pieces[i] = diacritize_piece_ws(keep[si][pi], arg[r0], strip_ws)

    for k in range(0, len(order), batch):
        flush(order[k:k + batch])
    out, ptr = [], 0
    for pieces in keep:
        n_p = len(pieces)
        out.append("".join(out_pieces[ptr + q] for q in range(n_p)))
        ptr += n_p
    return out


def batched_window_marks_ws(model, windows, device, batch, strip_ws):
    """e99_batch_decode.batched_window_marks with optional whitespace stripping.

    The returned per-window mark list stays aligned to the ORIGINAL piece
    positions ('' at every whitespace position), which is the invariant
    assemble() relies on when it indexes votes[si][s + off].
    """
    metas = []
    for wi, (si, s, e, center, piece) in enumerate(windows):
        metas.append((wi, si, s, e, center,
                      TK.encode(_strip(piece) if strip_ws else piece)))
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
        with torch.no_grad(), _amp(device):
            logits = model(x, attn_mask=mask)["logits"]
        arg = logits.argmax(-1).tolist()
        for r0, i in enumerate(group):
            L0 = len(metas[i][5])
            marks = [marks_for_label(a) for a in arg[r0][:L0]]
            if strip_ws:
                piece = windows[i][4]
                full, kk = [], 0
                for c in piece:
                    if c in WS:
                        full.append("")
                    else:
                        full.append(marks[kk])
                        kk += 1
                out[i] = full
            else:
                out[i] = marks
    return metas, out


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
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--arms", default="single,nospace,centered,nospace_centered")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    t0 = time.time()
    ck = Path(args.ckpt)
    state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    if args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(state_p, map_location=device, weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print("[load] %.1fs params=%d ctx=%d" % (time.time() - t0,
          sum(p.numel() for p in model.parameters()), ctx), flush=True)

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    want = [a for a in args.arms.split(",")
            if a in ("single", "nospace", "centered", "nospace_centered")]
    arms = {}
    if "single" in want or "nospace" in want:
        for name, sw in (("single", False), ("nospace", True)):
            if name in want:
                t1 = time.time()
                arms[name] = predict_bares_ws(model, pairs, ctx, device, args.batch, sw)
                print("[%s] batched %.1fs" % (name, time.time() - t1), flush=True)
    if "centered" in want or "nospace_centered" in want:
        windows = []
        for si, (bare, _) in enumerate(pairs):
            n = len(bare)
            for s in V.window_starts(n, args.window, args.stride):
                e = min(s + args.window, n)
                windows.append((si, s, e, (s + e - 1) / 2.0, bare[s:e]))
                if e == n:
                    break
        print("[windows] %d windows over %d sentences" % (len(windows), len(pairs)), flush=True)
        for name, sw, arm in (("centered", False, "centered"),
                              ("nospace_centered", True, "centered")):
            if name not in want:
                continue
            t1 = time.time()
            metas, out = batched_window_marks_ws(model, windows, device, args.batch, sw)
            print("[%s] forward %.1fs" % (name, time.time() - t1), flush=True)
            arms[name] = assemble(pairs, metas, out, arm)
            print("[%s] assemble %.1fs" % (name, time.time() - t1), flush=True)

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
