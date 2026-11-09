"""E-094 - the WHITESPACE MISMATCH: the model was trained with NO space tokens.

FINDING THAT MOTIVATES THIS (verified, not assumed):
  data/diac/v3ctx256/tokens/train_ids.npy  ->  0 space tokens in 200000 rows
                                                (38,086,115 non-pad tokens)
  data/diac/v3ctx256/tokens/val_ids.npy    ->  0 space tokens in 5000 rows
  tokenizer.py:30 puts " " in the vocab (id 77, vocab size 171), so the space
  EMBEDDING ROW EXISTS but was never updated by any gradient.

bench.predict_bares (bench.py:75) encodes the raw bare text, which CONTAINS
spaces, so every inference forward pass feeds an untrained noise vector at every
word boundary. This is a train/inference distribution mismatch on the single most
linguistically meaningful character in the input.

DESIGN - a 2x2 factorial from ONE loaded model, so the whitespace effect is
isolated from the chunking effect (E-092/E-093 established the chunking effect):
                 spaces fed (as deployed)     spaces stripped (matches training)
  hard chunk     single                       nospace
  centered       centered                     nospace_centered

Stripping whitespace changes ONLY what the encoder sees: the pieces are chunked
identically from the ORIGINAL bare string, and whitespace is reinserted at its
original positions on output, so both sides are byte-aligned.

GPU inference - never run beside a live train.py (single-GPU rule).

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e94_nospace.py \
      --ckpt runs/diac/e73_soup/c4b_uniform \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src abdou_test --limit 400 --window 256 --stride 128 \
      --out-dir runs/diac/e94_nospace
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from model import build_from_config  # noqa: E402
from passthrough import is_arabic_base  # noqa: E402
from labels import marks_for_label  # noqa: E402
import tokenizer as TK  # noqa: E402
import bench as B  # noqa: E402
import eval_metrics as em  # noqa: E402
import e67_p6_vote as V  # noqa: E402

WS = set(" \t\n\r\x0b\x0c")


def predict_marks_ws(model, piece, device, skip_ws):
    """predict_marks with optional whitespace stripping. Marks stay aligned to
    the ORIGINAL piece positions (whitespace -> ""), so callers index as usual."""
    kept = "".join(c for c in piece if c not in WS) if skip_ws else piece
    ids = TK.encode(kept)
    if not ids:
        return [""] * len(piece)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones(1, 1, 1, len(ids), dtype=torch.bool, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x, attn_mask=mask)["logits"]
    arg = logits.argmax(-1)[0].tolist()
    marks = [marks_for_label(a) for a in arg]
    if not skip_ws:
        return marks
    out, k = [], 0
    for c in piece:
        if c in WS:
            out.append("")
        else:
            out.append(marks[k])
            k += 1
    return out


def diacritize_piece_ws(piece, pred, skip_ws):
    out, ti = [], 0
    for ch in piece:
        if skip_ws and ch in WS:
            out.append(ch)
            continue
        out.append(ch)
        if is_arabic_base(ch):
            out.append(marks_for_label(pred[ti]))
        ti += 1
    return "".join(out)


def predict_bares_ws(model, pairs, ctx, device, batch=64, skip_ws=False):
    """bench.predict_bares, verbatim in structure, with whitespace optional."""
    metas, keep = [], []
    for si, (bare, _) in enumerate(pairs):
        pieces = B.chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            enc = TK.encode("".join(c for c in pc if c not in WS) if skip_ws else pc)
            metas.append((si, pi, enc))
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
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, attn_mask=mask)["logits"]
        arg = logits.argmax(-1).tolist()
        for r0, i in enumerate(group):
            si, pi, _ = metas[i]
            out_pieces[i] = diacritize_piece_ws(keep[si][pi], arg[r0], skip_ws)

    for k in range(0, len(order), batch):
        flush(order[k:k + batch])
    out, ptr = [], 0
    for pieces in keep:
        out.append("".join(out_pieces[ptr + q] for q in range(len(pieces))))
        ptr += len(pieces)
    return out


def centered_window_predict_ws(model, bare, ctx, device, window, stride, skip_ws):
    """E-093's centered decode (NO vote), with whitespace optional."""
    n = len(bare)
    votes = [[] for _ in range(n)]
    for s in V.window_starts(n, window, stride):
        e = min(s + window, n)
        marks = predict_marks_ws(model, bare[s:e], device, skip_ws)
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
            out_chars.append(V._tie_break(cands, i) if cands else "")
    return "".join(out_chars)


def mcnemar(pa, pb, ref):
    """Paired word-level McNemar between two arms over the same refs."""
    disc = a_only = b_only = 0
    for x, y, r in zip(pa, pb, ref):
        bx = em.word_outcomes(x, r)
        by = em.word_outcomes(y, r)
        if len(bx) != len(by):
            continue
        for u, v in zip(bx, by):
            if u != v:
                disc += 1
                if u and not v:
                    a_only += 1
                else:
                    b_only += 1
    # exact binomial two-sided p on the discordant split
    from math import comb
    n = a_only + b_only
    if n == 0:
        return {"discordant": 0, "p": 1.0}
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return {"discordant": disc, "a_only": a_only, "b_only": b_only,
            "p": round(min(1.0, p), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="single,nospace,centered,nospace_centered",
                    help="comma-separated subset; the batched arms (single,nospace) "
                         "are ~10x faster than the per-window centered ones")
    args = ap.parse_args()
    want = set(args.arms.split(","))

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

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    arms = {k: [] for k in want}
    for i, (bare, _) in enumerate(pairs):
        if "single" in want:
            arms["single"].append(
                predict_bares_ws(model, [(bare, None)], ctx, device)[0])
        if "nospace" in want:
            arms["nospace"].append(
                predict_bares_ws(model, [(bare, None)], ctx, device, skip_ws=True)[0])
        if "centered" in want:
            arms["centered"].append(
                centered_window_predict_ws(model, bare, ctx, device,
                                           args.window, args.stride, False))
        if "nospace_centered" in want:
            arms["nospace_centered"].append(
                centered_window_predict_ws(model, bare, ctx, device,
                                           args.window, args.stride, True))
        if (i + 1) % 100 == 0:
            print("  pairs done:", i + 1, flush=True)

    refs_all = ["\t".join(refs) for _, refs in pairs]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, preds in arms.items():
        (outdir / ("%s.%s.txt" % (args.src, name))).write_text(
            "\n".join(preds) + "\n", encoding="utf-8")
    (outdir / ("%s.ref.txt" % args.src)).write_text(
        "\n".join(refs_all) + "\n", encoding="utf-8")

    report = {"src": args.src, "pairs": len(pairs), "window": args.window,
              "stride": args.stride, "arms": {}}
    for name, preds in arms.items():
        sm = em.summarize(em.score_corpus(preds, refs_all))
        report["arms"][name] = {
            "der_registered": sm.get("der_registered"),
            "der_anyalt": sm.get("der_anyalt"),
            "der_canonical": sm.get("der_canonical"),
            "macro_f1": sm.get("macro_f1"),
        }
    # non-degeneracy: every arm must strip back to the same bare
    base_ok = True
    for name, preds in arms.items():
        for p, (bare, _) in zip(preds, pairs):
            if B.strip_marks(p).replace(" ", "") != bare.replace(" ", ""):
                base_ok = False
                break
    report["non_degenerate"] = base_ok
    report["mcnemar"] = {}
    for a, b in (("single", "nospace"), ("centered", "nospace_centered"),
                 ("single", "centered")):
        if a in arms and b in arms:
            report["mcnemar"][f"{a}_vs_{b}"] = mcnemar(arms[a], arms[b], refs_all)
    (outdir / "e94_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
