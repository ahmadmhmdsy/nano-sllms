"""E-096 - encode EXACTLY the training input alphabet.

E-094 found that the model is fed untrained whitespace. Chasing the ROOT CAUSE
(not assumed - read from the pipeline) found the mismatch is far broader and is
BY DESIGN in the data pipeline:

  prepare_data.py:186-188
      bases = "".join(u.raw for u in units if u.is_target)
  passthrough.py:67-78
      is_target is True iff is_arabic_base(ch)
  passthrough.py:18-22
      is_arabic_base excludes tatweel; includes 0x0621-0x064A and 0x0671

So the TRAINING INPUT is exactly the string of is_arabic_base characters - a pure
letter stream with NO spaces, punctuation, digits, Latin, tatweel or <unk>. The
bincount over data/diac/v3ctx256/tokens/train_ids.npy confirms it independently:
133 of 171 vocab ids have count 0, and the 38 non-zero ids are pad + 37 letters.

bench.predict_bares (bench.py:75) instead encodes the RAW bare text, so inference
feeds untrained embeddings for 20-25 pct of all input tokens. E-094's "nospace"
arm removes only the whitespace share of that (~85 pct of the untrained tokens);
this script tests the PRINCIPLED arm - skip every character that is not an Arabic
base letter, i.e. reconstruct the training input distribution exactly.

Arms (all batched; no centered variant - it is CPU-bound at ~10x):
  single     : raw bare text, as deployed  (the mismatch)
  nospace    : whitespace skipped          (E-094's fix)
  base_only  : every non-is_arabic_base character skipped  (this arm)

No inline scoring: E-094's inline em.score_corpus used the in-memory pair list
while the registered scorers use splitlines(), and that shifted der_registered by
~0.4 pct on identical bytes. This script writes ONLY prediction/ref files; score
them with research/verdict_audit/tools/e95_gate_score.py, which is registered.

Usage:
  & .venv/Scripts/python.exe diacritizer/scripts/e96_baseonly.py \
      --ckpt runs/diac/e73_soup/c4b_uniform \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src abdou_test --limit 400 \
      --out-dir runs/diac/e96_baseonly
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
import passthrough as PT  # noqa: E402
from labels import marks_for_label  # noqa: E402
import tokenizer as TK  # noqa: E402
import bench as B  # noqa: E402

WS = set(" \t\n\r\x0b\x0c")


def skip_ws(ch):
    return ch in WS


def skip_nonbase(ch):
    return not PT.is_arabic_base(ch)


ARMS = {"single": None, "nospace": skip_ws, "base_only": skip_nonbase}


def diacritize_piece_pred(piece, pred, skip):
    """bench.diacritize_piece with a skip predicate. ti indexes the ENCODED
    sequence (skipped chars contribute no position), exactly as E-094 does."""
    out, ti = [], 0
    for ch in piece:
        if skip is not None and skip(ch):
            out.append(ch)
            continue
        out.append(ch)
        if PT.is_arabic_base(ch):
            out.append(marks_for_label(pred[ti]))
        ti += 1
    return "".join(out)


def predict_bares_pred(model, pairs, ctx, device, batch=64, skip=None):
    metas, keep = [], []
    for si, (bare, _) in enumerate(pairs):
        pieces = B.chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            enc = TK.encode("".join(c for c in pc if not skip(c)) if skip else pc)
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
            out_pieces[i] = diacritize_piece_pred(keep[si][pi], arg[r0], skip)

    for k in range(0, len(order), batch):
        flush(order[k:k + batch])
    out, ptr = [], 0
    for pieces in keep:
        out.append("".join(out_pieces[ptr + q] for q in range(len(pieces))))
        ptr += len(pieces)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="single,nospace,base_only")
    args = ap.parse_args()
    want = [a for a in args.arms.split(",") if a in ARMS]

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
    print("loaded pairs:", len(pairs), flush=True)

    arms = {}
    for name in want:
        preds = predict_bares_pred(model, pairs, ctx, device, skip=ARMS[name])
        arms[name] = preds
        print("  arm done:", name, flush=True)

    refs_all = ["\t".join(refs) for _, refs in pairs]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, preds in arms.items():
        (outdir / ("%s.%s.txt" % (args.src, name))).write_text(
            "\n".join(preds) + "\n", encoding="utf-8")
    (outdir / ("%s.ref.txt" % args.src)).write_text(
        "\n".join(refs_all) + "\n", encoding="utf-8")

    # non-degeneracy: every arm must strip back to the same bare
    base_ok = True
    for name, preds in arms.items():
        for p, (bare, _) in zip(preds, pairs):
            if B.strip_marks(p).replace(" ", "") != bare.replace(" ", ""):
                base_ok = False
                print("  NON-DEGENERATE FAIL:", name, flush=True)
                break
    report = {"src": args.src, "pairs": len(pairs), "arms": want,
              "non_degenerate": base_ok,
              "note": "prediction/ref files only; score with e95_gate_score.py"}
    (outdir / "e96_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
