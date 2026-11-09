"""R54 gate bench: run a trained diacritizer checkpoint over external
benchmark sources and emit predicted diacritized text (1 line = 1 sentence)
for diacritizer/scripts/eval.py compare.

Sources (--src): fadel_test | sadeed25 | wikinews2024 | wikinews2014 | abdou_test

Prediction path (routing D6): chunk the bare text into <=ctx pieces, forward
the model, argmax at Arabic-base positions only, marks_for_label decode, mark
insertion; every non-base byte is passthrough-copied so reassembly is
byte-exact.

2026-09-23 OPTIMIZATION (v2):
  Cross-sentence global batcher: ALL pieces of the whole gate get
  length-sorted and forwarded in bucket batches with BOOLEAN key-padding
  masks (fp16 logits bit-identical to the bs1 forward at real positions;
  a float mask would act as additive bias and is forbidden). batch=1 ==
  legacy. VRAM peak reported in the final JSON line.
GPU required (fp16 autocast); never run beside a live train.py.
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
from labels import marks_for_label  # noqa: E402
from passthrough import is_arabic_base  # noqa: E402
import yaml  # noqa: E402

PREPARED = REPO / "data" / "diac" / "prepared"
RAW = REPO / "data" / "diac" / "raw"

MARK_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670), (0x0653, 0x065F),
               (0x06D6, 0x06ED))

def strip_marks(text):
    """Bare text: drop every combining mark; NFC-normalize."""
    out = []
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in MARK_RANGES):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

def chunks_of(bare, size):
    return [(bare[o:o + size]) for o in range(0, len(bare), size)]

def diacritize_piece(piece, pred):
    out = []
    ti = 0
    for ch in piece:
        out.append(ch)
        if is_arabic_base(ch):
            out.append(marks_for_label(pred[ti]))
        ti += 1
    return "".join(out)

def predict_bares(model, pairs, ctx, device, batch=64):
    """Cross-sentence global batcher (v2): all pieces length-sorted and
    bucket-batched with bool padding masks; per-sentence reassembly keeps the
    original byte-exact order. batch=1 == legacy bs1 numbers."""
    metas = []
    keep = []
    for si, (bare, _) in enumerate(pairs):
        pieces = chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            metas.append((si, pi, TK.encode(pc)))
    order = sorted(range(len(metas)), key=lambda i: len(metas[i][2]))
    out_pieces = [None] * len(metas)

    def flush(group):
        if not group:
            return
        T = max(len(metas[i][2]) for i in group)
        x = torch.zeros(len(group), T, dtype=torch.long, device=device)
        mask = torch.zeros(len(group), 1, 1, T, dtype=torch.bool, device=device)
        for r0, i in enumerate(group):
            L0 = len(metas[i][2])
            x[r0, :L0] = torch.tensor(metas[i][2], dtype=torch.long)
            mask[r0, 0, 0, :L0] = True
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, attn_mask=mask)["logits"]
        arg = logits.argmax(-1).tolist()
        for r0, i in enumerate(group):
            si, pi, _ = metas[i]
            out_pieces[i] = diacritize_piece(keep[si][pi], arg[r0])

    for k in range(0, len(order), batch):
        flush(order[k:k + batch])
    out = []
    ptr = 0
    for pieces in keep:
        n_p = len(pieces)
        out.append("".join(out_pieces[ptr + q] for q in range(n_p)))
        ptr += n_p
    return out

def load_pairs(src, ctx):
    """Yield (bare_input, [ref, ...]) sentence tuples for a source name."""
    def ok(text):
        return bool(strip_marks(text).strip())
    if src == "fadel_test":
        # EXTERNAL gate policy: raw file, never the prepared val split.
        p = RAW / "fadel" / "test.txt"
        for l in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
            ref = unicodedata.normalize("NFC", l.strip())
            bare = strip_marks(l)
            if ok(bare):
                yield bare, [ref]
        return
    if src == "abdou_test":
        # HELD-OUT control: abdou test split, NEVER training (E-17).
        import pandas as pd
        df = pd.read_parquet(RAW / "abdou_tashkeel" / "data" /
                             "test-00000-of-00001.parquet")
        for r in df.itertuples():
            gold = unicodedata.normalize("NFC", str(r.vocalized))
            if not ok(gold):
                continue
            yield strip_marks(str(r.non_vocalized)) or strip_marks(gold), [gold]
        return
    if src == "sadeed25":
        import pandas as pd
        df = pd.read_parquet(RAW / "sadeed_25" / "sadeed25.parquet")
        for r in df.itertuples():
            gold = unicodedata.normalize("NFC", str(r.output))
            if not ok(gold):
                continue
            g_lines = gold.split("\n")
            b_lines = str(r.input).split("\n")
            if len(g_lines) > 1 and len(g_lines) == len(b_lines):
                for g_line, b_line in zip(g_lines, b_lines):
                    bare_l = strip_marks(b_line)
                    if ok(bare_l):
                        yield bare_l, [g_line]
            elif ok(gold):
                yield strip_marks(str(r.input)) or strip_marks(gold), [gold]
        return
    p = RAW / "wikinews" / (f"wikinews{src[-4:]}_multi_ref.diac")
    for l in p.read_text(encoding="utf-8").splitlines():
        sx = l.strip()
        if src == "wikinews2014" and sx.startswith("#"):
            continue
        ref = unicodedata.normalize("NFC", sx)
        bare = strip_marks(sx)
        if ok(bare):
            yield bare, [ref]

def main():
    import sys as _s
    _s.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir OR final dir")
    ap.add_argument("--src", required=True,
                    choices=["fadel_test", "sadeed25", "wikinews2024",
                             "wikinews2014", "abdou_test"])
    ap.add_argument("--out", required=True, help="prediction file (1 line/sent)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="pieces per GPU forward (cross-sentence bucketing; "
                         "1 = legacy bs1 path)")
    ap.add_argument("--config", default=None,
                    help="explicit yaml for the model shape (REQUIRED for "
                         "any final whose arch differs from diac_pilot.yaml)")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(state_p, map_location="cuda", weights_only=False)
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    else:
        cfg = json.loads(state.get("config", "{}")) or yaml.safe_load(
            (REPO / "configs" / "diac_pilot128.yaml").read_text(encoding="utf-8"))
    device = "cuda"
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # auto-profile reuse (2026-09-23): when --batch is ABSENT the profile
    # store decides (least-step reuse). Explicit --batch always wins.
    batch_used = args.batch
    batch_src = "explicit-or-default"
    if batch_used is None:
        try:
            from pipeline.profiles import ProfileStore
            _bs = ProfileStore().batch_for("gate_bench", cfg)
            if _bs is not None:
                batch_used = _bs
                batch_src = "profile-store:" + str(_bs)
            else:
                batch_used = 64
                batch_src = "no-profile-fallback-64"
        except Exception as _exc:
            print({"profile_store_error": str(_exc)[-100:]})
            batch_used = 64
            batch_src = "store-unavailable-64"
    pairs = []
    for bare, refs in load_pairs(args.src, cfg["model"]["ctx"]):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break
    preds = predict_bares(model, pairs, cfg["model"]["ctx"], device, batch_used)
    refs_all = ["\t".join(refs) for _, refs in pairs]
    n = len(pairs)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    refp = outp.with_suffix(".ref.txt")
    outp.write_text("\n".join(preds) + "\n", encoding="utf-8")
    refp.write_text("\n".join(refs_all) + "\n", encoding="utf-8")
    print(json.dumps({"ckpt": str(ck), "src": args.src, "sentences": n,
                      "batch": batch_used, "batch_source": batch_src,
                      "params": n_params, "out": args.out,
                      "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "ref_out": str(refp)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
