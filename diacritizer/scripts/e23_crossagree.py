"""E-23e — Z-Mahmood + gold cross-agreement validator (S5).

For a candidate text file (UTF-8, one text per line; bare-ish or with
diacritics: both are normalized, model runs on bare inputs):
  1. run gold (D-line) and Z-Mahmood on each line (bare);
  2. merge both predictions' word-level label maps;
  3. write:
     - <out>.agree.pred.txt   (labels both models agree on; disagreement =
                               bare skeleton, i.e. FAILED to certify)
     - <out>.flagged.txt (lines with >=flag-threshold disagreement words)
     - <out>.summary.json (per-line agreement %, totals)

Usage:
  & .\.venv\Scripts\python.exe -X utf8 diacritizer/scripts/e23_crossagree.py SRC OUT [--device cpu|cuda] [--limit N]

CPU-safe beside any train job (--device cpu default).
"""
import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DIAC = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670"
STRIP = re.compile("[" + DIAC + "]")
KEYMAP = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0649": "\u064a", "\u0640": ""})

def bare_key(w):
    return STRIP.sub("", w).translate(KEYMAP)

def marks_map(pred):
    """bare_key -> set of mark per-base positions encoded as string 'd<idx>=<label>'."""
    m = {}
    for w in pred.split():
        k = bare_key(w)
        # extract label tuple: base letters + mark labels per base
        labels = []
        idx = 0
        for ch in w:
            if "\u064b" <= ch <= "\u0652":
                labels.append((idx, ch))
            elif ch == "\u0627" or "\u0621" <= ch <= "\u064a":
                idx += 1
        m.setdefault(k, set()).add(tuple(labels))
    return m

def agree_line(pred_a, pred_g):
    """Fraction of base positions where A and gold maps agree (both non-empty)."""
    am, gm = marks_map(pred_a), marks_map(pred_g)
    keys = set(am) & set(gm)
    if not keys:
        return 0.0, 1
    agree = sum(1 for k in keys if am[k] & gm[k])
    return agree / len(keys), len(keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_prefix")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flag-below", type=float, default=0.50,
                    help="lines with agreement < this -> flagged.txt")
    args = ap.parse_args()

    import torch, yaml
    sys.path.insert(0, str(REPO / "diacritizer" / "src"))
    sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
    sys.path.insert(0, str(REPO / "models/e19/zmahood-src/src"))

    from model import build_from_config
    import bench as B
    import yaml as _y
    ck = torch.load(REPO / "runs/diac/stage2b2500/final/model.pt", map_location="cpu", weights_only=False)
    gcfg = _y.safe_load((REPO / "runs/diac/stage2b2500/final/config.yaml").read_text(encoding="utf-8"))
    gold = build_from_config(gcfg["model"], 171).eval()
    gold.load_state_dict(ck)
    if args.device == "cuda":
        gold = gold.to("cuda")
    ctx = gcfg["model"]["ctx"]
    from diacritize import Diacritizer as ZM
    zm = ZM.from_pretrained(no_cache=True)

    dev = args.device
    lines = [l.strip() for l in open(args.src, encoding="utf-8") if l.strip()]
    if args.limit:
        lines = lines[:args.limit]
    agree_out, flagged = [], []
    summary = {"lines": len(lines), "flagged": 0, "mean_agreement": 0.0}
    tot = 0.0
    out_path = Path(args.out_prefix)
    for n, text in enumerate(lines):
        bare = STRIP.sub("", " ".join(text.split()))
        if not bare:
            continue
        gp = B.predict_bare(gold, bare, ctx, dev)
        zp = zm.diacritize(bare)
        frac, npos = agree_line(zp, gp)
        tot += frac
        if frac < args.flag_below:
            flagged.append(text)
        agree_out.append(gp)
        summary["mean_agreement"] = round(tot / (n + 1), 4)
        if (n + 1) % 25 == 0:
            print(f"[{n + 1}/{len(lines)}] mean agreement {tot / (n + 1):.3f}", flush=True)
    Path(args.out_prefix + ".agree.pred.txt").write_text("\n".join(agree_out) + "\n", encoding="utf-8")
    Path(args.out_prefix + ".flagged.txt").write_text("\n".join(flagged) + "\n", encoding="utf-8")
    summary["flagged"] = len(flagged)
    summary["mean_agreement"] = round(tot / max(1, len(agree_out)), 4)
    Path(args.out_prefix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", summary)

main()
