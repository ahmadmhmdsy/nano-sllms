
r"""Stage 0 post-analysis - the per-line structure behind the aggregate CER.

Answers two things the aggregate hides:
  1. Is data/hflines/val uniformly mediocre, or bimodal (printed lines the model
     reads well vs handwriting it cannot)?
  2. What fraction of data/hflines/val behaves like PRINTED text (both arms
     decode it under 20% CER)? That fraction sizes the handwriting problem for
     Stage 1. This is a PROXY, not ground truth - both candidate models are
     trained on synthetic printed Arabic, so agreement at low CER is strong
     evidence of print but not a label.

Usage: & .\.venv\Scripts\python.exe arabic_ocr\scripts\stage0_analyze.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs/ocr_bakeoff/per_image.jsonl"
OUT = REPO / "runs/ocr_bakeoff"
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 80), (80, 101)]


def main() -> int:
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines() if x.strip()]
    models = sorted({r["model"] for r in rows})
    by = defaultdict(dict)
    refs = {}
    for r in rows:
        by[(r["set"], r["image"])][r["model"]] = r["cer"]
        refs[(r["set"], r["image"])] = r["ref"]
    sets = sorted({r["set"] for r in rows})

    out = {"models": models, "sets": {}, "label_schema": {}}
    for sn in sets:
        keys = [k for k in by if k[0] == sn]
        entry = {"n": len(keys), "buckets": {}, "both_lt20_pct": None}
        for mn in models:
            cers = [by[k].get(mn) for k in keys if mn in by[k]]
            b = {}
            for lo, hi in BUCKETS:
                b["%d-%d" % (lo, hi - 1)] = round(100.0 * sum(1 for c in cers if lo <= c < hi) / max(1, len(cers)), 1)
            entry["buckets"][mn] = b
        both = [k for k in keys if all(by[k].get(m, 999) < 20 for m in models)]
        hard = [k for k in keys if all(by[k].get(m, 0) > 80 for m in models)]
        entry["both_lt20_pct"] = round(100.0 * len(both) / max(1, len(keys)), 1)
        entry["both_gt80_pct"] = round(100.0 * len(hard) / max(1, len(keys)), 1)
        out["sets"][sn] = entry

    # Classify the in-domain set so Stage 1 can stratify on it.
    cls = {}
    for k in [k for k in by if k[0] == "hflines_val"]:
        c = [by[k].get(m) for m in models]
        if all(x is not None and x < 20 for x in c):
            lab = "printed_like"
        elif all(x is not None and x > 80 for x in c):
            lab = "hard_likely_handwritten"
        else:
            lab = "middle"
        cls[k[1]] = {"c": {m: by[k].get(m) for m in models}, "class": lab, "ref_len": len(refs[k])}
    (OUT / "hflines_classified.json").write_text(json.dumps(cls, ensure_ascii=False, indent=1), encoding="utf-8")

    for sn in sets:
        d = REPO / ("data/hflines/val" if sn == "hflines_val" else "data/benchmark/" + sn)
        try:
            data = json.loads((d / "labels.json").read_text(encoding="utf-8"))
            out["label_schema"][sn] = sorted({k for r in data[:50] for k in r})
        except Exception as exc:  # noqa: BLE001
            out["label_schema"][sn] = "ERR %s" % exc

    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    counts = defaultdict(int)
    for v in cls.values():
        counts[v["class"]] += 1
    print("hflines_val classes:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
