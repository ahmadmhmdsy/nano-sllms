"""Stage 0.5c (b) - zero-shot evaluation of sdkv2/muharaf-arabic-ocr (TASKS 134).

The second in-domain teacher candidate: character-level CNN-Transformer-CTC
trained on the Muharaf corpus (card: test CER 0.1331 with the full calibrated
decode + neural-LM rerank + risk reranker; ~0.170 for the backbone alone under
plain greedy CTC).

Protocol is IDENTICAL to E-89 (eval_fixfips_khatt.py) so the numbers compare
directly with the 46.93% / 57.80% pair:
  * greedy CTC decode of the model's own logits, blank_id = 0
  * NO language model, NO reranker (the shipped char_ngram_lm / neural_char_lm /
    risk_reranker checkpoints are deliberately unused)
  * CER = sum(edit_distance(hyp, ref)) / sum(len(ref)) on raw strings
  * the same Arabic orthographic normalization as E-89 as a secondary metric
  * median line CER + the cer_lt10/20/50 tail, because a mean hides the tail

Model code is the author's own (submission_code/torch_models.py +
ocr_helpers.py), imported from the local download under arabic_ocr/models/ -
nothing is pip-installed.

Usage:
  & .venv/Scripts/python.exe arabic_ocr/scripts/eval_muharaf_hflines.py --set all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata as ud
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
CKPT_PATH = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"
OUT_DIR = ROOT / "runs" / "stage05c"

# identical control to E-89: PaddleOCR PP-OCRv6_small_rec, same corpus,
# zero-shot free-running un-mirrored.
CONTROL_CER_HFLINES = 57.80

SETS = {
    "hflines": {
        "dir": ROOT / "data" / "hflines" / "val",
        "tag": "muharaf_hflines_val",
        "control": CONTROL_CER_HFLINES,
    },
    # Stage 0.5d: the REAL held-out splits - never in any teacher's training
    # data. No control number has been measured on them, so control=None and no
    # delta is reported until one is.
    "muharaf_test": {
        "dir": ROOT / "data" / "hf" / "muharaf" / "test",
        "tag": "muharaf_test",
        "control": None,
    },
    "muharaf_val": {
        "dir": ROOT / "data" / "hf" / "muharaf" / "validation",
        "tag": "muharaf_validation",
        "control": None,
    },
    "khatt_test": {
        "dir": ROOT / "data" / "hf" / "khatt_teklia" / "test",
        "tag": "khatt_teklia_test",
        "control": None,
    },
}

# 'all' keeps its original Stage 0.5c meaning; the held-out splits are selected
# by name (or as a comma list) so the tracked summary files stay stable.
ALL_SETS = ["hflines"]

sys.path.insert(0, str(MODEL_DIR))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
from PIL import Image                   # noqa: E402

from submission_code.torch_models import load_ctc_backbone, recognise_line   # noqa: E402


# ---------------------------------------------------------------- metrics
def edit_distance(a, b) -> int:
    """Levenshtein distance, sequence-agnostic (str or list)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


_re_diac = re.compile(r"[\u064B-\u0652]")


def norm_ar(s: str) -> str:
    """E-89's normalization (author metrics.py) - kept byte-identical on purpose."""
    s = ud.normalize("NFKC", s)
    s = s.replace("\u0640", "")
    s = _re_diac.sub("", s)
    s = (s.replace("\u0623", "\u0627")
           .replace("\u0625", "\u0627")
           .replace("\u0622", "\u0627")
           .replace("\u0649", "\u064A"))
    return re.sub(r"\s+", " ", s).strip()


def load_items(set_dir: Path):
    labels = json.loads((set_dir / "labels.json").read_text(encoding="utf-8"))
    items = []
    for r in labels:
        name = r.get("image") or r.get("img") or r.get("file")
        items.append((set_dir / name, r.get("text", r.get("label", ""))))
    return items


def sha256_head(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ---------------------------------------------------------------- evaluation
def evaluate(name, model, charset, limit: int = 0, want_examples: int = 0):
    spec = SETS[name]
    items = load_items(spec["dir"])
    if limit:
        items = items[:limit]

    refs, hyps, paths, cers = [], [], [], []
    t0 = time.perf_counter()

    for i, (p, ref) in enumerate(items):
        hyp = recognise_line(model, charset, str(p))
        refs.append(ref)
        hyps.append(hyp)
        paths.append(str(p.relative_to(ROOT)))
        cers.append(100.0 * edit_distance(hyp, ref) / max(1, len(ref)))
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print("    %d/%d  %.2f img/s" % (i + 1, len(items), (i + 1) / max(el, 1e-9)),
                  flush=True)

    elapsed = time.perf_counter() - t0

    perline = [{"image": paths[i], "ref": refs[i], "hyp": hyps[i],
                "cer": round(cers[i], 2)} for i in range(len(refs))]

    def agg(hyp_list):
        ed = sum(edit_distance(h, r) for h, r in zip(hyp_list, refs))
        rc = sum(len(r) for r in refs)
        ew = rw = exact = 0
        for h, r in zip(hyp_list, refs):
            hw, rww = h.split(), r.split()
            ew += edit_distance(hw, rww)
            rw += len(rww)
            if h == r:
                exact += 1
        n = max(1, len(hyp_list))
        return {"cer": 100.0 * ed / max(1, rc), "wer": 100.0 * ew / max(1, rw),
                "exact_pct": 100.0 * exact / n}

    def agg_norm(hyp_list):
        ed = sum(edit_distance(norm_ar(h), norm_ar(r)) for h, r in zip(hyp_list, refs))
        rc = sum(len(norm_ar(r)) for r in refs)
        ew = rw = 0
        for h, r in zip(hyp_list, refs):
            hw, rww = norm_ar(h).split(), norm_ar(r).split()
            ew += edit_distance(hw, rww)
            rw += len(rww)
        return {"cer_norm": 100.0 * ed / max(1, rc), "wer_norm": 100.0 * ew / max(1, rw)}

    n = max(1, len(items))
    main, norm = agg(hyps), agg_norm(hyps)
    result = {
        "tag": spec["tag"],
        "dataset": str(spec["dir"].relative_to(ROOT)).replace("\\", "/"),
        "n": len(items),
        "model_id": "sdkv2/muharaf-arabic-ocr",
        "checkpoint": str(CKPT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256_head": sha256_head(CKPT_PATH),
        "checkpoint_bytes": CKPT_PATH.stat().st_size,
        "architecture": "CNN-Transformer-CTC (height-attention pooling), char-level",
        "decoder": "greedy CTC, no language model, no reranker",
        "params": sum(p.numel() for p in model.parameters()),
        "charset_size": charset.vocab_size,
        "cer": round(main["cer"], 2),
        "wer": round(main["wer"], 2),
        "exact_pct": round(main["exact_pct"], 2),
        "cer_norm": round(norm["cer_norm"], 2),
        "wer_norm": round(norm["wer_norm"], 2),
        "cer_median": round(sorted(cers)[len(cers) // 2], 2) if cers else None,
        "cer_lt10_pct": round(100.0 * sum(1 for c in cers if c < 10) / n, 1),
        "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1),
        "cer_lt50_pct": round(100.0 * sum(1 for c in cers if c < 50) / n, 1),
        "seconds": round(elapsed, 1),
        "img_per_s": round(len(items) / max(elapsed, 1e-9), 2),
        "control_cer": spec.get("control"),
        "control_model": "medyas/arabic_PP-OCRv6_small_rec (zero-shot, un-mirrored, dynamic width)",
        "_perline": perline,
    }
    if result["control_cer"] is not None:
        result["delta_vs_control_pp"] = round(result["control_cer"] - result["cer"], 2)

    if want_examples:
        idx = list(range(min(want_examples, len(items))))
        result["examples"] = [
            {"idx": i, "image": paths[i], "ref": refs[i], "hyp": hyps[i],
             "cer": round(cers[i], 2)}
            for i in idx
        ]
    return result


def write_examples_file(results, path: Path):
    lines = []
    for r in results:
        if not r.get("examples"):
            continue
        lines.append("=" * 100)
        ctl = "n/a" if r["control_cer"] is None else "%.2f%%" % r["control_cer"]
        lines.append("DATASET: %s  (n=%d, CER=%.2f%%, control=%s)" %
                     (r["dataset"], r["n"], r["cer"], ctl))
        lines.append("=" * 100)
        for ex in r["examples"]:
            lines.append("")
            lines.append("[%d] %s   line CER=%.2f%%" % (ex["idx"], ex["image"], ex["cer"]))
            lines.append("  REF: %s" % ex["ref"])
            lines.append("  HYP: %s" % ex["hyp"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hflines",
                    help="comma-separated set names, or 'all' (= %s)" % ",".join(ALL_SETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--examples", type=int, default=20)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = None
    if args.device != "auto":
        device = torch.device(args.device)
    print("checkpoint: %s (%d bytes, sha256 %s...)" %
          (CKPT_PATH.name, CKPT_PATH.stat().st_size, sha256_head(CKPT_PATH)), flush=True)
    model, charset, cfg = load_ctc_backbone(str(CKPT_PATH), device=device)
    print("device: %s  charset=%d  cfg=%s" %
          (next(model.parameters()).device, charset.vocab_size, json.dumps(cfg)), flush=True)
    print("params: %d" % sum(p.numel() for p in model.parameters()), flush=True)

    if args.set == "all":
        names = list(ALL_SETS)
    else:
        names = [s.strip() for s in args.set.split(",") if s.strip()]
    unknown = [s for s in names if s not in SETS]
    if unknown:
        ap.error("unknown set(s) %s; known: %s" % (", ".join(unknown), ", ".join(SETS)))
    results = []
    for name in names:
        print("== evaluating %s (%s)" % (name, SETS[name]["dir"]), flush=True)
        r = evaluate(name, model, charset, limit=args.limit,
                     want_examples=args.examples)
        results.append(r)
        perline = r.pop("_perline", [])
        print(json.dumps({k: v for k, v in r.items() if k != "examples"},
                         ensure_ascii=True, indent=2), flush=True)
        pl_path = OUT_DIR / ("perline_%s.jsonl" % r["tag"])
        with pl_path.open("w", encoding="utf-8") as fh:
            for row in perline:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("wrote %s (%d lines)" % (pl_path.relative_to(ROOT), len(perline)), flush=True)

    suffix = "_limit%d" % args.limit if args.limit else ""
    for r in results:
        out = OUT_DIR / ("eval_muharaf_%s%s.json" % (r["tag"], suffix))
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % out.relative_to(ROOT), flush=True)

    tags = [r["tag"] for r in results]
    if tags == ["muharaf_hflines_val"]:
        ex_name = "examples_muharaf_hflines_val%s.txt" % suffix
    else:
        ex_name = "examples_muharaf_%s%s.txt" % ("_".join(tags), suffix)
    ex_path = OUT_DIR / ex_name
    write_examples_file(results, ex_path)
    print("wrote %s" % ex_path.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
