"""Stratified Muharaf-overlap analysis of the Stage 0.5c (b) zero-shot eval.

Reads runs/stage05c/perline_muharaf_hflines_val.jsonl (one record per val line)
and splits the 1,654 val lines by whether their transcript appears in the
Muharaf manifests the teacher was trained on:

  * in_muharaf_train        - transcript is in the teacher's TRAINING manifests
                              => an in-train score, NOT transfer evidence
  * muharaf_val_or_test_only- in the teacher's val/test manifests (still its data)
  * not_in_any_muharaf_split- the only honest held-out stratum

Transcript matching OVER-estimates overlap (a repeated sentence can match
another image) and UNDER-estimates it (spacing/punctuation differences defeat
exact matching), so it is a bounds-style indicator. The held-out stratum is a
lower bound on the honest error; the in-train stratum is an upper bound on how
good the teacher looks with its own data.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata as ud
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
PERLINE = ROOT / "runs" / "stage05c" / "perline_muharaf_hflines_val.jsonl"
DEST = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
OUT = ROOT / "runs" / "stage05c" / "overlap_muharaf_hflines.json"

MANIFESTS = {
    "train": ["muharaf_train.csv", "muharaf_train_extra_1.csv", "muharaf_train_extra_2.csv"],
    "val": ["muharaf_validation.csv"],
    "test": ["muharaf_test.csv"],
}


def norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = s.replace("\u0640", "")
    s = re.sub(r"[\u064b-\u0652]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def manifest_set(names):
    got = set()
    for n in names:
        p = Path(hf_hub_download("sdkv2/muharaf-arabic-ocr",
                                 f"data/manifests/{n}", local_dir=str(DEST)))
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("text_raw", "text_norm"):
                    v = r.get(k)
                    if v:
                        got.add(norm(v))
    return got


def ed(a, b) -> int:
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


def stats(rows):
    if not rows:
        return {"n": 0}
    ed_sum = sum(ed(r["hyp"], r["ref"]) for r in rows)
    ch_sum = sum(len(r["ref"]) for r in rows)
    cers = sorted(r["cer"] for r in rows)
    n = len(rows)
    exact = sum(1 for r in rows if r["hyp"] == r["ref"])
    return {
        "n": n,
        "cer": round(100.0 * ed_sum / max(1, ch_sum), 2),
        "cer_median": round(cers[n // 2], 2),
        "exact_pct": round(100.0 * exact / n, 1),
        "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1),
        "cer_lt50_pct": round(100.0 * sum(1 for c in cers if c < 50) / n, 1),
    }


train_set = manifest_set(MANIFESTS["train"])
val_set = manifest_set(MANIFESTS["val"])
test_set = manifest_set(MANIFESTS["test"])

rows = [json.loads(l) for l in PERLINE.read_text(encoding="utf-8").splitlines() if l.strip()]
buckets = {"in_muharaf_train": [], "muharaf_val_or_test_only": [],
           "not_in_any_muharaf_split": [], "all": []}
for r in rows:
    t = norm(r["ref"])
    if t in train_set:
        k = "in_muharaf_train"
    elif t in val_set or t in test_set:
        k = "muharaf_val_or_test_only"
    else:
        k = "not_in_any_muharaf_split"
    buckets[k].append(r)
    buckets["all"].append(r)

report = {
    "note": "transcript-overlap stratification of the Stage 0.5c (b) zero-shot "
            "Muharaf eval; bounds-style, not an exact image-overlap rate",
    "teacher": "sdkv2/muharaf-arabic-ocr",
    "control_cer_hflines": 57.80,
    "muharaf_manifest_texts": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
    "strata": {k: stats(v) for k, v in buckets.items()},
}
tot = len(rows)
for k in ("in_muharaf_train", "muharaf_val_or_test_only", "not_in_any_muharaf_split"):
    report["strata"][k]["pct_of_val"] = round(100.0 * len(buckets[k]) / max(1, tot), 1)
report["headline"] = {
    "all_val_cer": report["strata"]["all"]["cer"],
    "pct_of_val_in_teacher_train": report["strata"]["in_muharaf_train"]["pct_of_val"],
    "held_out_cer": report["strata"]["not_in_any_muharaf_split"]["cer"],
    "held_out_n": report["strata"]["not_in_any_muharaf_split"]["n"],
    "verdict": ("the 23.64% headline is dominated by lines the teacher TRAINED on; "
                "the held-out stratum is the only transfer number"),
}

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True, indent=2))
print("wrote", OUT.relative_to(ROOT))
