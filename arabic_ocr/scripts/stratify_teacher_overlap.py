"""Put both Stage 0.5c teacher candidates on the same honest footing.

data/hflines was built by arabic_ocr/scripts/build_hf_lines.py from
data/hf (Teklia/KHATT_v1.0 + aamijar/muharaf-public) + data/reallines, excluding
every split named 'test', with no per-line provenance recorded. Its val split is
a random 6% sample of that union.

Both candidate teachers are trained on part of precisely that union:
  * FixFips/arabicocr-khatt        trained on KHATT
  * sdkv2/muharaf-arabic-ocr       trained on Muharaf (the model's own manifests
                                   list 22,091 training lines)
so a raw CER on data/hflines/val mixes in-train lines with held-out ones.

This script recovers per-line provenance by normalized transcript matching
against the local corpora, then reports each teacher's CER split into
in-teacher-train vs held-out. The held-out stratum is the ONLY comparable number.

Matching is a bounds-style indicator: it over-estimates overlap (a repeated
sentence can match a different image) and under-estimates it (spacing and
punctuation differences defeat exact matching).
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata as ud
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
OUT = ROOT / "runs" / "stage05c" / "teacher_overlap_stratification.json"

LOCAL_SOURCES = {
    "khatt_train": "data/hf/khatt_teklia/train/labels.json",
    "khatt_val": "data/hf/khatt_teklia/validation/labels.json",
    "khatt_test": "data/hf/khatt_teklia/test/labels.json",
    "muharaf_pub_train": "data/hf/muharaf/train/labels.json",
    "muharaf_pub_val": "data/hf/muharaf/validation/labels.json",
    "muharaf_pub_test": "data/hf/muharaf/test/labels.json",
    "reallines_train": "data/reallines/train/labels.json",
    "reallines_val": "data/reallines/val/labels.json",
}
MUHARAF_HF_TRAIN = ["muharaf_train.csv", "muharaf_train_extra_1.csv", "muharaf_train_extra_2.csv"]

TEACHERS = {
    "FixFips/arabicocr-khatt": {
        "perline": "runs/stage05c/perline_hflines_val.jsonl",
        "trained_on": "KHATT (no manifest published; proxied by the local Teklia KHATT copy)",
        "in_train": ["khatt_train"],
        "in_train_alt": ["khatt_train", "khatt_val"],
        "headline_cer": 46.93,
    },
    "sdkv2/muharaf-arabic-ocr": {
        "perline": "runs/stage05c/perline_muharaf_hflines_val.jsonl",
        "trained_on": "Muharaf (author's own 22,091-line training manifests)",
        "in_train": ["muharaf_hf_train"],
        "in_train_alt": ["muharaf_hf_train", "muharaf_pub_val"],
        "headline_cer": 23.64,
    },
}


def norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = s.replace("\u0640", "")
    s = re.sub(r"[\u064b-\u0652]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def local_texts(rel: str):
    p = ROOT / rel
    if not p.exists():
        return set()
    rows = json.loads(p.read_text(encoding="utf-8-sig"))
    return {norm(r.get("text", "")) for r in rows}


def muharaf_hf_texts():
    got = set()
    for n in MUHARAF_HF_TRAIN:
        p = Path(hf_hub_download("sdkv2/muharaf-arabic-ocr",
                                 f"data/manifests/{n}", local_dir=str(DEST)))
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("text_raw", "text_norm"):
                    if r.get(k):
                        got.add(norm(r[k]))
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
        return {"n": 0, "pct_of_val": 0.0}
    ed_sum = sum(ed(r["hyp"], r["ref"]) for r in rows)
    ch_sum = sum(len(r["ref"]) for r in rows)
    cers = sorted(r["cer"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "cer": round(100.0 * ed_sum / max(1, ch_sum), 2),
        "cer_median": round(cers[n // 2], 2),
        "exact_pct": round(100.0 * sum(1 for r in rows if r["hyp"] == r["ref"]) / n, 1),
        "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1),
    }


source_sets = {k: local_texts(v) for k, v in LOCAL_SOURCES.items()}
source_sets["muharaf_hf_train"] = muharaf_hf_texts()

report = {
    "note": "bounds-style transcript-overlap stratification; not an exact image-overlap rate",
    "corpus_counts": {k: len(v) for k, v in sorted(source_sets.items())},
    "teachers": {},
}

# provenance of the val set (which source corpus each line's transcript comes from)
val_pl = [json.loads(l) for l in
          (ROOT / "runs/stage05c/perline_hflines_val.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
prov = {}
for k in source_sets:
    prov[k] = sum(1 for r in val_pl if norm(r["ref"]) in source_sets[k])
report["val_provenance_counts"] = prov
report["val_n"] = len(val_pl)

for tname, spec in TEACHERS.items():
    rows = [json.loads(l) for l in
            (ROOT / spec["perline"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    main = set().union(*[source_sets[k] for k in spec["in_train"]])
    alt = set().union(*[source_sets[k] for k in spec["in_train_alt"]])
    in_rows = [r for r in rows if norm(r["ref"]) in main]
    out_rows = [r for r in rows if norm(r["ref"]) not in main]
    alt_out = [r for r in rows if norm(r["ref"]) not in alt]
    report["teachers"][tname] = {
        "trained_on": spec["trained_on"],
        "in_train_sets": spec["in_train"],
        "headline_cer_as_reported": spec["headline_cer"],
        "all_val": stats(rows),
        "in_teacher_train": stats(in_rows),
        "held_out": stats(out_rows),
        "held_out_alt_sets": spec["in_train_alt"],
        "held_out_alt": stats(alt_out),
    }
    t = report["teachers"][tname]
    t["comparable_held_out_cer"] = t["held_out"]["cer"]
    t["contamination_pct"] = t["in_teacher_train"].get("pct_of_val") or round(
        100.0 * len(in_rows) / max(1, len(rows)), 1)
    t["in_teacher_train"]["pct_of_val"] = t["contamination_pct"]
    t["held_out"]["pct_of_val"] = round(100.0 * len(out_rows) / max(1, len(rows)), 1)

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True, indent=2))
print("wrote", OUT.relative_to(ROOT))
