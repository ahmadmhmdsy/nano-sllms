"""Apples-to-apples teacher comparison on the SHARED held-out set.

Each teacher's own held-out stratum contains different lines, so those numbers
are not comparable. This computes the lines that are outside BOTH teachers'
training corpora and scores both teachers on exactly those lines.
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
OUT = ROOT / "runs" / "stage05c" / "teacher_clean_comparison.json"
MUHARAF_HF_TRAIN = ["muharaf_train.csv", "muharaf_train_extra_1.csv", "muharaf_train_extra_2.csv"]


def norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = s.replace("\u0640", "")
    s = re.sub(r"[\u064b-\u0652]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def local_texts(rel):
    p = ROOT / rel
    if not p.exists():
        return set()
    return {norm(r.get("text", "")) for r in json.loads(p.read_text(encoding="utf-8-sig"))}


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
        return {"n": 0}
    ed_sum = sum(ed(r["hyp"], r["ref"]) for r in rows)
    ch = sum(len(r["ref"]) for r in rows)
    cers = sorted(r["cer"] for r in rows)
    n = len(rows)
    return {"n": n, "cer": round(100.0 * ed_sum / max(1, ch), 2),
            "cer_median": round(cers[n // 2], 2),
            "exact_pct": round(100.0 * sum(1 for r in rows if r["hyp"] == r["ref"]) / n, 1),
            "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1)}


def load(rel):
    return [json.loads(l) for l in (ROOT / rel).read_text(encoding="utf-8").splitlines() if l.strip()]


khatt_train = local_texts("data/hf/khatt_teklia/train/labels.json")
muh_train = muharaf_hf_texts()
khatt_alt = khatt_train | local_texts("data/hf/khatt_teklia/validation/labels.json")
muh_alt = muh_train | local_texts("data/hf/muharaf/validation/labels.json")
union_main = khatt_train | muh_train
union_alt = khatt_alt | muh_alt

fix = load("runs/stage05c/perline_hflines_val.jsonl")
muh = load("runs/stage05c/perline_muharaf_hflines_val.jsonl")

report = {"note": "shared held-out comparison; the ONLY directly comparable teacher number",
          "reference_control_unmirrored_zeroshot": 57.80, "sets": {}}

for label, excl in (("clean_main", union_main), ("clean_strict", union_alt)):
    f_rows = [r for r in fix if norm(r["ref"]) not in excl]
    m_rows = [r for r in muh if norm(r["ref"]) not in excl]
    assert len(f_rows) == len(m_rows)
    entry = {"n_lines": len(f_rows),
             "khatt_train": len(khatt_train), "muharaf_hf_train": len(muh_train),
             "in_training_union": len(fix) - len(f_rows),
             "FixFips/arabicocr-khatt": stats(f_rows),
             "sdkv2/muharaf-arabic-ocr": stats(m_rows)}
    report["sets"][label] = entry

c = report["sets"]["clean_main"]
report["verdict"] = {
    "fixfips_cer_shared_clean": c["FixFips/arabicocr-khatt"]["cer"],
    "muharaf_cer_shared_clean": c["sdkv2/muharaf-arabic-ocr"]["cer"],
    "better_teacher": ("FixFips/arabicocr-khatt"
                       if c["FixFips/arabicocr-khatt"]["cer"] <= c["sdkv2/muharaf-arabic-ocr"]["cer"]
                       else "sdkv2/muharaf-arabic-ocr"),
    "vs_control_pp_fixfips": round(57.80 - c["FixFips/arabicocr-khatt"]["cer"], 2),
    "vs_control_pp_muharaf": round(57.80 - c["sdkv2/muharaf-arabic-ocr"]["cer"], 2),
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True, indent=2))
