"""Durable probe: is data/ocr_textpool/texts.jsonl safe as an LM corpus?

Checks whether any eval-split reference line appears verbatim (or verbatim
after whitespace removal) inside the pool, which would be test leakage.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(".")
spec = importlib.util.spec_from_file_location("clc", ROOT / "arabic_ocr" / "scripts" / "char_lm_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def norm(t):
    return " ".join(C.strip_diacritics(t or "").split())


def hard(t):
    return "".join(ch for ch in norm(t) if not ch.isspace())


pool = []
with open(ROOT / "data" / "ocr_textpool" / "texts.jsonl", encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        t = norm(rec.get("plain") or rec.get("text") or "")
        if t:
            pool.append(t)
pool_set = set(pool)
pool_blob = "\n".join(pool)
pool_hard = "".join(hard(t) for t in pool)
print("pool texts=%d chars=%d hard_chars=%d" % (len(pool), len(pool_blob), len(pool_hard)))

for split in ["muharaf/validation", "muharaf/test", "khatt_teklia/test", "benchmark/isippt", "benchmark/khatt"]:
    folder = ROOT / "data" / ("hf/" + split if "hf" in split or split.startswith("muharaf") or split.startswith("khatt_teklia") else split)
    folder = ROOT / "data" / "hf" / split if (split.startswith("muharaf") or split.startswith("khatt_teklia")) else ROOT / "data" / split
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    refs = [norm(r.get("text", "")) for r in labels]
    refs = [r for r in refs if r]
    exact = sum(1 for r in refs if r in pool_set)
    blob = sum(1 for r in refs if len(r) >= 12 and r in pool_blob)
    hards = sum(1 for r in refs if len(hard(r)) >= 12 and hard(r) in pool_hard)
    print("%-24s n=%-5d exact=%-5d blob=%-5d hard=%-5d" % (split, len(refs), exact, blob, hards))
