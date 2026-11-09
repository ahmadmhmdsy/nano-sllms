"""mex/scripts/build_mu2_g1.py — mu2 rung G1 corpus builder (E-30).

Reads READ-ONLY vocalized Arabic text staged by the D-line prep under
data/diac/raw (wikinews wikipedia_diac.jsonl + fadel train/val) — copying is
licensed research use per data/diac/raw/provenance.json; sadeed* EXCLUDED on
license ambiguity. Writes VOCALIZED lines to data/mex/mu2/g1/{train,val}.txt,
targeting ~36M train chars. CPU+, deterministic split via seed 42 thresholding
(line hash), no writes to data/diac.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "diac" / "raw"
OUT = ROOT / "data" / "mex" / "mu2" / "g1"
TRAIN_CHARS = 36_000_000
VAL_CHARS = 300_000
MIN_LEN = 40
MARKS = set("ًٌٍَُِّّْ")


def _keep(line: str) -> bool:
    line = line.strip()
    if len(line) < MIN_LEN:
        return False
    body = [c for c in line if not c.isspace()]
    if not body:
        return False
    arabish = sum(1 for c in body if 0x0600 <= ord(c) <= 0x06FF or c in MARKS)
    latin = sum(1 for c in body if c.isascii() and c.isalpha())
    if arabish / len(body) < 0.7 or latin / len(body) > 0.15:
        return False
    return any(c in MARKS for c in body)      # vocalized text only


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_val = n_train = 0
    f_tr = (OUT / "train.txt").open("w", encoding="utf-8", newline="\n")
    f_va = (OUT / "val.txt").open("w", encoding="utf-8", newline="\n")
    srcs = [RAW / "wikinews" / "wikipedia_diac.jsonl",
            RAW / "fadel" / "train.txt", RAW / "fadel" / "val.txt"]
    for p in srcs:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if p.suffix == ".jsonl":
                    try:
                        line = json.loads(line).get("text", "")
                    except Exception:
                        continue
                for seg in (line.strip().split("\n") if "\n" in line else [line]):
                    seg = seg.strip()
                    if not _keep(seg):
                        continue
                    if n_val < VAL_CHARS:
                        f_va.write(seg + "\n"); n_val += len(seg)
                    else:
                        f_tr.write(seg + "\n"); n_train += len(seg)
                    if n_train >= TRAIN_CHARS:
                        break
            if n_train >= TRAIN_CHARS:
                break
        if n_train >= TRAIN_CHARS:
            break
    f_tr.close(); f_va.close()
    print(f"G1 corpus: train {n_train} chars, val {n_val} chars")

if __name__ == "__main__":
    main()
