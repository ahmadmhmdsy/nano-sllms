# mex/scripts/build_x1_words.py — CPU-only, deterministic; NO deletions.
"""Sample X1 bare|vocalized word lines from the committed E-20 word cache.

Writes data/mex/x1/{train,val,test}.txt — one 'bare|vocalized\n' per line.
Capped sample (train 60k / val 1k / test 2k) keeps X1 small: the μ0 question
is feasibility at ~203K, not D-line SOTA.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "e19" / "our_word_cache.json"
OUT = ROOT / "data" / "mex" / "x1"
CAPS = {"train": 60_000, "val": 1_000, "test": 2_000}
MARKS = set("ًٌٍَُِّّْ")


def _pairs() -> list[tuple[str, str]]:
    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("words"), dict):
        raw = raw["words"]          # Step-1 probe: {meta: {...}, words: {bare: voc}}
    out = []
    for k, v in raw.items():                      # {bare: vocalized-str} per Step-1 probe
        voc = v if isinstance(v, str) else (v.get("vocalized") if isinstance(v, dict) else None)
        if (isinstance(voc, str) and 2 <= len(k) <= 30 and len(voc) > len(k)
                and any(c in MARKS for c in voc)
                and "|" not in k and "|" not in voc
                and "\n" not in k and "\n" not in voc):
            out.append((k, voc))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = _pairs()
    rng = random.Random("mex-x1")
    rng.shuffle(lines)
    n_used = 0
    with (OUT / "test.txt").open("w", encoding="utf-8", newline="\n") as f_test, \
         (OUT / "val.txt").open("w", encoding="utf-8", newline="\n") as f_val, \
         (OUT / "train.txt").open("w", encoding="utf-8", newline="\n") as f_train:
        handles = [("test", f_test, CAPS["test"]), ("val", f_val, CAPS["val"]),
                   ("train", f_train, CAPS["train"])]
        idx = 0
        for kind, fh, cap in handles:
            wrote = 0
            while wrote < cap and idx < len(lines):
                bare, voc = lines[idx]; idx += 1
                fh.write(f"{bare}|{voc}\n")
                wrote += 1
                n_used += 1
    print(f"X1 words written: {n_used} (head idx={idx})")


if __name__ == "__main__":
    main()
