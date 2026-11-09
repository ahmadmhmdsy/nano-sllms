"""Sadeed overlap gate (pre-registered mandatory check, plan section 2.1).

Measures text overlap between the Sadeed_Tashkeela TRAIN/TEST shards and the
SadeedDiac-25 benchmark gate rows, at mark-stripped normalization:
  exact row match (normalized string set intersection)
  and 40-char shingle containment (near-dup; shingle overlap per gate row).
Writes data/diac/bench/sadeed_overlap_report.json.
CPU only - safe beside any GPU run.
"""
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "diac" / "raw"
OUT = REPO / "data" / "diac" / "bench" / "sadeed_overlap_report.json"
MARK_CHARS = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0653\u0654\u0655")


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch not in MARK_CHARS)
    return " ".join(s.split())


def shingles(s, k=40):
    t = norm(s)
    t = t.replace(" ", "")
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(0, len(t) - k + 1, 1)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    shingle_index = set()
    exact_set = set()
    char_budget = 0
    train_files = sorted((RAW / "sadeed_tashkeela").glob("data/train-*.parquet"))
    test_files = sorted((RAW / "sadeed_tashkeela").glob("data/test-*.parquet"))
    for fp in train_files + test_files:
        pf = pq.ParquetFile(str(fp))
        for batch in pf.iter_batches(batch_size=2048, columns=["input"]):
            for t in batch.column(0).to_pylist():
                if isinstance(t, str) and t.strip():
                    n = norm(t)
                    exact_set.add(n)
                    shingle_index |= shingles(t)
                    char_budget += len(n)
    print("corpus rows:", len(exact_set), "chars:", char_budget)

    gate = pq.read_table(str(RAW / "sadeed_25" / "sadeed25.parquet"))
    rows = gate.to_pylist()
    ex_hit = 0
    sh_frac = []
    for row in rows:
        n = norm(row.get("input", ""))
        if n in exact_set:
            ex_hit += 1
        sh = [c in shingle_index for c in shingles(row.get("input", ""))
              if len(c) >= 40]
        if sh:
            sh_frac.append(sum(sh) / len(sh))
    print("gate rows:", len(rows))
    print("exact gate-row matches:", ex_hit)
    print("mean shingle-overlap (corpus text seen in gate):",
          round(sum(sh_frac) / max(1, len(sh_frac)), 4))
    OUT.write_text(json.dumps({
        "train_files": [p.name for p in train_files],
        "test_files": [p.name for p in test_files],
        "corpus_chars": char_budget,
        "corpus_rows": len(exact_set),
        "gate_rows": len(rows),
        "gate_exact_hits": ex_hit,
        "gate_mean_shingle_overlap": round(sum(sh_frac) / max(1, len(sh_frac)), 4),
        "verdict": "OVERLAP-FLAGGED" if ex_hit > 0 or (sum(sh_frac) / max(1, len(sh_frac))) > 0.25 else "GATE-CLEAN",
    }, indent=2), encoding="utf-8")
    print("verdict:", json.loads(OUT.read_text(encoding="utf-8"))["verdict"])


if __name__ == "__main__":
    main()
