"""E-22 — shingle dedupe the QCRI wiki corpus against our 4 gate/bench texts.

Line is DROPPED if >=1% of its 10-word Arabic shingles appear in any gate
(over-dropping is fine; gates are sacred). Writes
data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl + counts.

CPU-only; run beside a train job is fine (single thread, low RAM).
"""
import json, re, sys

def norm_text(t):
    # Arabic bases only + spaces -> word shingle normalization
    t = re.sub("[^\u0621-\u064a ]", " ", t)
    return " ".join(t.split())

def shingles(words, k=10):
    return {tuple(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}

gate_files = [
    "data/diac/raw/fadel/test.txt",
    "data/diac/raw/wikinews/wikinews2014_multi_ref.diac",
    "data/diac/raw/wikinews/wikinews2024_multi_ref.diac",
    "data/diac/raw/sadeed_25/sadeed25.parquet",  # skipped inline below
]
gate_shingles = set()
for path in gate_files:
    try:
        if path.endswith(".parquet"):
            continue
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        print("[skip-missing]", path)
        continue
    for ln in lines:
        w = norm_text(ln).split()
        if len(w) >= 10:
            gate_shingles |= shingles(w)
print("gate shingles:", len(gate_shingles), flush=True)

# sadeed25 parquet gate
import pyarrow.parquet as pq
pf = pq.ParquetFile("data/diac/raw/sadeed_25/sadeed25.parquet")
for batch in pf.iter_batches(batch_size=1024):
    for col in batch.column_names:
        for t in batch.column(col).to_pylist():
            w = norm_text(str(t)).split()
            if len(w) >= 10:
                gate_shingles |= shingles(w)
print("gate shingles (with sadeed):", len(gate_shingles), flush=True)

src = "data/diac/raw/qcri_diac_clone/Wikipedia_20240420.diac.jsonl"
out = open("data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl", "w", encoding="utf-8")
kept = dropped = 0
for line in open(src, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    texts = [v for v in json.loads(line).values() if isinstance(v, str) and len(v) > 40]
    if not texts:
        dropped += 1
        continue
    w = norm_text(" ".join(texts)).split()
    sh = shingles(w)
    if sh and len([s for s in sh if s in gate_shingles]) / len(sh) >= 0.01:
        dropped += 1
        continue
    out.write(line + "\n")
    kept += 1
    if kept % 5000 == 0:
        print("kept", kept, "dropped", dropped, flush=True)
out.close()
print("DONE kept", kept, "dropped", dropped)
