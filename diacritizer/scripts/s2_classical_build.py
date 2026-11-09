"""E-23b S2 — classical tashkeela-family expansion, stage 1.

Streams the three ungated HF tashkeela-mirror datasets, converts to
(bases, labels) 128-char windows via the SAME PT.parse/TK pipeline, and
drops: (a) windows whose 10-word shingles hit the 4 gates OR fadel train
(the already-owned classical pool), (b) exact book-level head-hash dupes.

Output: data/diac/v3t/classical_rows.jsonl ({"b","l"} rows, same format as
e22_pack_qcri staging) + counters. Stops at NEW_WINDOWS target.
CPU + net only; background-safe. Env: PYTHONIOENCODING=utf-8.
"""
import json, os, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import tokenizer as TK
import passthrough as PT

CTX = 128
TARGET = int(os.environ.get("S2_TARGET", "1200000"))
OUTD = REPO / "data" / "diac" / "v3t"
OUTD.mkdir(parents=True, exist_ok=True)
OUT = OUTD / "classical_rows.jsonl"

def norm_text(t):
    t = unicodedata.normalize("NFC", str(t))
    return " ".join(t.split())

def wnorm(t):
    return " ".join(re.sub("[^\u0621-\u064a ]", " ", str(t)).split())

import re

def shingles(words, k=10):
    return [" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))]

def windows(text, ctx=CTX):
    if len(text) <= ctx:
        yield text
        return
    words = text.split(" ")
    cur = ""
    for w in words:
        if len(w) > ctx:
            if cur:
                yield cur.rstrip(" ")
                cur = ""
            yield w[:ctx]
            continue
        cand = (cur + " " + w).strip(" ")
        if len(cand) > ctx and cur:
            yield cur
            cur = w
        else:
            cur = cand
    if cur:
        yield cur

# --- shingle barrier: gates + fadel train ---
barrier = set()
for path in [
    "data/diac/raw/fadel/test.txt",
    "data/diac/raw/fadel/val.txt",
    "data/diac/raw/fadel/train.txt",
    "data/diac/raw/wikinews/wikinews2014_multi_ref.diac",
    "data/diac/raw/wikinews/wikinews2024_multi_ref.diac",
]:
    p = REPO / path
    if not p.exists():
        print("[skip]", path); continue
    for ln in open(p, encoding="utf-8"):
        w = wnorm(ln).split()
        for s in shingles(w):
            barrier.add(hash(s))
import pyarrow.parquet as pq
pf = pq.ParquetFile(REPO / "data/diac/raw/sadeed_25/sadeed25.parquet")
for batch in pf.iter_batches(batch_size=1024):
    for col in batch.column_names:
        for t in batch.column(col).to_pylist():
            w = wnorm(t).split()
            for s in shingles(w):
                barrier.add(hash(s))
print("barrier shingles:", len(barrier), flush=True)

def parse_window(text, out_f, acc):
    try:
        units = PT.parse(text)
    except PT.QuarantineError:
        acc["quar"] += 1
        return
    bases = "".join(u.raw for u in units if u.is_target)
    labels = [u.label for u in units if u.is_target]
    if not bases:
        return
    out_f.write(json.dumps({"b": list(bases), "l": labels}, ensure_ascii=False) + "\n")
    acc["kept"] += 1

from datasets import load_dataset

def stream_books():
    ds = load_dataset("community-datasets/tashkeela", split="train", streaming=True)
    for r in ds:
        yield "books", str(r.get("book", "")), str(r.get("text", ""))

def stream_asas():
    ds = load_dataset("asas-ai/Tashkeela", split="train", streaming=True)
    for r in ds:
        yield "asas", "", str(r.get("text", ""))

def stream_arbml():
    ds = load_dataset("arbml/tashkeela", split="train", streaming=True)
    for r in ds:
        yield "arbml", "", str(r.get("diacratized", "") or "")

acc = {"kept": 0, "quar": 0, "hit": 0, "dupdoc": 0, "shortclean": 0}
seen_doc = set()
src_counts = {"books": 0, "asas": 0, "arbml": 0}
stop = False
with open(OUT, "w", encoding="utf-8") as out_f:
    for srcname, book, text in list(stream_books()) + list(stream_asas()) + list(stream_arbml()):
        if acc["kept"] >= TARGET:
            break
        if srcname == "books":
            src_counts["books"] += 1
            head = norm_text(text)[:400]
            key = hash(head)
            if not head or key in seen_doc:
                acc["dupdoc"] += 1
                continue
            seen_doc.add(key)
        elif srcname == "asas":
            src_counts["asas"] += 1
        else:
            src_counts["arbml"] += 1
        text = norm_text(text)
        if len(text) > 4000000:
            text = text[:4000000]
        for win in windows(text):
            wn_w = wnorm(win).split()
            if len(wn_w) >= 10:
                sh = shingles(wn_w)
                if any(hash(s) in barrier for s in sh):
                    acc["hit"] += 1
                    continue
            if len(win) < 40:
                continue
            parse_window(win, out_f, acc)
            if acc["kept"] >= TARGET:
                stop = True
                break
        if stop:
            break
        if acc["kept"] % 50000 < 40:
            print("kept", acc["kept"], "hit", acc["hit"], "quar", acc["quar"], "dupdoc", acc["dupdoc"], "at", srcname, src_counts, flush=True)

print("DONE", json.dumps(acc), src_counts, flush=True)
