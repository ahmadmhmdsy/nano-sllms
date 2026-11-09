"""E-23c part 2 (disk-light) - pack ZM rows into v3qz tokens = v3q train + ZM windows.

Single staging pass writing rows DIRECTLY in shuffled order; sources stay
memmaps/in-RAM, so peak disk is only the final 2.7 GB. Val copied IDENTICAL.
"""
import json, re, sys
from pathlib import Path
import numpy as np, random

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import tokenizer as TK
import passthrough as PT

CTX = 128
V3Q = REPO / "data" / "diac" / "v3q" / "tokens"
OUTD = REPO / "data" / "diac" / "v3qz" / "tokens"
OUTD.mkdir(parents=True, exist_ok=True)

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

kept = quarantined = 0
rows = []
for line in open(REPO / "data/diac/raw/e23c_zm_labels/zm_labels.jsonl", encoding="utf-8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    diac = r.get("diac") or ""
    if len(diac) < 40:
        continue
    for w in windows(diac):
        try:
            units = PT.parse(w)
        except PT.QuarantineError:
            quarantined += 1
            continue
        bases = "".join(u.raw for u in units if u.is_target)
        labels = [u.label for u in units if u.is_target]
        if not bases:
            continue
        rows.append((bases, labels))
        kept += 1
        if kept % 20000 == 0:
            print("zm rows", kept, "q", quarantined, flush=True)
print("ZM rows:", kept, "quarantined:", quarantined, flush=True)

zm_ids = np.zeros((len(rows), CTX), dtype=np.int64)
zm_ys = np.full((len(rows), CTX), -1, dtype=np.int8)
for i, (b, l) in enumerate(rows):
    seq = TK.encode(b)[:CTX]
    zm_ids[i, :len(seq)] = seq
    zm_ys[i, :len(seq)] = l[:len(seq)]
n_zm = len(rows)
del rows

tr = np.load(str(V3Q / "train_ids.npy"), mmap_mode="r")
tr_y = np.load(str(V3Q / "train_y.npy"), mmap_mode="r")
n_v3q = len(tr)
n = n_v3q + n_zm
print("v3q:", n_v3q, "zm:", n_zm, "->", n, flush=True)

o = np.lib.format.open_memmap(str(OUTD / "train_ids.npy"), mode="w+", dtype=np.int64, shape=(n, CTX))
oy = np.lib.format.open_memmap(str(OUTD / "train_y.npy"), mode="w+", dtype=np.int8, shape=(n, CTX))
rng = random.Random(20260918)
perm = np.arange(n); rng.shuffle(perm)
CH = 100000
for s0 in range(0, n, CH):
    e = min(n, s0 + CH)
    take = perm[s0:e]
    mv = take < n_v3q
    o[s0:e][mv] = tr[take[mv]]
    o[s0:e][~mv] = zm_ids[take[~mv] - n_v3q]
    oy[s0:e][mv] = tr_y[take[mv]]
    oy[s0:e][~mv] = zm_ys[take[~mv] - n_v3q]
    o.flush(); oy.flush()
    print("chunk", e, flush=True)
import shutil
shutil.copyfile(V3Q / "val_ids.npy", str(OUTD / "val_ids.npy"))
shutil.copyfile(V3Q / "val_y.npy", str(OUTD / "val_y.npy"))
print("v3qz DONE rows:", n)
