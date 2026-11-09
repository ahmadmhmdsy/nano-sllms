"""E-22 — pack gate-deduped QCRI text and CONCAT into a v3q tokens set.

Output data/diac/v3q/tokens/{train_ids,train_y}.npy =
  v3 train rows (lex-kept order; re-shuffled here with the pages concat)
  + qcri_gatesafe windows (<=128 chars, space-split, PT.parse).
Val arrays are COPIED UNCHANGED from v3 -> direct comparability with arm A.
No cache in this arm (user dropped that line).

CPU background-safe; memmap-backed, chunked to keep RAM bounded.
"""
import json, re, sys
from pathlib import Path
import numpy as np, random

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import tokenizer as TK
import passthrough as PT

CTX = 128
V3 = REPO / "data" / "diac" / "v3" / "tokens"
OUTD = REPO / "data" / "diac" / "v3q" / "tokens"
OUTD.mkdir(parents=True, exist_ok=True)

def norm_text(t):
    # NFC and trim (same as prepare_data)
    import unicodedata
    return unicodedata.normalize("NFC", t.strip())

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

# pass 1: parse qcri lines into (bases, labels) pairs, chunk-stash to a temp jsonl
tmp = REPO / "data" / "diac" / "v3q" / "qcri_rows.jsonl"
quarantined = kept = 0
with tmp.open("w", encoding="utf-8") as f:
    for line in open(REPO / "data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl", encoding="utf-8"):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        text = obj.get("text")
        if not text:
            text = next((v for v in obj.values() if isinstance(v, str) and len(v) > 40), None)
        if not text:
            continue
        for w in windows(norm_text(text)):
            try:
                units = PT.parse(w)
            except PT.QuarantineError:
                quarantined += 1
                continue
            bases = "".join(u.raw for u in units if u.is_target)
            labels = [u.label for u in units if u.is_target]
            if not bases:
                continue
            f.write(json.dumps({"b": list(bases), "l": labels}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50000 == 0:
                print("rows", kept, "quarantined", quarantined, flush=True)
print("qcri rows:", kept, "quarantined:", quarantined, flush=True)

# pass 2: pack qcri rows -> memmap chunk writer
n_q = kept
tr = np.load(str(V3 / "train_ids.npy"))
tr_y = np.load(str(V3 / "train_y.npy"))
n_v3 = len(tr)
print("v3 train rows:", n_v3, "qcri rows:", n_q, flush=True if False else "")

# build qcri packed arrays directly to disk
q_ids_path = OUTD / "q_ids.npy"
q_y_path = OUTD / "q_y.npy"
ids = np.lib.format.open_memmap(q_ids_path, mode="w+", dtype=np.int64, shape=(n_q, CTX))
ys = np.lib.format.open_memmap(q_y_path, mode="w+", dtype=np.int8, shape=(n_q, CTX))
ids[:] = TK.PAD
ys[:] = -1
i = 0
with tmp.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        b = r["b"]
        seq = TK.encode(b)[:CTX]
        L = len(seq)
        ids[i, :L] = seq
        ys[i, :L] = r["l"][:L]
        i += 1
        if i % 100000 == 0:
            print("packed", i, flush=True)
print("packed qcri", i, flush=True)
del ids, ys
q_ids = np.load(str(q_ids_path), mmap_mode="r")
q_y = np.load(str(q_y_path), mmap_mode="r")

n = n_v3 + n_q
o_ids = np.lib.format.open_memmap(str(OUTD / "train_ids.npy"), mode="w+", dtype=np.int64, shape=(n, CTX))
o_y = np.lib.format.open_memmap(str(OUTD / "train_y.npy"), mode="w+", dtype=np.int8, shape=(n, CTX))
# stage: v3 block + qcri block, then shuffle rows in chunks
o_ids[0:n_v3] = tr
o_y[0:n_v3] = tr_y
o_ids[n_v3:] = q_ids
o_y[n_v3:] = q_y
o_ids.flush(); o_y.flush()
rng = random.Random(20260917)
perm = np.arange(n); rng.shuffle(perm)
import tempfile
tmp_ids = OUTD / "shuffle_ids.npy"
tmp_ys = OUTD / "shuffle_ys.npy"
sid = np.lib.format.open_memmap(str(tmp_ids), mode="w+", dtype=np.int64, shape=(n, CTX))
sy = np.lib.format.open_memmap(str(tmp_ys), mode="w+", dtype=np.int8, shape=(n, CTX))
CH = 100000
for s in range(0, n, CH):
    e = min(n, s + CH)
    take = perm[s:e]
    sid[s:e] = o_ids[take]
    sy[s:e] = o_y[take]
sid.flush(); sy.flush()
del o_ids, o_y
_gc = __import__('gc'); _gc.collect()
import shutil as _sh
if (OUTD / "train_ids.npy").exists():
    (OUTD / "train_ids.npy").unlink()
if (OUTD / "train_y.npy").exists():
    (OUTD / "train_y.npy").unlink()
_sh.copyfile(tmp_ids, OUTD / "train_ids.npy")
_sh.copyfile(tmp_ys, OUTD / "train_y.npy")
tmp_ids.unlink(); tmp_ys.unlink()
del q_ids, q_y, tr, tr_y
# val copied IDENTICAL from v3
import shutil
shutil.copyfile(V3 / "val_ids.npy", str(OUTD / "val_ids.npy"))
shutil.copyfile(V3 / "val_y.npy", str(OUTD / "val_y.npy"))
print("v3q DONE rows:", n, "v3:", n_v3, "qcri:", n_q)
