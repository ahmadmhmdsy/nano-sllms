"""E-23b S2 part 2 (disk-light) — pack classical rows into v3t = v3q train + classical windows.

Same discipline as e23_pack_zm: rows ({"b","l"}) staged in classical_rows.jsonl
are tokenized straight into two in-RAM arrays, concatenated with v3q train
memmaps, single-pass shuffled, val copied IDENTICAL from v3q.
CPU only; peak disk = final npy pair.
"""
import json, sys
from pathlib import Path
import numpy as np, random

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import tokenizer as TK

CTX = 128
V3Q = REPO / "data" / "diac" / "v3q" / "tokens"
OUTD = REPO / "data" / "diac" / "v3t" / "tokens"
OUTD.mkdir(parents=True, exist_ok=True)

rows = []
for line in open(REPO / "data/diac/v3t/classical_rows.jsonl", encoding="utf-8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    b = r.get("b")
    if not b:
        continue
    rows.append((b, r["l"]))
    if len(rows) % 100000 == 0:
        print("loaded", len(rows), flush=True)
print("staged rows:", len(rows), flush=True)

t_ids = np.zeros((len(rows), CTX), dtype=np.int64)
t_ys = np.full((len(rows), CTX), -1, dtype=np.int8)
for i, (b, l) in enumerate(rows):
    seq = TK.encode(b)[:CTX]
    t_ids[i, :len(seq)] = seq
    t_ys[i, :len(seq)] = l[:len(seq)]
n_t = len(rows)
del rows

tr = np.load(str(V3Q / "train_ids.npy"), mmap_mode="r")
tr_y = np.load(str(V3Q / "train_y.npy"), mmap_mode="r")
n_v3q = len(tr)
n = n_v3q + n_t
print("v3q:", n_v3q, "classical:", n_t, "-> total:", n, flush=True)

o = np.lib.format.open_memmap(str(OUTD / "train_ids.npy"), mode="w+", dtype=np.int64, shape=(n, CTX))
oy = np.lib.format.open_memmap(str(OUTD / "train_y.npy"), mode="w+", dtype=np.int8, shape=(n, CTX))
rng = random.Random(20260918)
perm = np.arange(n)
rng.shuffle(perm)
CH = 100000
for s0 in range(0, n, CH):
    e = min(n, s0 + CH)
    take = perm[s0:e]
    mv = take < n_v3q
    o[s0:e][mv] = tr[take[mv]]
    o[s0:e][~mv] = t_ids[take[~mv] - n_v3q]
    oy[s0:e][mv] = tr_y[take[mv]]
    oy[s0:e][~mv] = t_ys[take[~mv] - n_v3q]
    o.flush()
    oy.flush()
    print("chunk", e, flush=True)

import shutil
shutil.copyfile(V3Q / "val_ids.npy", str(OUTD / "val_ids.npy"))
shutil.copyfile(V3Q / "val_y.npy", str(OUTD / "val_y.npy"))
print("v3t DONE rows:", n, flush=True)
