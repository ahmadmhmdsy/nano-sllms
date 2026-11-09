import json, sys, os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = "data/diac"

def cmp_prefix(new, ref, label, cut, chunk=200000):
    A = np.load(new, mmap_mode="r"); B = np.load(ref, mmap_mode="r")
    print(f"{label}: new {A.shape} dtype {A.dtype} | ref {B.shape} dtype {B.dtype}")
    n = min(A.shape[0], B.shape[0]); bad = 0; first = None
    for i in range(0, n, chunk):
        x = np.asarray(A[i:i+chunk, :cut]); y = np.asarray(B[i:i+chunk])
        d = x != y; c = int(d.sum())
        if c:
            bad += c
            if first is None:
                idx = np.argwhere(d)[:3]
                first = [(int(i+r), int(cc), int(x[r,cc]), int(y[r,cc])) for r,cc in idx]
    print(f"  prefix mismatch = {bad} cells over {n} rows; first {first}")
    if A.shape[0] != B.shape[0]:
        print(f"  ROW COUNT DIFFERS: {A.shape[0]} vs {B.shape[0]}")
    return bad == 0 and A.shape[0] == B.shape[0] and A.shape[1] == 256

ok = True
for name in ["train_ids", "train_y"]:
    ok = cmp_prefix(f"{D}/v3ctx256/tokens/{name}.npy", f"{D}/v3/tokens/{name}.npy", "TRAIN " + name, 128) and ok
for name in ["val_ids", "val_y"]:
    ok = cmp_prefix(f"{D}/v3ctx256/tokens/{name}.npy", f"{D}/v3/tokens/{name}.npy", "VAL " + name, 128) and ok

print("=== label range (PAD convention) ===")
for name in ["train_y", "val_y"]:
    a = np.load(f"{D}/v3ctx256/tokens/{name}.npy", mmap_mode="r")
    mn, mx, nneg = 10**9, -10**9, 0
    for i in range(0, a.shape[0], 200000):
        c = np.asarray(a[i:i+200000])
        mn = min(mn, int(c.min())); mx = max(mx, int(c.max())); nneg += int((c < 0).sum())
    print(f"  {name}: min {mn} max {mx} negative cells {nneg}")

print("=== real text beyond col 128 ===")
a = np.load(f"{D}/v3ctx256/tokens/train_ids.npy", mmap_mode="r")
tot = 0; rows = 0
for i in range(0, a.shape[0], 200000):
    c = np.asarray(a[i:i+200000, 128:])
    m = c != 0
    tot += int(m.sum()); rows += int(m.any(axis=1).sum())
print(f"  train_ids: {tot} nonzero cells beyond 128; {rows} of {a.shape[0]} rows carry real text there")
print("ACCEPTANCE:", "PASS" if ok else "FAIL")
