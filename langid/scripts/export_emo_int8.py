"""DA-2 int8 export + numpy latency probe + per-language agreement.

Usage: python langid/scripts/export_emo_int8.py [--model runs/langid_da2/emo_best.pt]
Writes runs/langid_da2/emo_int8.npz and prints size + agreement + latency.
"""
import argparse
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model
from langid.src.emo_model import encode_batch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data", "langid", "emo")
OUT = os.path.join(BASE, "runs", "langid_da2")
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(OUT, "emo_best.pt"))
    ap.add_argument("--out", default=os.path.join(OUT, "emo_int8.npz"))
    args = ap.parse_args()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    W = ck["emb"].float().numpy()
    B = ck["bias"].float().numpy()
    scale = np.abs(W).max(axis=0) / 127.0
    scale[scale == 0] = 1e-9
    Wq = np.round(W / scale).astype(np.int8)
    np.savez_compressed(args.out, Wq=Wq, scale=scale, bias=B.astype(np.float32),
             labels=np.array(ck["labels"], dtype=object))
    size = os.path.getsize(args.out)
    print("int8 art MiB:", round(size / 1048576, 3), "(bar <= 3.0)")

    rows = []
    with open(os.path.join(DATA, "test.tsv"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), "\t".join(parts[2:])))

    K = len(ck["labels"])
    for lang_code in ["ar", "en"]:
        sub = [r for r in rows if r[0] == lang_code][:1000]
        agree = 0
        n = 0
        t0 = time.perf_counter()
        for _, _, t in sub:
            ids, offs = encode_batch([t])
            blist = ids.tolist()
            s32 = np.zeros(K, dtype=np.float64)
            s8 = np.zeros(K, dtype=np.float64)
            for b in blist:
                w32 = W[b]
                s32 += w32
                s8 += Wq[b] * scale
            p32 = int(np.argmax(s32 + B))
            p8 = int(np.argmax(s8 + B))
            agree += (p32 == p8)
            n += 1
        dt = (time.perf_counter() - t0) / max(n, 1)
        print(lang_code, "agree:", agree, "/", n, "ms/text(python-loop proxy):", round(dt * 1000, 4))


if __name__ == "__main__":
    main()
