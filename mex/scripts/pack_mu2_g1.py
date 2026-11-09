"""mex/scripts/pack_mu2_g1.py — pack the G1 char-LM corpus (E-30).

Same stream contract as mex/scripts/pack.py (uint32 PackedDataset shards,
newline IN-vocab, blocks of ctx=96) but shards at ~2M ids so the train pool
spans many files like a full corpus. Uses the SAME CharVocab as x1 (ME-D1
shared vocab, id-compatible with the mu0 line and its eval harness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.vocab import CharVocab

CTX = 96
SHARD_TOKENS = 2_000_000
OUT = ROOT / "data" / "mex" / "mu2" / "g1" / "tokens"


def ids_of(f: Path, voc: CharVocab) -> list[int]:
    ids: list[int] = []
    with f.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            ids.extend(voc.encode(line + "\n"))
    return ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    voc = CharVocab()
    for split in ("val", "train"):
        ids = ids_of(ROOT / "data" / "mex" / "mu2" / "g1" / (split + ".txt"), voc)
        arr = np.asarray(ids, dtype=np.uint32)
        n_sh = 1 if split == "val" else (len(arr) + SHARD_TOKENS - 1) // SHARD_TOKENS
        for s in range(n_sh):
            chunk = arr[s * SHARD_TOKENS:(s + 1) * SHARD_TOKENS]
            chunk.tofile(OUT / (split + "_" + f"{s:04d}.bin"))
        print(f"{split}: {len(arr)} ids -> {n_sh} shard(s) = "
              f"{len(arr) // CTX} blocks")

if __name__ == "__main__":
    main()
