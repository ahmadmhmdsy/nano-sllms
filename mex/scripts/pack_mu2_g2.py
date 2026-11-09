"""mex/scripts/pack_mu2_g2.py — pack G2 streams (E-31). Same contract as G1.

train shards: masked corpus (85%) + clean replay tail (15%) both matching the
train_*.bin glob so the sampler sees the 85/15 mix block-wise. val_task gets
the usual val_*.bin name for the in-train gate; the retention probe stays out
of tokens_dir (holdout_retent.bin) and is evaluated post-hoc.
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
OUT = ROOT / "data" / "mex" / "mu2" / "g2" / "tokens"
SRC = ROOT / "data" / "mex" / "mu2" / "g2"


def ids_of(f: Path, voc) -> list[int]:
    ids: list[int] = []
    with f.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            ids.extend(voc.encode(line + "\n"))
    return ids


def write(ids, prefix: str) -> None:
    arr = np.asarray(ids, dtype=np.uint32)
    n_sh = max(1, (len(arr) + SHARD_TOKENS - 1) // SHARD_TOKENS)
    for s in range(n_sh):
        arr[s * SHARD_TOKENS:(s + 1) * SHARD_TOKENS].tofile(
            OUT / (prefix + f"_{s:04d}.bin"))
    print(f"{prefix}: {len(arr)} ids -> {n_sh} shard(s) = {len(arr) // CTX} blocks")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    voc = CharVocab()
    write(ids_of(SRC / "train_mask.txt", voc), "train_mask")
    write(ids_of(SRC / "train_replay.txt", voc), "train_replay")
    write(ids_of(SRC / "val_task.txt", voc), "val_task")
    arr = np.asarray(ids_of(SRC / "holdout_retent.txt", voc), dtype=np.uint32)
    (SRC / "holdout_retent.bin").write_bytes(arr.tobytes())
    print(f"holdout_retent (post-hoc only): {arr.size} ids")


if __name__ == "__main__":
    main()
