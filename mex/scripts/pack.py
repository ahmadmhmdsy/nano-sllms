"""Pack each \u03bc0 task's raw .txt into uint32 PackedDataset shards.

src/data.py contract: shards are uint32 id streams; train.py loads
'train_*.bin' + 'val_*.bin' from data.tokens_dir and slices blocks of
seq_len = model.ctx. Lines are concatenated; newline chars are IN-vocab.
Block boundary = mid-task is fine: the causal LM learns the format either way.

NOTE (equal tokens, ME-D5): the control's train stream is the UNION of every
task's train + val text, so control train tokens == sum(expert train tokens)
+ sum(expert val tokens consumed by each expert's val_0000.bin) by
construction. gen_data caps are fixed (Tasks 2/3), so the control is ~4x
each expert only in PARAMETERS, not tokens; the \u03bc0 report lists the actual
token counts of the five runs side by side. Each expert's val_0000.bin stays
held-out REAL blocks (never seen in that expert's train); test.txt stays
unseen by packing entirely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.vocab import CharVocab

TASKS = ["x1", "x2", "x3", "x4"]
CTX = 96  # mu0 context window (model.ctx)


def pack(split_files: list[Path], out_prefix: Path, ctx: int) -> int:
    voc = CharVocab()
    ids: list[int] = []
    for f in split_files:
        ids.extend(voc.encode(f.read_text(encoding="utf-8")))
    arr = np.asarray(ids, dtype=np.uint32)
    shard = out_prefix  # single venue, tiny data
    arr.tofile(shard.with_suffix(".bin"))
    print(f"packed {shard.with_suffix('.bin')} : {arr.size} ids = {arr.size // ctx} blocks")
    return int(arr.size)


def main() -> None:
    for t in TASKS:
        src = ROOT / "data" / "mex" / t
        tdir = src / "tokens"  # tokens live beside raw under tokens_dir convention
        tdir.mkdir(parents=True, exist_ok=True)
        pack([src / "train.txt"], tdir / "train_0000", ctx=CTX)
        pack([src / "val.txt"] if (src / "val.txt").exists() else [src / "test.txt"],
             tdir / "val_0000", ctx=CTX)
    # control = union of every task's train + val text
    ctrl = ROOT / "data" / "mex" / "control"
    ctdir = ctrl / "tokens"
    ctdir.mkdir(parents=True, exist_ok=True)
    concat = []
    for t in TASKS:
        for k in ("train", "val"):
            p = ROOT / "data" / "mex" / t / f"{k}.txt"
            if p.exists():
                concat.append(p)
    pack(concat, ctdir / "train_0000", ctx=CTX)
    pack([ROOT / "data" / "mex" / t / "val.txt" for t in TASKS
          if (ROOT / "data" / "mex" / t / "val.txt").exists()],
         ctdir / "val_0000", ctx=CTX)


if __name__ == "__main__":
    main()
