"""Pair-aligned mount dataset: student uint32 blocks zip with the teacher
per-block uint32/len blocks produced by scripts/mount_teacher_stream.py.
Asserts equal block counts at init; collate-friendly __getitem__ returns
input_ids (student; labels implicit - the causal-LM head shifts internally),
teacher_ids and teacher_pad_mask (True = PAD, exactly at positions
>= teacher_len[i]; pads are never attended).

Runtime contract note (Task 3 brief Step 3.2 adaptation): this repo's
PackedDataset.__getitem__ returns a dict {"input_ids", "labels"} of int64
numpy arrays (src/data.py), not a bare array - unwrap "input_ids".
Block counts are derived from the .bin sizes themselves (meta.json may hold
stale single-split stats).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import PackedDataset


class MountDataset(Dataset):
    def __init__(self, student_dir: Path, teacher_dir: Path, split: str,
                 seq_len: int):
        self.student = PackedDataset(
            sorted(Path(student_dir).glob(split + "_*.bin")), seq_len)
        # glob("train_*.bin") ALSO matches train_*.len.bin (the brief's
        # literal code collided live: a uint32 read of a len shard is
        # garbage) - explicit .len.bin exclusion filter.
        self.t_files = [f for f in sorted(Path(teacher_dir).glob(split + "_*.bin"))
                        if not f.name.endswith(".len.bin")]
        self.len_files = sorted(Path(teacher_dir).glob(split + "_*.len.bin"))
        n_s = len(self.student)
        blocks_t = [np.fromfile(f, dtype=np.uint32).reshape(-1, seq_len)
                    for f in self.t_files]
        self.teacher = np.concatenate(blocks_t, axis=0)
        lens = np.concatenate([np.fromfile(f, dtype=np.int16)
                               for f in self.len_files])
        assert len(self.teacher) == n_s == len(lens), "block count mismatch"
        self.teacher_lens = lens

    def __len__(self):
        return len(self.student)

    def __getitem__(self, i):
        s = self.student[i]
        if isinstance(s, dict):          # PackedDataset contract, runtime-verified
            s = s["input_ids"]
        t = self.teacher[i]
        m = np.arange(len(t)) >= int(self.teacher_lens[i])
        return {"input_ids": torch.from_numpy(np.asarray(s, dtype=np.int64)),
                "teacher_ids": torch.from_numpy(np.asarray(t, dtype=np.int64)),
                "teacher_pad_mask": torch.from_numpy(m)}