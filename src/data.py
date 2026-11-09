"""Fixed-length packed-token dataset over uint32 .bin shards (memmap-backed)."""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """Yields fixed-length token blocks from packed uint32 shards.

    Shards are memory-mapped and never fully loaded; block->shard resolution
    is precomputed so __getitem__ is O(1). labels == input_ids: the causal-LM
    head shifts labels internally (LlamaForCausalLM convention).
    """

    def __init__(self, shards, seq_len: int):
        self.seq_len = int(seq_len)
        if isinstance(shards, (str, Path)):
            shards = [shards]
        self.shards = sorted(Path(p) for p in shards)
        if not self.shards:
            raise FileNotFoundError("no .bin shards given to PackedDataset")
        self._maps = []
        self._ends = []  # cumulative block counts (per shard, inclusive)
        total = 0
        for p in self.shards:
            arr = np.memmap(p, dtype=np.uint32, mode="r")
            n_blocks = len(arr) // self.seq_len
            if n_blocks == 0:
                continue  # shard smaller than one block: skip, not crash
            self._maps.append(arr)
            total += n_blocks
            self._ends.append(total)
        self.n_blocks = total
        if self.n_blocks == 0:
            raise ValueError(f"packed shards hold fewer than seq_len={self.seq_len} tokens")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self.n_blocks
        if idx >= self.n_blocks:
            raise IndexError(idx)
        s = bisect_right(self._ends, idx)
        prev = self._ends[s - 1] if s > 0 else 0
        off = (idx - prev) * self.seq_len
        block = np.asarray(self._maps[s][off : off + self.seq_len], dtype=np.int64)
        return {"input_ids": block, "labels": block.copy()}
