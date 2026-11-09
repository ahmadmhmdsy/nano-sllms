"""Precompute teacher-ensemble logits for the ENTIRE union dataset.

MU1 arm D (EXPERIMENTS row E-29). Writes an .npy memmap:

    shape = [n_train_blocks, ctx, vocab]   row i == block i of PackedDataset

aligned with the packed uint32 train tokens (data/mex/control/tokens/
train_0000.bin) so the per-block lookup in mex/scripts/train_distill.py is
O(1). The teachers are the four 12K expert finals under
runs/mex/archive_12000/{x1..x4}/final, loaded through the mex_x{1..4}
configs via src.model.build_model + load_finetune_init, eval() + no_grad;
their logits are averaged in fp32 per block, exactly as
mex.src.kd.TeacherEnsemble.ensemble_logits computes.

Determinism: a FIXED block batch size (--batch-blocks, default 64) walks
the sorted train shards left to right; identical input files yield
identical block batching, so re-runs produce byte-identical rows (no
shuffling anywhere in this script).

Disk gate: fp32 costs n_blocks * ctx * vocab * 4 bytes (~1.28 GB for the
3.29M-token union control data). Default --dtype auto picks fp32 when more
than 2 GB of headroom remains after the write (checked with
shutil.disk_usage), else fp16 (~0.64 GB).

Usage:
  .venv/Scripts/python.exe mex/scripts/distill_cache.py \
        --config configs/mex_distill.yaml [--dtype fp32|fp16|auto] \
        [--max-blocks N]   # N > 0 only for smoke caches
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FP32_HEADROOM_SAMPLES = 2_000_000_000


def free_bytes_after(path: Path, write_bytes: int) -> int:
    """Headroom left on a path's volume after writing write_bytes."""
    _total, _used, free = shutil.disk_usage(str(path))
    return int(free) - int(write_bytes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="path to configs/mex_distill.yaml")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16"])
    ap.add_argument("--batch-blocks", type=int, default=64,
                    help="fixed teacher batch size (determinism: keep fixed)")
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="0 = every train block; N > 0 = smoke caches only")
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from transformers import AutoTokenizer

    from mex.src.kd import TeacherEnsemble
    from src.data import PackedDataset     # the EXACT training dataset class

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    kd_cfg = cfg["kd"]
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"]["name"])
    seq_len = int(cfg["model"]["ctx"])
    vocab = max(int(cfg["tokenizer"]["vocab_size"]), len(tok))

    shards = (ROOT / cfg["data"]["tokens_dir"]).glob("train_*.bin")
    train_ds = PackedDataset(shards, seq_len)
    n_blocks = len(train_ds)
    limit = min(args.max_blocks, n_blocks) if args.max_blocks > 0 else n_blocks
    itemsize = 2 if args.dtype == "fp16" else 4
    write_bytes = n_blocks * seq_len * vocab * itemsize
    print(f"[cache] blocks total={n_blocks} to_cache={limit} "
          f"seq_len={seq_len} vocab={vocab}", flush=True)

    # Disk gate BEFORE allocating the memmap (~1.28 GB fp32 here).
    dtype = args.dtype
    if dtype == "auto":
        dtype = ("fp32" if free_bytes_after(ROOT, write_bytes)
                 > FP32_HEADROOM_SAMPLES else "fp16")
        print(f"[cache] dtype auto -> {dtype}", flush=True)
    np_dtype = {"fp32": np.float32, "fp16": np.float16}[dtype]
    out_path = ROOT / kd_cfg["teacher_cache"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np_dtype,
        shape=(n_blocks, seq_len, vocab))
    cache[:] = 0   # rows beyond --max-blocks stay 0 = miss/pad semantics

    # Teachers on CPU fp32: the GPU budget belongs to the student only.
    configs = [yaml.safe_load((ROOT / p).read_text(encoding="utf-8"))
               for p in kd_cfg["teacher_configs"]]
    final_dirs = [str(ROOT / kd_cfg["teacher_root"] / e / "final")
                  for e in kd_cfg["teacher_experts"]]
    ensemble = TeacherEnsemble(configs, final_dirs, device="cpu")

    bs = max(1, int(args.batch_blocks))
    done = 0
    for start in range(0, limit, bs):
        end = min(start + bs, limit)
        ids = torch.stack([
            torch.as_tensor(train_ds[i]["input_ids"].copy())
            for i in range(start, end)])
        logits = ensemble.ensemble_logits(ids)      # fixed batch: [b, L, V]
        cache[start:end] = logits.numpy().astype(np_dtype)
        done += end - start
        if done % (bs * 16) == 0 or end == limit:
            print(f"[cache] {done}/{limit} rows", flush=True)
    cache.flush()
    del cache
    print(f"[cache] wrote {out_path} rows={done} dtype={dtype} "
          f"bytes={write_bytes / 1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
