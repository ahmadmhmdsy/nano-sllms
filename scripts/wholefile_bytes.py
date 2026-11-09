"""Stream whole Python files from an UNGATED GitHub corpus into ByteTokenizer id shards.

Source: codeparrot/codeparrot-clean (whole python files, deduplicated, quality-filtered). The
bigcode stacks are gated and this token has no grant for them (probe 2026-10-10: the-stack-smol,
starcoderdata and the-stack-dedup all raise DatasetNotFoundError "is a gated dataset"; the
md_measure study streamed starcoderdata successfully earlier, so the grant has since lapsed).

E-121 arm B (corpus SHAPE). The E-117 byte arm read single functions extracted from CodeSearchNet
(163,296,228 bytes, one function per row), so imports, classes, several functions per file and
__main__ blocks were never in the data. This builder keeps the WHOLE file: same byte budget,
different shape. Byte id = 4 + utf-8 byte (glm53_flash.tokenizer.ByteTokenizer).

CPU + network only. Regenerable: safe to delete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BYTE_OFFSET = 4  # ByteTokenizer.byte_offset
VAL_EVERY = 50   # 1 row in 50 goes to val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "wholefile" / "bytes")
    parser.add_argument("--dataset", default="codeparrot/codeparrot-clean",
                        help="whole-file python corpus; the bigcode stacks are gated (see docstring)")
    parser.add_argument("--data-dir", default="", help="config/data_dir; empty for single-config datasets")
    parser.add_argument("--target-train-bytes", type=int, default=163_296_228)
    parser.add_argument("--target-val-bytes", type=int, default=3_456_004)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import prepare_data  # noqa: E402  (module import runs _load_dotenv)
    from datasets import load_dataset  # noqa: E402

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit("refusing to overwrite %s (pass --force)" % args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    kwargs = {"split": "train", "streaming": True, "token": os.environ.get("HF_TOKEN")}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    stream = load_dataset(args.dataset, **kwargs)
    started = time.time()
    rows = kept = 0
    train_bytes = val_bytes = 0
    val_handle = (args.out / "val.bin").open("wb")
    train_handle = (args.out / "train.bin").open("wb")
    try:
        for row in stream:
            rows += 1
            try:
                text = prepare_data.text_of(row)
            except Exception:
                continue
            if not text or len(text) < args.min_chars:
                continue
            values = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET
            if kept % VAL_EVERY == 0 and val_bytes < args.target_val_bytes:
                values.tofile(val_handle)
                val_bytes += int(values.size)
            else:
                values.tofile(train_handle)
                train_bytes += int(values.size)
            kept += 1
            if kept % args.log_every == 0:
                print(json.dumps({"event": "progress", "rows": rows, "kept": kept,
                                  "train_bytes": train_bytes, "val_bytes": val_bytes,
                                  "seconds": round(time.time() - started, 1)}), flush=True)
            if train_bytes >= args.target_train_bytes and val_bytes >= args.target_val_bytes:
                break
    finally:
        train_handle.close()
        val_handle.close()

    meta = {
        "source": args.dataset, "data_dir": args.data_dir, "byte_offset": BYTE_OFFSET,
        "shape": "whole-file", "min_chars": args.min_chars, "val_every": VAL_EVERY,
        "rows_scanned": rows, "rows_kept": kept,
        "train": {"bytes": train_bytes, "path": "train.bin"},
        "val": {"bytes": val_bytes, "path": "val.bin"},
        "seconds": round(time.time() - started, 1),
        "epochs_at_5000_steps": {"byte_arm_positions": 5000 * 32 * 1024,
                                 "epochs": (5000 * 32 * 1024) / max(train_bytes, 1)},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "done", **meta}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
