"""Detokenize the parent target corpus into byte-id shards for the glm arch (E-117).

Reads data/target/tokens/*.bin (CodeLlama uint32 ids, packed by scripts/tokenize_data.py),
decodes with the tokenizer stored in runs/target/final, re-encodes the text as
ByteTokenizer ids (id = 4 + utf-8 byte) and writes single-file shards
data/target/bytes/{train,val}.bin plus a meta receipt carrying the measured
bytes-per-token and the target model's bits-per-byte conversion.

CPU only; no network (the tokenizer is already on disk). Regenerable: safe to delete.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "target" / "tokens"
BYTES = ROOT / "data" / "target" / "bytes"
TOKENIZER_DIR = ROOT / "runs" / "target" / "final"
BYTE_OFFSET = 4  # glm53_flash.tokenizer.ByteTokenizer.byte_offset
SPECIAL_BELOW = 3  # ByteTokenizer reserves 0 pad / 1 bos / 2 eos

# Recorded in runs/target/final/train_summary.json (the on-disk eval_report is stale).
FINAL_VAL_LOSS = 1.8641137587974261
BEST_VAL_LOSS = 1.851243257522583
BEST_STEP = 4000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing bytes dir")
    args = parser.parse_args()

    meta = json.loads((TOKENS / "meta.json").read_text(encoding="utf-8"))
    val_tokens = int(meta["val_tokens"])
    train_tokens = int(meta["train_tokens"])

    if BYTES.exists() and any(BYTES.iterdir()) and not args.force:
        raise SystemExit("refusing to overwrite %s (pass --force)" % BYTES)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    try:
        tokenizer.clean_up_tokenization_spaces = False
    except Exception as error:  # version drift: report, do not fail
        print(json.dumps({"warning": "clean_up_tokenization_spaces", "error": str(error)}), flush=True)

    def ids_of(name):
        path = TOKENS / name
        size = path.stat().st_size
        if size % 4:
            raise SystemExit("%s size %d is not a multiple of 4 (uint32 ids)" % (path, size))
        return np.memmap(path, dtype=np.uint32, mode="r")

    def byte_values(text):
        raw = text.encode("utf-8")
        return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET

    BYTES.mkdir(parents=True, exist_ok=True)

    # --- val first: it is small and yields the bytes-per-token / BPB headline ---
    val_ids = ids_of("val_000.bin")
    val_text = tokenizer.decode(val_ids.tolist(), skip_special_tokens=True)
    val_values = byte_values(val_text)
    val_values.tofile(BYTES / "val.bin")
    clean = [int(v) for v in val_ids.tolist() if int(v) >= SPECIAL_BELOW]
    reencoded = tokenizer(val_text, add_special_tokens=False)["input_ids"]
    compared = min(len(clean), len(reencoded))
    matched = sum(1 for a, b in zip(clean[:compared], reencoded[:compared]) if a == b)
    val_bytes = int(val_values.size)
    bytes_per_token = val_bytes / val_tokens
    report = {
        "tokenizer": str(TOKENIZER_DIR.relative_to(ROOT)),
        "byte_offset": BYTE_OFFSET,
        "source": str(TOKENS.relative_to(ROOT)),
        "val": {
            "tokens": val_tokens,
            "bytes": val_bytes,
            "bytes_per_token": bytes_per_token,
            "roundtrip_clean_tokens": len(clean),
            "roundtrip_reencoded": len(reencoded),
            "roundtrip_compared": compared,
            "roundtrip_match_fraction": (matched / compared) if compared else 0.0,
        },
        "target_loss_conversion": {
            "final_val_loss_nats_per_token": FINAL_VAL_LOSS,
            "final_bits_per_byte": FINAL_VAL_LOSS / math.log(2) / bytes_per_token,
            "best_val_loss_nats_per_token": BEST_VAL_LOSS,
            "best_bits_per_byte": BEST_VAL_LOSS / math.log(2) / bytes_per_token,
            "best_step": BEST_STEP,
        },
        "shards": [],
    }
    print(json.dumps({"event": "val", **report["val"], **report["target_loss_conversion"]}, sort_keys=True), flush=True)

    # --- train shards, concatenated in name order into one file ---
    names = sorted(p.name for p in TOKENS.glob("train_*.bin"))
    if not names:
        raise SystemExit("no train shards found under %s" % TOKENS)
    total = 0
    with (BYTES / "train.bin").open("wb") as handle:
        for name in names:
            ids = ids_of(name)
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            values = byte_values(text)
            values.tofile(handle)
            total += int(values.size)
            row = {"name": name, "tokens": int(ids.size), "bytes": int(values.size)}
            report["shards"].append(row)
            print(json.dumps({"event": "train_shard", **row}, sort_keys=True), flush=True)
            del ids, text, values

    report["train"] = {"tokens": train_tokens, "bytes": total, "shards": len(names)}
    report["epochs_at_5000_steps"] = {"byte_arm_positions": 5000 * 32 * 1024,
                                      "byte_arm_epochs": (5000 * 32 * 1024) / total if total else 0.0}
    (BYTES / "meta.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "done", "train_bytes": total, "val_bytes": val_bytes,
                      "bytes_per_token": bytes_per_token, "meta": str(BYTES / "meta.json")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
