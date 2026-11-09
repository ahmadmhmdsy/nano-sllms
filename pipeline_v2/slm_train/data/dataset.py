"""PackedDataset: an explicit memmap-backed packed token dataset."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from slm_train.contracts.hash import canonical_sha256
from slm_train.contracts.tokenized import TokenizedDataManifest
from slm_train.data.packing import MASK_UINT8, TOKEN_UINT32
from slm_train.data.shards import ShardRecord, verify_shard_records
from slm_train.data.tokenizer import (
    TokenizerError, assert_unpinned, reopen_identity, tokenizer_from_record,
)


class DatasetError(ValueError):
    """Fail-closed packed-dataset construction failure."""


class PackedDataset(Dataset):
    """Validated, memmap-backed packed token dataset."""

    def __init__(self, manifest, sequence_length, shard_paths=None, mask_paths=None):
        if not isinstance(manifest, TokenizedDataManifest):
            raise DatasetError(
                "the aligned tokenized-data manifest is required: FAIL")
        self.manifest = manifest
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise DatasetError("sequence_length must be positive: FAIL")
        if int(manifest.sequence_length) != self.sequence_length:
            raise DatasetError(
                "the manifest sequence length disagrees with the requested one: FAIL")
        self.mask_paths = list(mask_paths or [None for _ in manifest.shards])
        # prerequisites are validated BEFORE any memmap is constructed (M5)
        if manifest.packing_mode == "document_masked" and any(
                path is None for path in self.mask_paths):
            raise DatasetError("document_masked requires one loss-mask file per shard")
        self.shard_paths = list(shard_paths or [Path(item.path) for item in manifest.shards])
        self.shards = [np.memmap(path, dtype=TOKEN_UINT32, mode="r") for path in self.shard_paths]
        self.masks = [
            (np.memmap(path, dtype=MASK_UINT8, mode="r") if path is not None else None)
            for path in self.mask_paths
        ]
        self.blocks = sum(len(shard) // self.sequence_length for shard in self.shards)
        if int(manifest.usable_blocks) != self.blocks:
            raise DatasetError("the recorded usable block count disagrees: FAIL")

    def __len__(self):
        return self.blocks

    def __getitem__(self, index):
        if index < 0:
            index += self.blocks
        if index < 0 or index >= self.blocks:
            raise IndexError(index)
        remaining = int(index)
        for shard_index, shard in enumerate(self.shards):
            count = len(shard) // self.sequence_length
            if remaining < count:
                offset = remaining * self.sequence_length
                block = np.asarray(
                    shard[offset:offset + self.sequence_length], dtype=np.int64)
                mask_shard = self.masks[shard_index]
                if mask_shard is None:
                    return {
                        "input_ids": block, "labels": block.copy(),
                        "loss_mask": np.ones_like(block, dtype=np.uint8),
                    }
                mask = np.asarray(
                    mask_shard[offset:offset + self.sequence_length], dtype=np.uint8)
                return {"input_ids": block, "labels": block.copy(), "loss_mask": mask.copy()}
            remaining -= count
        raise IndexError(index)

    @classmethod
    def from_manifest(
        cls, manifest_path, sequence_length: int = 1,
        *, expected_generation=None, expected_tokenizer_hash=None,
    ) -> "PackedDataset":
        """Validate the published token manifest and open its shards."""
        manifest_path = Path(manifest_path)
        record = _load_tokens_record(manifest_path)
        parent = manifest_path.parent
        tokenizer_record = dict(record.get("tokenizer", {}))
        manifest, extra_record = tokenizer_from_record(tokenizer_record)
        try:
            assert_unpinned(reopen_identity(manifest, extra_record))
        except TokenizerError as error:
            raise DatasetError("the tokenized manifest tokenizer is unpinned: FAIL") from error
        if expected_tokenizer_hash is not None and canonical_sha256(manifest.model_dump()) != expected_tokenizer_hash:
            raise DatasetError("the recorded tokenizer identity disagrees with the expected pinning: FAIL")
        if int(record["sequence_length"]) != int(sequence_length):
            raise DatasetError(
                "the manifest sequence length disagrees with the requested one: FAIL")
        shard_paths, mask_paths = [], []
        for shard_record in record.get("shards", []):
            resolved = parent / str(shard_record["path"])
            if not resolved.is_file():
                raise DatasetError(f"the tokenized shard is missing: {shard_record['path']}")
            shard_paths.append(str(resolved))
            mask_relative = shard_record.get("mask_path")
            if mask_relative is not None:
                mask_resolved = parent / str(mask_relative)
                if not mask_resolved.is_file():
                    raise DatasetError(f"the recorded loss mask is missing: {mask_relative}")
                mask_paths.append(str(mask_resolved))
            else:
                mask_paths.append(None)
        if record.get("packing_mode") == "document_masked" and not all(mask_paths):
            raise DatasetError("document_masked requires one loss-mask file per shard")
        # the labeled validation is wired into the real consumption path:
        # completion marker (including the primary manifest binding),
        # tokenized-manifest cross-check, shard/mask size/hash read-back,
        # counts identity, and tokenizer vocabulary all fail closed here.
        from slm_train.data.validate import validate_packed_tokens
        validate_packed_tokens(
            manifest_path,
            sequence_length=int(record["sequence_length"]),
            expected_generation=expected_generation,
            expected_tokenizer_hash=expected_tokenizer_hash,
            strict=True,
        )
        tokenized = TokenizedDataManifest(
            schema_version=1,
            shards=tuple(
                {"schema_version": 1, "path": str(shard["path"]),
                 "size_bytes": int(shard["size_bytes"]), "sha256": str(shard["sha256"])}
                for shard in record.get("shards", [])
            ),
            mask_path=(str(record["shards"][0].get("mask_path"))
                       if any(shard.get("mask_path") for shard in record.get("shards", []))
                       else None),
            sequence_length=int(record["sequence_length"]),
            packing_mode=str(record["packing_mode"]),
            usable_blocks=int(dict(record.get("counts", {})).get("usable_blocks", 0)),
            dropped_tail_tokens=int(dict(record.get("counts", {})).get("dropped_tail_tokens", 0)),
            overflow_blocks=int(dict(record.get("counts", {})).get("overflow_blocks", 0)),
            source_data_hash=str(record.get("source_data_hash", "")),
        )
        # the recorded shards verify by read-back; open them resolved
        verify_shard_records(parent, [
            ShardRecord(path=str(parent / s["path"]), size_bytes=int(s["size_bytes"]),
                        sha256=str(s["sha256"]),
                        mask_path=(str(parent / s["mask_path"]) if s.get("mask_path") else None),
                        mask_size_bytes=(int(s["mask_size_bytes"]) if s.get("mask_size_bytes") is not None else None),
                        mask_sha256=(str(s["mask_sha256"]) if s.get("mask_sha256") is not None else None))
            for s in record.get("shards", [])
        ])
        dataset = cls(tokenized, int(record["sequence_length"]),
                      shard_paths=shard_paths, mask_paths=mask_paths)
        return dataset


def _load_tokens_record(manifest_path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise DatasetError(
            f"the tokenized manifest is missing: {manifest_path} "
            "(a completion marker requires it before training)")
    try:
        value = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise DatasetError(f"the tokenized manifest is unreadable: {manifest_path} ({error})") from error
    if not isinstance(value, dict):
        raise DatasetError("the tokenized manifest is not a JSON object: FAIL")
    if value.get("kind") != "tokens-packed":
        # wrong file or a generation that was never completed: fail closed on completion
        raise DatasetError(
            f"the manifest kind is not tokens-packed and its completion marker does not "
            f"certify a token generation: FAIL")
    record = {**value}
    shards = record.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DatasetError(
            "the tokenized manifest records no shard list: the completion marker "
            "requires it before training")
    for shard_record in shards:
        rel = str(shard_record.get("path", ""))
        if not rel or ".." in Path(rel).parts or Path(rel).is_absolute():
            raise DatasetError(f"the recorded shard path is not relative: {rel!r}")
    for key in ("generation", "source_data_hash", "sequence_length", "packing_mode", "tokenizer"):
        if key not in record:
            raise DatasetError(f"the tokenized manifest does not record {key}: FAIL")
    if "usable_blocks" not in record and "usable_blocks" not in record.get("counts", {}):
        raise DatasetError("the tokenized manifest does not record usable_blocks: FAIL")
    return record
