"""Token shard records, disk read-back verification, and atomic publication."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from slm_train.contracts.hash import sha256_file


@dataclass(frozen=True)
class ShardRecord:
    """One token shard exactly as the published manifest certifies it."""

    path: str
    size_bytes: int
    sha256: str
    mask_path: "Optional[str]" = None
    mask_size_bytes: "Optional[int]" = None
    mask_sha256: "Optional[str]" = None


def _require(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, found {actual!r}")


def verify_shard_records(root: "Path", shard_records: "list[ShardRecord]") -> None:
    """Re-read every published shard (and mask) and fail closed on any
    size or hash mismatch, on a missing file, or on a torn publication."""
    for index, record in enumerate(shard_records):
        shard_path = Path(root) / record.path
        if not shard_path.is_file():
            raise ValueError(f"the published token shard is missing: {record.path}")
        _require(shard_path.stat().st_size, record.size_bytes,
                 f"shard size read-back failed for {record.path}")
        _require(sha256_file(shard_path), record.sha256,
                 f"shard hash read-back failed for {record.path}")
        if record.mask_path is None:
            continue
        mask_path = Path(root) / record.mask_path
        if not mask_path.is_file():
            raise ValueError(f"the published loss mask is missing: {record.mask_path}")
        _require(mask_path.stat().st_size, record.mask_size_bytes,
                 f"mask size read-back failed for {record.mask_path}")
        _require(sha256_file(mask_path), record.mask_sha256,
                 f"mask hash read-back failed for {record.mask_path}")


def verify_ordered_shard_list(
    records: "list[ShardRecord]", manifest_shards: "list[dict[str, object]]") -> None:
    """The ordered shard list is exactly what the manifest certifies."""
    _require(len(manifest_shards), len(records), "ordered shard count")
    for index, (record, entry) in enumerate(zip(records, manifest_shards)):
        if record.path != str(entry.get("path")) or record.sha256 != str(entry.get("sha256")):
            raise ValueError(f"the ordered shard list disagrees at index {index}")
        if record.mask_path != entry.get("mask_path"):
            raise ValueError(f"the ordered mask list disagrees at index {index}")


def publish_shards(temp_dir: "Path", final_dir: "Path") -> "Path":
    """Rename an assembled publication into place; never overwrite."""
    import os
    import shutil
    try:
        os.rename(temp_dir, final_dir)
    except FileExistsError as error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileExistsError(
            f"a published token generation is never overwritten: {final_dir}") from error
    except OSError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return final_dir
