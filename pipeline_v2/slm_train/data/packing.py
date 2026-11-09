"""Explicit packing modes: concat and document_masked, both fail-closed.

concat preserves the source token order. document_masked keeps document
boundaries and one uint8 loss-mask bit per token; a backend that cannot
honor that contract is a labeled FAIL before training, never a silent
downgrade to concat. The final incomplete block is either explicitly
retained (padded with the recorded pad token and reported) or counted in
dropped_tail_tokens; it is never silently lost.
"""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from slm_train.artifacts.manifest import write_manifest
from slm_train.artifacts.markers import verify_completion_marker, write_completion_marker
from slm_train.artifacts.paths import V2Layout
from slm_train.contracts.hash import sha256_bytes, sha256_file, canonical_sha256
from slm_train.contracts.tokenized import TokenizedDataManifest, TokenizedShard
from slm_train.data.shards import ShardRecord, publish_shards, verify_shard_records
from slm_train.data.tokenizer import (
    TokenizerError,
    TokenizerIdentity,
    assert_unpinned,
    manifest_json,
    validate_token_ids,
)

TOKEN_UINT32 = np.dtype("<u4")
MASK_UINT8 = np.dtype("<u1")
PACKING_MODES = ("concat", "document_masked")
TOKENS_DIR = V2Layout.tokens
TOKENIZED_MANIFEST_NAME = "tokenized-manifest.json"
PREPARED_MANIFEST_NAME = "tokens-prepared-manifest.json"


class PackingError(TokenizerError):
    """Fail-closed packing failure, labeled FAIL by default."""


@dataclass(frozen=True)
class _Segment:
    tokens: "Sequence[int]"
    mask: "Optional[Sequence[int]]"


@dataclass(frozen=True)
class PackResult:
    counts: "dict[str, object]"
    shards: "list[ShardRecord]"
    manifest: "dict[str, object]"
    manifest_path: Path
    published_path: Path
    tokenizer: "Optional[TokenizerIdentity]" = None

    def tokenized_manifest(self) -> TokenizedDataManifest:
        return _tokenized_manifest_from_dict(self.manifest)


def tokenized_manifest_from_record(value: "dict[str, object]") -> TokenizedDataManifest:
    """Convert a PUBLISHED record into the strict contract.

    The published record is deliberately richer than the contract: it carries a
    kind, per-shard mask facts, and counts. A caller that needs the contract must
    go through here rather than validating the record directly, which fails on
    exactly those extra keys.
    """
    return _tokenized_manifest_from_dict(value)


def _tokenized_manifest_from_dict(value: "dict[str, object]") -> TokenizedDataManifest:
    counts = dict(value.get("counts", {}) or {})
    shards: "list[dict[str, object]]" = [
        {
            "schema_version": 1,
            "path": str(shard.get("path")),
            "size_bytes": int(shard["size_bytes"]),
            "sha256": str(shard["sha256"]),
        }
        for shard in value.get("shards", [])
    ]
    mask_paths = {str(shard.get("mask_path")) for shard in value.get("shards", [])
                  if shard.get("mask_path") is not None}
    record = {
        "schema_version": 1,
        "shards": tuple(shards),
        "mask_path": (str(sorted(mask_paths)[0]) if mask_paths else value.get("mask_path")),
        "sequence_length": int(value["sequence_length"]),
        "packing_mode": str(value["packing_mode"]),
        "usable_blocks": int(value.get("usable_blocks", counts.get("usable_blocks", 0))),
        "dropped_tail_tokens": int(value.get(
            "dropped_tail_tokens", counts.get("dropped_tail_tokens", 0))),
        "overflow_blocks": int(value.get(
            "overflow_blocks", counts.get("overflow_blocks", 0))),
        "source_data_hash": str(value.get("source_data_hash", "")),
    }
    return TokenizedDataManifest.model_validate(record)


def _plan_concat(
    tokens: "Sequence[int]",
    sequence_length: int,
    retain_tail: bool,
    pad_token_id: "Optional[int]",
    masks,
) -> "tuple[list[_Segment], dict[str, int]]":
    if masks is not None:
        raise PackingError("concat does not carry a loss mask")
    total = len(tokens)
    blocks = total // int(sequence_length)
    tail = total - blocks * sequence_length
    usable = blocks
    carried = list(tokens[: blocks * sequence_length])
    retained = 0
    padded_tokens = 0
    if retain_tail:
        if tail and pad_token_id is None:
            raise PackingError(
                "an incomplete tail requires an explicit pad token when retain_tail=True: FAIL")
        if tail:
            usable = blocks + 1
            padded_tokens = sequence_length - tail
            carried = list(tokens) + [int(pad_token_id)] * padded_tokens
    dropped_tail = 0 if retain_tail else tail
    counts = {
        "total_tokens": total,
        "usable_blocks": usable,
        "dropped_tail_tokens": dropped_tail,
        "retained_partial_blocks": 1 if (retain_tail and tail) else 0,
        "retained_partial_padded_tokens": padded_tokens,
    }
    return [_Segment(tokens=carried, mask=None)], counts


def _document_pairs(masks):
    documents = []
    for document in masks:
        if isinstance(document, dict):
            doc_tokens = [int(v) for v in document.get("tokens", [])]
            doc_mask = [int(v) for v in document.get("mask", [])]
        else:
            doc_tokens = [int(v) for v in document]
            doc_mask = [1] * len(doc_tokens)
        documents.append((doc_tokens, doc_mask))
    return documents


def _plan_document_masked(
    documents, sequence_length: int, retain_tail: bool, pad_token_id,
) -> "tuple[list[_Segment], dict[str, int]]":
    total = 0
    usable = 0
    dropped_tail = 0
    retained = 0
    padded_total = 0
    segments: "list[_Segment]" = []
    for document_tokens, document_mask in documents:
        if len(document_tokens) != len(document_mask):
            raise PackingError("one loss-mask bit per token is required: FAIL")
        for value in document_mask:
            if value not in (0, 1):
                raise PackingError("loss masks must contain 0/1 values only: FAIL")
        total += len(document_tokens)
        blocks = len(document_tokens) // sequence_length
        limit = blocks * sequence_length
        emit_mask = list(document_mask[:limit])
        if retain_tail and len(document_tokens) - limit:
            if pad_token_id is None:
                raise PackingError(
                    "an incomplete document tail requires an explicit pad token when retain_tail=True: FAIL")
            usable += 1
            retained += 1
            padded = sequence_length - (len(document_tokens) - limit)
            padded_total += padded
            document_tokens = list(document_tokens) + [int(pad_token_id)] * padded
            # The retained tail keeps the loss bits of its REAL tokens; only
            # the manufactured pad region is loss-off. The previous slice
            # (document_mask[:limit]) threw away the tail bits when a document
            # produced zero full blocks, which silently published an all-pad
            # mask: every token became loss-off and training saw a NaN batch.
            emit_mask = list(document_mask) + [0] * padded
            limit = len(document_tokens)
        elif not retain_tail:
            dropped_tail += len(document_tokens) - limit
        usable += blocks
        blocks = limit // sequence_length
        if blocks:
            segments.append(_Segment(
                tokens=list(document_tokens[: blocks * sequence_length]),
                mask=emit_mask,
            ))
    counts = {
        "total_tokens": total,
        "usable_blocks": usable,
        "dropped_tail_tokens": dropped_tail,
        "retained_partial_blocks": retained,
        "retained_partial_padded_tokens": padded_total,
    }
    return segments, counts


def _shard_record(record: "dict[str, object]") -> ShardRecord:
    return ShardRecord(
        path=str(record["path"]),
        size_bytes=int(record["size_bytes"]),
        sha256=str(record["sha256"]),
        mask_path=(str(record["mask_path"]) if record.get("mask_path") is not None else None),
        mask_size_bytes=(int(record["mask_size_bytes"]) if record.get("mask_size_bytes") is not None else None),
        mask_sha256=(str(record["mask_sha256"]) if record.get("mask_sha256") is not None else None),
    )


def record_to_json(record: ShardRecord) -> "dict[str, object]":
    return {
        "path": record.path, "size_bytes": record.size_bytes, "sha256": record.sha256,
        "mask_path": record.mask_path, "mask_size_bytes": record.mask_size_bytes,
        "mask_sha256": record.mask_sha256,
    }


def pack_tokens(
    tokens: "Sequence[int]",
    sequence_length: int,
    mode: str,
    root: "Path",
    *,
    retain_tail: bool = False,
    masks=None,
    documents: "Optional[Sequence[dict]]" = None,
    pad_token_id: "Optional[int]" = None,
    shard_max_tokens: int = 4_000_000,
    tokenizer: "TokenizerIdentity" = None,
    source_data_hash: str = "",
    overflow_blocks: int = 0,
) -> PackResult:
    """Pack a token stream into SHA-256-verified little-endian shards.

    An explicitly pinned tokenizer identity is mandatory: an unpinned
    tokenizer can never guarantee the recorded vocabulary, so packing
    without one is a labeled FAIL, never a silently invented pin.
    """
    if tokenizer is None:
        raise TokenizerError(
            "pack_tokens requires an explicitly pinned tokenizer identity: "
            "an unpinned tokenizer is FAIL")
    assert_unpinned(tokenizer)
    sequence_length = int(sequence_length)
    if sequence_length < 1:
        raise PackingError("sequence_length must be positive: FAIL")
    if mode not in PACKING_MODES:
        raise PackingError(f"packing mode must be one of {PACKING_MODES}: {mode!r}")
    token_ids = [int(v) for v in tokens]
    # every id is range-checked before any uint32 cast: a negative id is a
    # labeled FAIL, never a numpy OverflowError (M3)
    validate_token_ids(token_ids, tokenizer)
    if pad_token_id is not None:
        pad_id = int(pad_token_id)
        if pad_id < 0 or pad_id > int(tokenizer.manifest.max_token_id):
            raise TokenizerError("pad token id is out of range: FAIL")
    if mode == "concat":
        plan, packing_counts = _plan_concat(
            token_ids, sequence_length, retain_tail, pad_token_id, masks)
    else:
        if documents is not None:
            pairs = [
                ([int(v) for v in doc["tokens"]], [int(v) for v in doc["mask"]])
                for doc in documents
            ]
        elif masks is None:
            raise PackingError(
                "document_masked requires one loss-mask bit per token: FAIL")
        else:
            pairs = _document_pairs(masks)
        for doc_tokens, _mask in pairs:
            validate_token_ids(doc_tokens, tokenizer)
        plan, packing_counts = _plan_document_masked(
            pairs, sequence_length, retain_tail, pad_token_id)
    hash_payload = {
        "mode": mode, "sequence_length": sequence_length,
        "retain_tail": bool(retain_tail), "tokens": token_ids,
    }
    if mode == "document_masked":
        # The retained-tail mask contract (tail bits kept, pad bits 0) is part
        # of what a document_masked generation publishes; one schema number
        # keeps already-published broken-mask generations from being silently
        # reused after the packer fix.
        hash_payload["mask_schema"] = 2
    content_hash = sha256_bytes(canonical_sha256(hash_payload).encode("utf-8"))
    generation_id = content_hash[:16]
    publish_parent = Path(root) / TOKENS_DIR
    publish_parent.mkdir(parents=True, exist_ok=True)
    final_dir = publish_parent / generation_id
    if final_dir.exists():
        raise FileExistsError(
            f"a published token generation is never overwritten: {final_dir}")
    temp_dir = publish_parent / f".tmp-tokens-{generation_id}-{os.getpid()}-{os.urandom(4).hex()}"
    temp_dir.mkdir(parents=True)
    try:
        shard_json: "list[dict[str, object]]" = []
        for index, segment in enumerate(plan):
            shard_name = f"tokens-{index:05d}.bin"
            np.asarray(segment.tokens, dtype=TOKEN_UINT32).tofile(temp_dir / shard_name)
            record: "dict[str, object]" = {
                "path": shard_name,
                "size_bytes": (temp_dir / shard_name).stat().st_size,
                "sha256": sha256_file(temp_dir / shard_name),
                "mask_path": None, "mask_size_bytes": None, "mask_sha256": None,
            }
            if mode == "document_masked":
                mask_name = f"tokens-{index:05d}.mask.bin"
                np.asarray(segment.mask, dtype=MASK_UINT8).tofile(temp_dir / mask_name)
                record["mask_path"] = mask_name
                record["mask_size_bytes"] = (temp_dir / mask_name).stat().st_size
                record["mask_sha256"] = sha256_file(temp_dir / mask_name)
            shard_json.append(record)
        verify_shard_records(temp_dir, [_shard_record(r) for r in shard_json])
        counts = {
            **packing_counts,
            "dropped_tail": int(packing_counts["dropped_tail_tokens"]),
            "sequence_length": sequence_length, "packing_mode": mode,
            "overflow_blocks": int(overflow_blocks),
            "shard_count": len(shard_json),
            "mask_shards": sum(1 for r in shard_json if r.get("mask_path") is not None),
        }
        manifest = {
            "schema_version": 1, "kind": "tokens-packed",
            "generation": generation_id, "source_data_hash": str(source_data_hash),
            "packing_mode": mode, "sequence_length": sequence_length,
            "counts": counts, "shards": shard_json,
            "tokenizer": manifest_json(tokenizer),
        }
        write_manifest(temp_dir, "manifest.json", manifest)
        write_manifest(temp_dir, TOKENIZED_MANIFEST_NAME, {
            "schema_version": 1, "kind": "tokenized-data",
            "sequence_length": sequence_length, "packing_mode": mode,
            "usable_blocks": counts["usable_blocks"],
            "dropped_tail_tokens": counts["dropped_tail_tokens"],
            "overflow_blocks": counts["overflow_blocks"],
            "source_data_hash": str(source_data_hash or content_hash),
            "shards": shard_json,
        })
        prepared = {
            "schema_version": 1, "kind": "tokens-prepared",
            "content_hash": content_hash,
            "manifest": {
                "manifest.json": sha256_file(temp_dir / "manifest.json"),
                TOKENIZED_MANIFEST_NAME: sha256_file(temp_dir / TOKENIZED_MANIFEST_NAME),
                **{r["path"]: r["sha256"] for r in shard_json},
                **{str(r["mask_path"]): str(r["mask_sha256"]) for r in shard_json
                   if r.get("mask_path") is not None},
            },
        }
        write_manifest(temp_dir, PREPARED_MANIFEST_NAME, prepared)
        payloads: "dict[str, bytes]" = {}
        payloads["manifest.json"] = (temp_dir / "manifest.json").read_bytes()
        payloads[TOKENIZED_MANIFEST_NAME] = (temp_dir / TOKENIZED_MANIFEST_NAME).read_bytes()
        for shard_record in shard_json:
            payloads[str(shard_record["path"])] = (temp_dir / str(shard_record["path"])).read_bytes()
            if shard_record.get("mask_path") is not None:
                payloads[str(shard_record["mask_path"])] = (temp_dir / str(shard_record["mask_path"])).read_bytes()
        write_completion_marker(temp_dir, payloads, PREPARED_MANIFEST_NAME)
        expected = ["manifest.json", TOKENIZED_MANIFEST_NAME] + [
            name for shard_record in shard_json
            for name in ((str(shard_record["path"]),) +
                         ((str(shard_record["mask_path"]),) if shard_record.get("mask_path") is not None else ()))
        ]
        if verify_completion_marker(temp_dir, expected) is not True:
            raise PackingError("token shard publication read-back verification is FAIL")
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    publish_shards(temp_dir, final_dir)
    verify_shard_records(final_dir, [_shard_record(r) for r in shard_json])
    return PackResult(
        counts=counts, shards=[_shard_record(r) for r in shard_json], manifest=manifest,
        manifest_path=final_dir / "manifest.json", published_path=final_dir,
        tokenizer=tokenizer,
    )
