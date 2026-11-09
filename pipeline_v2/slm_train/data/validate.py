"""Explicit packed-token validation with the approved label vocabulary.

Validation labels are exactly PASS, FAIL, SKIPPED, BLOCKED,
NEEDS_USER_DECISION. Every check reopens published bytes from disk and
fails closed; the engine receives a labeled report before any training.
No library exception leaks: a structurally corrupt record is rejected by
the shape gate before any field is trusted, and every remaining check is
wrapped into one of the approved labels.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

from slm_train.artifacts.markers import verify_completion_marker
from slm_train.contracts.hash import canonical_sha256
from slm_train.data.dataset import DatasetError, _load_tokens_record
from slm_train.data.packing import TOKEN_UINT32, TOKENIZED_MANIFEST_NAME
from slm_train.data.shards import ShardRecord, verify_shard_records
from slm_train.data.tokenizer import (
    TokenizerError, assert_unpinned, reopen_identity, tokenizer_from_record,
)

LABELS = ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION")
_ORDER = {"PASS": 0, "SKIPPED": 1, "NEEDS_USER_DECISION": 2, "BLOCKED": 3, "FAIL": 4}


def _worst(labels) -> str:
    label = "PASS"
    for value in labels:
        if _ORDER.get(value, 0) > _ORDER[label]:
            label = value
    return label


# The published counts section also carries the packing parameters
# (sequence_length, packing_mode) beside the counts themselves; only the
# integer counts below are read as numbers and are therefore type-checked.
_COUNT_FIELDS = (
    "usable_blocks",
    "dropped_tail_tokens",
    "retained_partial_padded_tokens",
    "total_tokens",
    "overflow_blocks",
    "dropped_tail",
    "shard_count",
    "mask_shards",
)


def _structural_finding(record: "dict") -> str:
    """Return a labeled FAIL reason for a malformed manifest, else "".

    Structural corruption is rejected here, before any field is trusted, so
    the checks below can never hand a caller a raw TypeError, ValueError,
    KeyError, or AttributeError raised by a malformed published record.
    """
    sequence_length = record.get("sequence_length")
    if (isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int) or sequence_length < 1):
        return "the recorded sequence length is not a positive integer: FAIL"
    packing_mode = record.get("packing_mode")
    if not isinstance(packing_mode, str) or not packing_mode:
        return "the recorded packing mode is not a non-empty string: FAIL"
    if not isinstance(record.get("tokenizer"), dict):
        return "the recorded tokenizer identity is not a JSON object: FAIL"
    counts = record.get("counts", {})
    if counts is None:
        counts = {}
    if not isinstance(counts, dict):
        return "the recorded counts section is not a JSON object: FAIL"
    for key in _COUNT_FIELDS:
        value = counts.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            return f"the recorded count {key!r} is not an integer: FAIL"
    shards = record.get("shards")
    if not isinstance(shards, list) or not shards:
        return "the manifest records no ordered shard list: FAIL"
    for index, entry in enumerate(shards):
        if not isinstance(entry, dict):
            return f"shard entry {index} is not a JSON object: FAIL"
        for key in ("path", "size_bytes", "sha256"):
            if key not in entry:
                return f"shard entry {index} is missing the required {key!r} key: FAIL"
        for key in ("path", "mask_path"):
            value = entry.get(key)
            if value is None:
                continue
            if (not isinstance(value, str) or not value
                    or ".." in Path(value).parts or Path(value).is_absolute()):
                return f"the recorded {key} is not a relative name: {value!r}: FAIL"
        for key in ("size_bytes", "mask_size_bytes"):
            value = entry.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                return f"the recorded shard {key!r} is not an integer: FAIL"
        for key in ("sha256", "mask_sha256"):
            value = entry.get(key)
            if value is not None and not isinstance(value, str):
                return f"the recorded shard {key!r} is not a string: FAIL"
        masked = entry.get("mask_path") is not None
        complete = (entry.get("mask_size_bytes") is not None
                    and entry.get("mask_sha256") is not None)
        if masked != complete:
            return f"shard entry {index} records an incomplete loss-mask triple: FAIL"
    return ""


def _count(counts: "dict", key: str, default: int) -> int:
    """Read a recorded count without ever raising: corruption is already FAIL."""
    value = counts.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return int(default)
    return int(value)


def validate_packed_tokens(
    manifest_path,
    *,
    sequence_length: "Optional[int]" = None,
    expected_generation: "Optional[str]" = None,
    expected_tokenizer_hash: "Optional[str]" = None,
    strict: bool = False,
) -> dict:
    """Reopen a published token generation from disk and label every check."""
    manifest_path = Path(manifest_path)
    parent = manifest_path.parent
    checks: "dict[str, str]" = {}
    failure: "Optional[str]" = None

    def fail(name: str, message: str) -> None:
        nonlocal failure
        checks[name] = "FAIL"
        if failure is None:
            failure = message

    def finish(overflow_blocks: int = 0) -> dict:
        label = _worst(checks.values())
        result = {
            "label": label, "checks": dict(checks),
            "overflow_blocks": int(overflow_blocks),
            "path": str(manifest_path), "reason": str(failure or ""),
        }
        if strict and label != "PASS":
            raise TokenizerError(
                str(failure or f"packed-token validation is {label}"), label=label)
        return result

    # -- the manifest is re-read from disk, then gated on its shape: a
    #    structurally corrupt record never reaches a downstream check ------
    try:
        raw = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        fail("manifest_load", f"the tokenized manifest is unreadable: FAIL ({error})")
        return finish()
    if not isinstance(raw, dict):
        fail("manifest_load", "the tokenized manifest is not a JSON object: FAIL")
        return finish()
    finding = _structural_finding(raw)
    if finding:
        fail("manifest_structure", finding)
        return finish()
    checks["manifest_structure"] = "PASS"
    try:
        record = _load_tokens_record(manifest_path)
        checks["manifest_load"] = "PASS"
    except DatasetError as error:
        fail("manifest_load", str(error))
        return finish()
    seq = int(record["sequence_length"])
    packing_mode = str(record.get("packing_mode"))

    # -- ordered shard list: relative, listed, no duplicates, fully recorded --
    shards: "list[ShardRecord]" = []
    known_paths: "set[str]" = set()
    ordered_ok = True
    for shard_record in record.get("shards", []):
        path = str(shard_record.get("path", ""))
        if (not path or path in known_paths
                or ".." in Path(path).parts or Path(path).is_absolute()):
            ordered_ok = False
        known_paths.add(path)
        if (not shard_record.get("sha256")
                or not isinstance(shard_record.get("size_bytes"), int)):
            ordered_ok = False
        mask_path = (str(shard_record["mask_path"])
                     if shard_record.get("mask_path") is not None else None)
        shards.append(ShardRecord(
            path=path,
            size_bytes=int(shard_record.get("size_bytes", 0)),
            sha256=str(shard_record.get("sha256", "")),
            mask_path=mask_path,
            mask_size_bytes=(int(shard_record["mask_size_bytes"])
                             if shard_record.get("mask_size_bytes") is not None else None),
            mask_sha256=(str(shard_record["mask_sha256"])
                         if shard_record.get("mask_sha256") is not None else None),
        ))
    if ordered_ok:
        checks["ordered_shard_list"] = "PASS"
    else:
        fail("ordered_shard_list", "the ordered shard list is malformed: FAIL")

    # -- shard and mask read-back: every size and hash is re-read from disk --
    try:
        verify_shard_records(parent, shards)
        checks["shard_readback"] = "PASS"
    except (ValueError, OSError, FileNotFoundError) as error:
        fail("shard_readback", str(error))

    # -- mask completeness ----------------------------------------------------
    if packing_mode == "document_masked" and any(s.mask_path is None for s in shards):
        fail("mask_completeness",
             "document_masked requires one loss-mask file per shard: FAIL")
    else:
        checks["mask_completeness"] = "PASS"

    # -- the completion marker must certify the primary manifest, the
    #    tokenized manifest, and every shard and mask payload ---------------
    payload_names = ["manifest.json", TOKENIZED_MANIFEST_NAME] + [
        name for shard_record in shards
        for name in ((shard_record.path,) +
                     ((shard_record.mask_path,) if shard_record.mask_path is not None else ()))
    ]
    try:
        if verify_completion_marker(parent, payload_names) is not True:  # pragma: no cover
            raise TokenizerError(f"completion marker verification is FAIL for {manifest_path}")
        checks["completion_marker"] = "PASS"
    except (ValueError, OSError, FileNotFoundError, TokenizerError) as error:
        fail("completion_marker", str(error))

    # -- the tokenized manifest must agree with the certified token manifest --
    counts = dict(record.get("counts") or {})
    try:
        tokenized_json = json.loads(
            (parent / TOKENIZED_MANIFEST_NAME).read_bytes().decode("utf-8"))
        if not isinstance(tokenized_json, dict):
            raise ValueError("the tokenized-data manifest is not a JSON object: FAIL")
        if (int(tokenized_json.get("sequence_length", -1)) != seq
                or int(tokenized_json.get("usable_blocks", -1)) != int(counts.get("usable_blocks", -1))
                or int(tokenized_json.get("dropped_tail_tokens", -1)) != int(counts.get("dropped_tail_tokens", -1))
                or tokenized_json.get("shards") != record.get("shards")):
            raise ValueError(
                "the tokenized-data manifest disagrees with the certified token "
                "manifest: FAIL")
        checks["tokenized_manifest_crosscheck"] = "PASS"
    except (ValueError, TypeError, KeyError, AttributeError, OSError, UnicodeDecodeError) as error:
        fail("tokenized_manifest_crosscheck", f"{error}: FAIL")

    # -- counts identity (retain-tail padding stays accounted for, M2) --------
    try:
        total_tokens = 0
        usable = 0
        for shard_record in shards:
            shard = np.memmap(parent / shard_record.path, dtype=TOKEN_UINT32, mode="r")
            total_tokens += int(len(shard))
            usable += int(len(shard)) // seq
    except (ValueError, TypeError, OSError) as error:
        fail("counts_identity",
             f"the published token shards are unreadable: FAIL ({error})")
    else:
        usable_recorded = _count(counts, "usable_blocks", usable)
        dropped = _count(counts, "dropped_tail_tokens", 0)
        retained_padded = _count(counts, "retained_partial_padded_tokens", 0)
        recorded_total = _count(counts, "total_tokens", 0)
        # The shard bytes carry usable*seq tokens. The recorded total refers to
        # the REAL token stream, so the two differ by exactly the tail: a
        # retained tail is padded up to a block and its padding is subtracted,
        # while a dropped tail never reached the shards and is subtracted. The
        # sign matters and a plus here only passes when the tail happens to be
        # zero, which is why a fixture could not catch it.
        if (usable == usable_recorded
                and usable * seq - retained_padded == recorded_total - dropped):
            checks["counts_identity"] = "PASS"
        else:
            fail("counts_identity", "the recorded counts identity is violated: FAIL")

    # -- tokenizer identity, pinning, and vocabulary --------------------------
    tokenizer = None
    extra_record: "dict[str, object]" = {}
    try:
        tokenizer, extra_record = tokenizer_from_record(record.get("tokenizer", {}))
        checks["tokenizer_manifest"] = "PASS"
    except (TokenizerError, TypeError, KeyError, AttributeError) as error:
        fail("tokenizer_manifest", f"{error}: FAIL")
    if tokenizer is not None:
        try:
            assert_unpinned(reopen_identity(tokenizer, extra_record))
            checks["tokenizer_pinned"] = "PASS"
        except TokenizerError as error:
            fail("tokenizer_pinned", str(error))
        try:
            hash_record = canonical_sha256(tokenizer.model_dump())
        except (ValueError, TypeError) as error:
            fail("tokenizer_identity",
                 f"the recorded tokenizer identity is not canonical: FAIL ({error})")
        else:
            if expected_tokenizer_hash is not None and hash_record != expected_tokenizer_hash:
                fail("tokenizer_identity",
                     "the recorded tokenizer identity disagrees with the expected "
                     "pinning: FAIL")
            else:
                checks["tokenizer_identity"] = "PASS"
        try:
            vocabulary_size = int(tokenizer.vocabulary_size)
            max_token_id = int(tokenizer.max_token_id)
            for shard_record in shards:
                data = np.memmap(parent / shard_record.path, dtype=TOKEN_UINT32, mode="r")
                if len(data):
                    highest = int(max(data))
                    if highest < 0 or highest >= vocabulary_size or highest > max_token_id:
                        raise TokenizerError(
                            f"token id {highest} is out of the recorded vocabulary "
                            f"range [0, {max_token_id}]: FAIL")
            checks["token_vocabulary"] = "PASS"
        except TokenizerError as error:
            fail("token_vocabulary", str(error))
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            fail("token_vocabulary", f"{error}: FAIL")
    # -- requested sequence length --------------------------------------------
    try:
        requested_seq = None if sequence_length is None else int(sequence_length)
    except (ValueError, TypeError):
        requested_seq = None
        if sequence_length is not None:
            fail("sequence_length",
                 "the requested sequence length is not an integer: FAIL")
    if sequence_length is not None and requested_seq is not None and seq != requested_seq:
        fail("sequence_length",
             "the published sequence length disagrees with the requested one: FAIL")
    elif sequence_length is None or requested_seq == seq:
        checks["sequence_length"] = "PASS"
    # -- expected source generation -------------------------------------------
    if (expected_generation is not None
            and str(record.get("generation")) != str(expected_generation)):
        fail("source_generation",
             f"the manifest completion marker certifies generation "
             f"{record.get('generation')}, not {expected_generation}: FAIL")
    else:
        checks["source_generation"] = "PASS"
    return finish(overflow_blocks=_count(counts, "overflow_blocks", 0))
