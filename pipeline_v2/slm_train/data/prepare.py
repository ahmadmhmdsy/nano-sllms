"""Deterministic preparation and atomic generation publication.

Dedupe happens before splitting; the split is a pure function of dataset
identity, generation identity, document id, and split salt. Preparation
publishes a temporary sibling generation directory first, writes the raw
and prepared manifests and the completion marker, reopens and re-verifies
every hash, and only then atomically publishes the generation. A published
generation is never partially overwritten and never changed; instead a
changed source requires a new generation identity.
"""
import json
import os
import shutil
import uuid
from datetime import datetime, timezone

from pathlib import Path
from typing import Iterable, Mapping

from slm_train.artifacts.markers import (
    verify_completion_marker,
    write_completion_marker,
)
from slm_train.artifacts.manifest import write_manifest
from slm_train.contracts.data import DataManifest
from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256
from slm_train.data.dedupe import (
    apply_normalization,
    dedupe_before_split,
    document_id,
    text_key,
)
from slm_train.data.generation import PreparedGeneration, generation_key_for
from slm_train.data.provenance import (
    freshness_report,
    provenance_record,
    verify_published_generation,
    write_freshness_report,
)
from slm_train.data.sources.base import FreshnessError, SourceError
from slm_train.data.split import (
    HELD_OUT,
    TRAIN,
    VALIDATION,
    assign_split,
    overlap_between,
    require_no_overlap,
    role_forces_held_out,
)
from slm_train.errors import PipelineError
from slm_train.identity.run_id import SAFE_RUN_COMPONENT

PREPARED_DIR = "data/v2/prepared"
PAYLOAD_NAME = "prepared.jsonl"
RAW_MANIFEST_NAME = "raw-manifest.json"
PREPARED_MANIFEST_NAME = "prepared-manifest.json"

REQUIRED_OPTIONS = ("dataset_id", "generation_id", "split_salt", "validation_fraction")


class PreparationError(PipelineError):
    """Fail-closed preparation failure carrying a validation label."""

    def __init__(self, message: str, label: str = "FAIL") -> None:
        super().__init__(message)
        self.label = label


def _validate_options(options) -> dict:
    for name in REQUIRED_OPTIONS:
        if name not in options:
            raise ValueError(f"preparation options require {name}")
    dataset_id = str(options["dataset_id"])
    generation_id = str(options["generation_id"])
    if not SAFE_RUN_COMPONENT.fullmatch(dataset_id):
        raise ValueError(f"dataset_id is not a safe path component: {dataset_id!r}")
    if not SAFE_RUN_COMPONENT.fullmatch(generation_id):
        raise ValueError(f"generation_id is not a safe path component: {generation_id!r}")
    int(options["split_salt"])
    fraction = float(options["validation_fraction"])
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if "dedupe" in options and not isinstance(options["dedupe"], bool):
        raise ValueError("options['dedupe'] must be a bool")
    min_chars = int(options.get("min_chars", 0))
    if min_chars < 0:
        raise ValueError("min_chars must be nonnegative")
    max_chars = options.get("max_chars", None)
    if max_chars is not None and int(max_chars) < 1:
        raise ValueError("max_chars must be a positive int")
    # Declared filters reach the real filter here (A6): a "min_chars:n" or
    # "max_chars:n" string in the declared filters is a filter value, not a
    # recorded intent.
    for item in options.get("filters", ()) or ():
        name, _sep, raw = str(item).partition(":")
        try:
            parsed = int(raw.strip())
        except ValueError as error:
            raise ValueError(
                "declared filter value is not an integer: " + str(item)
            ) from error
        if name.strip() == "min_chars":
            if parsed < 0:
                raise ValueError("min_chars filter must be nonnegative")
            min_chars = parsed
        elif name.strip() == "max_chars":
            if parsed < 1:
                raise ValueError("max_chars filter must be positive")
            max_chars = parsed
        else:
            raise ValueError("unknown data filter: " + str(item))
    # normalization is a certification thread: what the options declare is what
    # _consume_source applies and what every manifest below records.
    normalization = str(options.get("normalization") or "ws-collapse")
    from slm_train.data.dedupe import _probe_normalization

    _probe_normalization(normalization)
    return {
        "dataset_id": dataset_id,
        "generation_id": generation_id,
        "split_salt": int(options["split_salt"]),
        "validation_fraction": fraction,
        "dedupe": bool(options.get("dedupe", False)),
        "normalization": normalization,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "regeneration_command": options.get("regeneration_command", ()),
        "license_note": options.get("license_note", ""),
        "provenance_note": options.get("provenance_note", ""),
        "split_policy": {
            "salt": int(options["split_salt"]),
            "validation_fraction": fraction,
            "policy": "deterministic-hash",
        },
    }


def _consume_source(reader, resolved) -> "tuple[list, dict, dict]":
    documents: "list[dict]" = []
    state = {"rows_seen": 0, "filter_rejected": 0, "truncated": 0}
    for row in reader.iter_rows():
        state["rows_seen"] += 1
        text, applied_normalization = apply_normalization(
            row.get("text", ""), resolved["normalization"]
        )
        if len(text) < resolved["min_chars"]:
            state["filter_rejected"] += 1
            continue
        if resolved["max_chars"] is not None and len(text) > int(resolved["max_chars"]):
            text = text[: int(resolved["max_chars"])]
            state["truncated"] += 1
        documents.append({
            "source_id": reader.source_id(),
            "source_row_index": int(row["_row_index"]),
            "doc_id": document_id(reader.source_id(), int(row["_row_index"]), text),
            "text": text,
        })
    documents = sorted(documents, key=lambda doc: (doc["source_id"], doc["source_row_index"]))
    if reader.required and state["rows_seen"] == 0:
        raise PreparationError("an empty required source set is FAIL")
    return documents, state, {"source_id": reader.source_id()}


def _record_counts(state, documents_before_dedupe, documents_final, duplicates_dropped) -> dict:
    """Record preparation counters with an explicit, checkable identity.

    rows_kept is the final post-filter, post-dedupe count and never conflates
    the intermediate stages: rows_seen == rows_after_filter + filter_rejected
    and rows_after_filter == rows_kept + duplicates_dropped. Any violated
    identity fails the record closed.
    """
    counts = {
        "rows_seen": state["rows_seen"],
        "filter_rejected": state["filter_rejected"],
        "rows_after_filter": documents_before_dedupe,
        "duplicates_dropped": duplicates_dropped,
        "rows_kept": documents_final,
        "truncated": state["truncated"],
    }
    if counts["rows_seen"] != counts["rows_after_filter"] + counts["filter_rejected"]:
        raise PreparationError("preparation counter identity is violated: FAIL")
    if counts["rows_after_filter"] != counts["rows_kept"] + counts["duplicates_dropped"]:
        raise PreparationError("preparation counter identity is violated: FAIL")
    return counts


def _absent_verification(reader) -> dict:
    """The verification record for an explicitly-expected absent source."""
    return {
        "source_id": reader.source_id(),
        "semantic_fingerprint": reader.semantic_fingerprint(),
        "label": "SKIPPED",
        "freshness": "SKIPPED",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "required": reader.required,
        "absent_source_decision": reader.absent_source.model_dump(),
    }


def prepare_generation(reader, options, root) -> PreparedGeneration:
    """Prepare one generation deterministically and publish it atomically.

    root acts as the V2 layout root; the generation publishes under
    data/v2/prepared/<dataset_id>/<generation_id>.
    """
    root = Path(root)
    resolved = _validate_options(options)
    absent_expected = (
        not reader.required
        and reader.absent_source is not None
        and reader.absent_source.absent_content == "expected"
    )
    # Revalidation first: every consuming stage re-checks the pinned identity.
    # An optional source whose recorded absent-content decision expects the
    # file to be absent takes the declared branch here, before any FAIL.
    try:
        verification = reader.revalidate()
    except FreshnessError:
        raise
    except SourceError:
        raise
    except FileNotFoundError as error:
        if absent_expected:
            verification = _absent_verification(reader)
        elif not reader.required:
            # An optional source absent at revalidation either takes the
            # recorded decision deferred to the user or fails closed for the
            # user decision, never silently.
            raise PreparationError(
                "an optional source is absent but no explicit absent-content "
                "decision was recorded: NEEDS_USER_DECISION", label="NEEDS_USER_DECISION") from error
        else:
            raise PreparationError(
                f"revalidation failed for a pinned source: {error}: FAIL", label="FAIL") from error

    provenance_entry = provenance_record(reader)
    if absent_expected and verification.get("absent_source_decision") is not None:
        # Declared expected-absent source: consume it as an empty source, and
        # record the explicit decision in the published evidence.
        documents, state = [], {"rows_seen": 0, "filter_rejected": 0, "truncated": 0}
    else:
        try:
            documents, state, _ = _consume_source(reader, resolved)
        except FileNotFoundError as error:
            if reader.required:
                raise PreparationError(
                    f"an empty required source set is FAIL: {error}", label="FAIL") from error
            if reader.absent_source is not None and reader.absent_source.absent_content == "expected":
                verification = _absent_verification(reader)
                documents, state = [], {"rows_seen": 0, "filter_rejected": 0, "truncated": 0}
            else:
                raise PreparationError(
                    "an optional source is absent but no explicit absent-content "
                    "decision was recorded: NEEDS_USER_DECISION", label="NEEDS_USER_DECISION") from error

    documents_before_dedupe = len(documents)
    if resolved["dedupe"]:
        documents, duplicates_dropped = dedupe_before_split(documents, text_key)
    else:
        duplicates_dropped = 0

    if reader.required and not documents:
        raise PreparationError("an empty required preparation set is FAIL")

    train_ids: "list[str]" = []
    validation_ids: "list[str]" = []
    held_out_ids: "list[str]" = []
    held_out_role = role_forces_held_out(reader.role)
    for document in documents:
        if held_out_role:
            # A test or external-gate source is a certification that this
            # corpus is held out. It may never enter the training split, so
            # the role owns the split instead of the hash window (A11).
            held_out_ids.append(document["doc_id"])
            continue
        assigned = assign_split(
            resolved["dataset_id"], resolved["generation_id"],
            document["doc_id"], resolved["split_salt"], resolved["validation_fraction"])
        if assigned == VALIDATION:
            validation_ids.append(document["doc_id"])
        else:
            train_ids.append(document["doc_id"])
    # The overlap between the actually collected train and validation sets is
    # always computed over the published ids and any overlap fails closed.
    require_no_overlap(train_ids, validation_ids)
    overlapping_ids = overlap_between(train_ids, validation_ids)

    # The applied normalization is the authority here: it is what the rows
    # actually went through, and every manifest below records exactly that.
    applied_mode = resolved["normalization"]
    content_hash = canonical_sha256({
        "documents": documents,
        "sources": [provenance_entry],
        "filters": {
            "min_chars": resolved["min_chars"],
            "max_chars": resolved["max_chars"],
        },
        "normalization": applied_mode,
        "split_policy": resolved["split_policy"],
    })

    counts = _record_counts(state, documents_before_dedupe, len(documents), duplicates_dropped)
    counts["overlap_count"] = len(overlapping_ids)
    counts[HELD_OUT] = len(held_out_ids)

    manifest = DataManifest(
        schema_version=1,
        source_ids=(reader.source_id(),),
        source_roles={reader.source_id(): reader.role},
        source_identity={
            "kind": str(provenance_entry["kind"]),
            "root_hash": str(provenance_entry["root_hash"]),
            "repo_id": str(provenance_entry["repo_id"]),
            "revision": str(provenance_entry["revision"]),
        },
        source_fingerprint=reader.semantic_fingerprint(),
        freshness_policy="pinned",
        supersession_policy="explicit-generation",
        dataset_config={"dataset_id": resolved["dataset_id"],
                        "generation_id": resolved["generation_id"]},
        split_policy=resolved["split_policy"],
        filters={"min_chars": resolved["min_chars"], "max_chars": resolved["max_chars"]},
        normalization=applied_mode,
        dedupe_mode="sha-normalized-first" if resolved["dedupe"] else "none",
        document_id_fields=("source_id", "source_row_index", "text"),
        document_counts={TRAIN: len(train_ids), VALIDATION: len(validation_ids),
                         HELD_OUT: len(held_out_ids)},
        license_note=reader.license_note,
        provenance_note=reader.provenance_note,
        overlap_report={"overlap_count": len(overlapping_ids),
                        "overlapping_ids": list(overlapping_ids)},
        truncation_counts={"truncated": state["truncated"]},
        rejection_counts={"filter_rejected": state["filter_rejected"],
                          "duplicates_dropped": duplicates_dropped},
        regeneration_command=resolved["regeneration_command"],
        content_hash=content_hash,
    )

    raw_manifest = {
        "schema_version": 1,
        "kind": "data-raw",
        "validation_labels": {reader.source_id(): verification["label"]},
        "sources": [provenance_entry],
        "counts": dict(counts),
        "optional_absent_source": (None if verification.get("absent_source_decision") is None
                                   else verification["absent_source_decision"]),
        "filters": {"min_chars": resolved["min_chars"], "max_chars": resolved["max_chars"],
                    "declared": list(options.get("filters", ()) or ())},
        "normalization": applied_mode,
        "regeneration_command": resolved["regeneration_command"],
    }

    train_selector = set(train_ids)
    held_out_selector = set(held_out_ids)
    prepared_payload_lines = [
        {
            "doc_id": document["doc_id"],
            "source_id": document["source_id"],
            "source_row_index": document["source_row_index"],
            "split": (HELD_OUT if document["doc_id"] in held_out_selector
                      else (TRAIN if document["doc_id"] in train_selector
                            else VALIDATION)),
            "role": reader.role,
            "text": document["text"],
        }
        for document in documents
    ]
    payload_bytes = (
        "".join(
            canonical_json_bytes(line).decode("utf-8") + "\n" for line in prepared_payload_lines
        ).encode("utf-8")
    )

    committed = _publish_generation(
        root=root,
        dataset_id=resolved["dataset_id"],
        generation_id=resolved["generation_id"],
        manifest=manifest,
        raw_manifest=raw_manifest,
        payload_bytes=payload_bytes,
    )
    # Consume-side gate (design 9.1): reopen the published completion marker
    # and validate the published manifest/schema identity before returning.
    verify_published_generation(committed, reader.source_id(), reader.semantic_fingerprint())

    semantic_hash = canonical_sha256({
        "content_hash": content_hash,
        "manifest": manifest.model_dump(),
        "sources": [provenance_entry],
    })
    report = freshness_report({
        "semantic_fingerprint": reader.semantic_fingerprint(),
        "source_id": reader.source_id(),
        "label": verification["label"],
        "generation_id": resolved["generation_id"],
    })
    write_freshness_report(root, resolved["dataset_id"], resolved["generation_id"], report)
    return PreparedGeneration(
        semantic_hash=semantic_hash,
        counts=dict(counts) | {
            TRAIN: len(train_ids),
            VALIDATION: len(validation_ids),
            HELD_OUT: len(held_out_ids),
        },
        published_path=committed,
        generation_key=generation_key_for(resolved["dataset_id"], semantic_hash),
        split_assignments={line["doc_id"]: line["split"] for line in prepared_payload_lines},
        instance_label="PASS",
        provenance={
            "source_roles": {reader.role: 1},
            "license_note": reader.license_note,
            "provenance_note": reader.provenance_note,
            "regeneration_command": resolved["regeneration_command"],
            "source_id": reader.source_id(),
        },
    )


def _publish_generation(root, dataset_id, generation_id, manifest, raw_manifest, payload_bytes) -> Path:
    publish_parent = root / PREPARED_DIR / dataset_id
    publish_parent.mkdir(parents=True, exist_ok=True)
    final_path = publish_parent / generation_id
    if final_path.exists():
        raise FileExistsError(
            f"a published generation is never overwritten: {final_path}")
    temp_parent = publish_parent
    temp_dir = temp_parent / f".tmp-{dataset_id}-{generation_id}-{os.getpid()}-{os.urandom(4).hex()}"
    try:
        temp_dir.mkdir()
        write_manifest(temp_dir, "manifest.json", manifest.model_dump())
        write_manifest(temp_dir, RAW_MANIFEST_NAME, raw_manifest)
        payload_target = temp_dir / PAYLOAD_NAME
        _atomic_payload_write(payload_target, payload_bytes)
        from slm_train.contracts.hash import sha256_bytes, sha256_file
        prepared_manifest = {
            "schema_version": 1,
            "kind": "data-prepared",
            "content_hash": manifest.content_hash,
            "manifest": {
                PAYLOAD_NAME: sha256_file(payload_target),
                RAW_MANIFEST_NAME: sha256_file(temp_dir / RAW_MANIFEST_NAME),
                "manifest.json": sha256_file(temp_dir / "manifest.json"),
            },
        }
        write_manifest(temp_dir, PREPARED_MANIFEST_NAME, prepared_manifest)
        write_completion_marker(
            temp_dir,
            {
                PAYLOAD_NAME: payload_target.read_bytes(),
                RAW_MANIFEST_NAME: (temp_dir / RAW_MANIFEST_NAME).read_bytes(),
                "manifest.json": (temp_dir / "manifest.json").read_bytes(),
            },
            PREPARED_MANIFEST_NAME,
        )
        if verify_completion_marker(temp_dir, [PAYLOAD_NAME, RAW_MANIFEST_NAME, "manifest.json"]) is not True:
            raise PreparationError("publication read-back verification is FAIL")
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=False)
        raise
    try:
        os.rename(temp_dir, final_path)
    except FileExistsError as error:
        # Fail closed on the filesystem operation itself, never on a
        # checked-then-renamed pre-check alone (time-of-check/time-of-use).
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileExistsError(
            f"a published generation is never overwritten: {final_path}") from error
    except OSError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return final_path


def _atomic_payload_write(target: Path, payload: bytes) -> None:
    from slm_train.artifacts.atomic import atomic_write_bytes
    atomic_write_bytes(target, payload)
