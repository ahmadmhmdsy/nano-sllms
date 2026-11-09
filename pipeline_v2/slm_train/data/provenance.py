"""Source provenance records and mutable freshness reporting.

Provenance is immutable evidence recorded alongside a generation; freshness
reports are mutable evidence that never touch the semantic generation hash.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from slm_train.artifacts.atomic import replace_mutable_bytes
from slm_train.contracts.hash import canonical_json_bytes

FRESHNESS_DIR = "_freshness"
PUBLISHED_PAYLOADS = ("prepared.jsonl", "raw-manifest.json", "manifest.json")


def verify_published_generation(published_path, source_id: str, source_fingerprint: str) -> dict:
    """Reopen a published generation's completion marker and manifest identity.

    Fail closed unless the completion marker certifies the full published
    payload set and the manifest reloads as schema-version 1 with the expected
    source identity. This is the consume-side revalidation gate (design 9.1).
    """
    from slm_train.artifacts.markers import verify_completion_marker

    path = Path(published_path)
    if verify_completion_marker(path, PUBLISHED_PAYLOADS) is not True:
        raise ValueError("published generation completion marker verification is FAIL")
    manifest = load_published_manifest(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("published manifest schema identity mismatch: FAIL")
    record = json.loads(manifest_json(path).decode("utf-8"))
    if str(record.get("source_fingerprint")) != str(source_fingerprint):
        raise ValueError("published manifest source fingerprint mismatch: FAIL")
    recorded_ids = record.get("source_ids") or []
    if source_id and source_id not in recorded_ids:
        raise ValueError("published manifest does not pin the declared source: FAIL")
    return record


def manifest_json(published_path) -> bytes:
    return (Path(published_path) / "manifest.json").read_bytes()


TUPLE_FIELDS = ("source_ids", "document_id_fields", "regeneration_command")


def load_published_manifest(published_path) -> dict:
    from slm_train.contracts.data import DataManifest

    record = json.loads(manifest_json(published_path).decode("utf-8"))
    restored = dict(record)
    for name in TUPLE_FIELDS:
        if isinstance(restored.get(name), list):
            restored[name] = tuple(restored[name])
    DataManifest.model_validate(restored)
    return record



def provenance_record(reader) -> dict:
    """Immutable provenance entry for one pinned source reader."""
    rels = []
    identity = reader.identity()
    for relative, _path in _source_files(reader):
        rels.append(relative)
    return {
        "source_id": reader.source_id(),
        "declared_source_id": reader.declared_source_id,
        "expected_row_budget": reader.expected_row_budget,
        "expected_token_budget": reader.expected_token_budget,
        "role": reader.role,
        "kind": identity["kind"],
        "root_hash": identity["root_hash"],
        "repo_id": identity.get("repo_id", ""),
        "revision": identity.get("revision", ""),
        "files": rels,
        "license_note": reader.license_note,
        "provenance_note": reader.provenance_note,
        "required": reader.required,
        "freshness_policy": "pinned",
        "supersession_policy": "explicit-generation",
    }


def _source_files(reader):
    for relative, path in reader.relative_files():
        yield relative, path


def freshness_report(source_state: dict, now: Optional[datetime] = None) -> dict:
    """A fresh immutable freshness report snapshot (mutable evidence)."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    report = dict(source_state)
    report["checked_at"] = moment.isoformat()
    report["fingerprint"] = source_state.get("semantic_fingerprint", "")
    return report


def write_freshness_report(root: Path, dataset_id: str, generation_id: str, report: dict) -> Path:
    """Persist one revalidation freshness report as explicit mutable evidence.

    The report lives outside the published generation directory and outside
    every manifest, so freshness verification never changes the semantic hash.
    """
    checked_at = str(report.get("checked_at", "")) or datetime.now(timezone.utc).isoformat()
    stamp = "".join(character if character.isalnum() else "-" for character in checked_at)
    # The filename carries the generation and source identity so successive
    # revalidations of different generations never collide on one report file.
    slug = "-".join(str(value) for value in (
        dataset_id, report.get("generation_id", ""), report.get("source_id", ""), stamp
    ))
    safe = "".join(character if character.isalnum() else "-" for character in slug)
    target = Path(root) / "data" / "v2" / "prepared" / FRESHNESS_DIR / dataset_id / f"{safe}.json"
    replace_mutable_bytes(target, canonical_json_bytes(report))
    return target
