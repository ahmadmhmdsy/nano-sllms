"""Completion markers: written last, after manifest and payload read-back.

The marker binds the exact sha256 of the published manifest bytes (read
back from disk) alongside every payload hash read back from disk. Every
verification path fails closed: a marker whose payload section is absent,
empty, or short of what the caller expects never verifies.
"""
import json
from pathlib import Path
from typing import Iterable, Mapping

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.artifacts.manifest import load_manifest
from slm_train.contracts.hash import canonical_json_bytes, sha256_bytes, sha256_file

MARKER_NAME = "COMPLETE.json"


def _load_json_manifest(root: Path, manifest_path: str) -> object:
    """The marker may only be built on top of a published, re-read manifest."""
    try:
        manifest = load_manifest(root, manifest_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"completion marker requires the manifest: {manifest_path}") from error
    if not isinstance(manifest, (dict, list)):
        raise ValueError(f"completion marker requires a JSON manifest: {manifest_path}")
    return manifest


def _publish_marker(root: Path, manifest_path: str, payload_hashes: "dict[str, str]") -> Path:
    """Bind the published manifest bytes and the certified payload hashes."""
    _load_json_manifest(root, manifest_path)
    manifest_bytes = (root / manifest_path).read_bytes()
    marker = {
        "status": "PASS",
        "manifest": manifest_path,
        "manifest_hash": sha256_bytes(manifest_bytes),
        "payload_hashes": dict(sorted(payload_hashes.items())),
    }
    atomic_write_bytes(root / MARKER_NAME, canonical_json_bytes(marker), replace_existing=True)
    return root / MARKER_NAME


def _read_marker(root: Path) -> dict:
    """Load COMPLETE.json; the marker must exist and be a JSON object."""
    marker_path = root / MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError(f"completion marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_bytes().decode("utf-8"))
    if not isinstance(marker, dict):
        raise ValueError(f"completion marker is not a JSON object: {marker_path}")
    return marker


def _verify_manifest_binding(root: Path, marker: dict) -> None:
    manifest_path = marker.get("manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("COMPLETE.json marker does not name a certified manifest")
    manifest = root / manifest_path
    if not manifest.is_file():
        raise FileNotFoundError(f"certified manifest is missing: {manifest_path}")
    if sha256_bytes(manifest.read_bytes()) != marker.get("manifest_hash"):
        raise ValueError("manifest hash mismatch against COMPLETE.json marker")


def _certified_payload_hashes(marker: dict) -> "dict[str, str]":
    """Read the payload map without ever defaulting a missing section to {}."""
    if "payload_hashes" not in marker:
        raise ValueError("COMPLETE.json marker is missing the payload_hashes section")
    payload_hashes = marker["payload_hashes"]
    if not isinstance(payload_hashes, dict):
        raise ValueError("COMPLETE.json marker payload_hashes is not a JSON object")
    return payload_hashes


def _verify_payload_bytes(root: Path, payload_hashes: "dict[str, str]") -> None:
    for name, expected in sorted(payload_hashes.items()):
        published = root / name
        if not published.is_file():
            raise ValueError(f"certified payload is missing: {name}")
        if sha256_file(published) != expected:
            raise ValueError(f"payload hash mismatch against COMPLETE.json marker: {name}")


def write_completion_marker(
    run_root: Path,
    payloads: Mapping[str, bytes],
    manifest_path: str,
    certified: "Mapping[str, str] | None" = None,
) -> Path:
    """Publish COMPLETE.json only after payload hashes verify against the
    manifest, re-reading each published payload and the manifest from disk.
    At least one payload is required; a genuinely empty publication uses
    write_payload_free_completion_marker."""
    if not payloads:
        raise ValueError(
            "a completion marker must certify at least one payload; use "
            "write_payload_free_completion_marker for a zero-payload publication")
    root = Path(run_root)
    manifest = _load_json_manifest(root, manifest_path)
    # `certified` lets a publisher whose manifest is a typed contract supply the
    # payload hash map directly, instead of restating it under a "manifest" key.
    if certified is not None:
        if not isinstance(certified, Mapping):
            raise ValueError("certified payload hashes must be a mapping")
        required = dict(certified)
    else:
        required = manifest.get("manifest", {}) if isinstance(manifest, dict) else {}
    payload_hashes: "dict[str, str]" = {}
    for name, payload in payloads.items():
        recorded = required.get(name)
        if recorded is None:
            raise ValueError(f"hash read-back failed: manifest does not cover {name}")
        if sha256_bytes(payload) != recorded:
            raise ValueError(f"hash read-back failed for {name}")
        published = root / name
        if not published.is_file():
            raise ValueError(f"hash read-back failed: published payload is missing: {name}")
        disk_hash = sha256_file(published)
        if disk_hash != recorded:
            raise ValueError(f"hash read-back failed for {name}")
        payload_hashes[name] = disk_hash
    return _publish_marker(root, manifest_path, payload_hashes)


def write_payload_free_completion_marker(run_root: Path, manifest_path: str) -> Path:
    """Explicit publication path for a run that genuinely certifies zero
    payloads: the manifest binding is still read back from disk."""
    return _publish_marker(Path(run_root), manifest_path, {})


def verify_completion_marker(run_root: Path, expected_payloads: Iterable[str]) -> bool:
    """Re-verify COMPLETE.json against current on-disk bytes; fail closed
    unless the marker carries a non-empty payload map covering every payload
    the caller expects. A zero-payload publication is verified through
    verify_payload_free_completion_marker, never through an implicit default.
    """
    root = Path(run_root)
    marker = _read_marker(root)
    _verify_manifest_binding(root, marker)
    payload_hashes = _certified_payload_hashes(marker)
    if not payload_hashes:
        raise ValueError(
            "COMPLETE.json marker certifies no payloads; use "
            "verify_payload_free_completion_marker for a zero-payload publication")
    expected = sorted(set(expected_payloads))
    if not expected:
        raise ValueError(
            "expected_payloads is empty; use verify_payload_free_completion_marker "
            "for a zero-payload publication")
    missing = [name for name in expected if name not in payload_hashes]
    if missing:
        raise ValueError(
            f"COMPLETE.json marker does not certify expected payload(s): {', '.join(missing)}")
    _verify_payload_bytes(root, payload_hashes)
    return True


def verify_payload_free_completion_marker(run_root: Path) -> bool:
    """Explicit verification for a publication that genuinely certifies zero
    payloads: the manifest binding must verify and the payload map must be
    present and empty."""
    root = Path(run_root)
    marker = _read_marker(root)
    _verify_manifest_binding(root, marker)
    if _certified_payload_hashes(marker):
        raise ValueError(
            "COMPLETE.json marker certifies payloads; use verify_completion_marker "
            "with the expected payload names")
    return True
