"""Unit tests for atomic artifact publication and path policy."""
from pathlib import Path

import pytest

from slm_train.artifacts.atomic import atomic_write_bytes, replace_mutable_bytes
from slm_train.artifacts.log_index import LogIndex, append_log_segment
from slm_train.artifacts.manifest import write_manifest
from slm_train.artifacts.markers import write_completion_marker
from slm_train.artifacts.paths import PathPolicy


def test_atomic_write_is_complete(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"complete")
    assert target.read_bytes() == b"complete"
    assert not list(tmp_path.glob(".artifact.bin.*"))


def test_immutable_write_does_not_replace_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"first"


def test_immutable_write_race_has_one_winner(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    target = tmp_path / "artifact.bin"

    def publish(value: bytes) -> str:
        try:
            atomic_write_bytes(target, value)
            return "published"
        except FileExistsError:
            return "collision"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert sorted(outcomes) == ["collision", "published"]
    assert target.read_bytes() in {b"first", b"second"}


def test_replace_mutable_updates_content(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    replace_mutable_bytes(target, b"one")
    assert target.read_bytes() == b"one"
    replace_mutable_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert not list(tmp_path.glob(".state.json.*"))


def test_failed_mid_write_leaves_no_canonical_file(tmp_path, monkeypatch):
    import os
    import slm_train.artifacts.atomic as atomic
    target = tmp_path / "a.bin"
    def boom(descriptor):
        raise OSError("simulated crash mid-write")
    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="simulated crash"):
        atomic.atomic_write_bytes(target, b"x" * 100_000)
    assert not target.exists()  # canonical name never holds a partial write
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_escape_is_rejected(tmp_path: Path) -> None:
    policy = PathPolicy(root=tmp_path)
    with pytest.raises(ValueError):
        policy.resolve("../outside")
    with pytest.raises(ValueError):
        policy.resolve("C:/outside")


def test_symlink_component_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-v2"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        PathPolicy(root=tmp_path).resolve("link/file.txt")


def test_run_evidence_path_rejects_unsafe_components() -> None:
    from slm_train.artifacts.paths import run_evidence_path
    assert run_evidence_path("core", "core-x") == "runs/v2/core/core-x"
    for bad_namespace in ("", ".", "..", "a/b", "a\\b", "C:", "C:foo", "C:/x", "//server/share", " lead"):
        with pytest.raises(ValueError):
            run_evidence_path(bad_namespace, "core-x")
    for bad_run_id in ("", ".", "..", "a/b", "C:x", " lead", "a" * 65):
        with pytest.raises(ValueError):
            run_evidence_path("core", bad_run_id)


def test_log_segments_follow_design_layout_per_kind(tmp_path: Path) -> None:
    """Design 8: logs/<kind>/segment-NNNNNN.jsonl with per-kind sequences."""
    root = tmp_path / "run"
    append_log_segment(root, kind="lease", sequence=1, payload=b"l1")
    append_log_segment(root, kind="lease", sequence=2, payload=b"l2")
    append_log_segment(root, kind="metrics", sequence=1, payload=b"m1")
    assert (root / "logs" / "lease" / "segment-000001.jsonl").read_bytes() == b"l1"
    assert (root / "logs" / "lease" / "segment-000002.jsonl").read_bytes() == b"l2"
    assert (root / "logs" / "metrics" / "segment-000001.jsonl").read_bytes() == b"m1"
    assert not (root / "logs" / "metrics" / "segment-000002.jsonl").exists()
    index = LogIndex.load_from(root)
    lease = index.state_for("lease")
    assert lease.first_sequence == 1 and lease.last_sequence == 2
    metrics = index.state_for("metrics")
    assert metrics.first_sequence == 1 and metrics.last_sequence == 1
    with pytest.raises(FileExistsError):
        append_log_segment(root, kind="lease", sequence=1, payload=b"overwrite")
    with pytest.raises(ValueError, match="gap"):
        append_log_segment(root, kind="lease", sequence=4, payload=b"skip")
    with pytest.raises(ValueError, match="gap"):  # per-kind: metrics gap despite lease at 2
        append_log_segment(root, kind="metrics", sequence=3, payload=b"skip")
    with pytest.raises(ValueError, match="design stream"):
        append_log_segment(root, kind="bogus", sequence=1, payload=b"x")


def test_log_index_hash_chain_per_kind(tmp_path: Path) -> None:
    root = tmp_path / "run"
    append_log_segment(root, kind="lease", sequence=1, payload=b"alpha")
    index = LogIndex.load_from(root)
    first = index.state_for("lease").index_hash
    assert LogIndex.load_from(root).state_for("lease").previous_index_hash == ""
    append_log_segment(root, kind="lease", sequence=2, payload=b"beta")
    second_state = LogIndex.load_from(root).state_for("lease")
    assert second_state.previous_index_hash == first  # genuine chain to prior index hash


def test_log_index_load_detects_tampered_previous_hash(tmp_path: Path) -> None:
    import json as _json
    root = tmp_path / "run"
    append_log_segment(root, kind="metrics", sequence=1, payload=b"a")
    append_log_segment(root, kind="metrics", sequence=2, payload=b"b")
    path = root / "logs" / "log-manifest.json"
    data = _json.loads(path.read_text(encoding="utf-8"))
    data["kinds"]["metrics"]["previous_index_hash"] = "tampered"
    path.write_text(_json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="tampered"):
        LogIndex.load_from(root)


def test_manifest_write_is_read_back(tmp_path: Path) -> None:
    records = [{"path": "a.bin", "sha256": "0" * 64}, {"path": "b.bin", "sha256": "1" * 64}]
    text = write_manifest(tmp_path, "manifests/data.json", records)
    assert (tmp_path / "manifests" / "data.json").read_text(encoding="utf-8") == text
    with pytest.raises(ValueError, match="read-back"):
        write_manifest(tmp_path, "manifests/data.json", "not-json" * 100)


def test_completion_marker_written_last(tmp_path: Path) -> None:
    """COMPLETE.json only appears after manifest + payload hash read-back verify."""
    payloads = {"payload.bin": b"hello"}
    with pytest.raises(FileNotFoundError, match="manifest"):
        write_completion_marker(tmp_path, payloads, manifest_path="manifests/missing.json")
    write_manifest(tmp_path, "manifests/data.json", [{"path": "payload.bin"}])
    with pytest.raises(ValueError, match="hash"):
        write_completion_marker(tmp_path, payloads, manifest_path="manifests/data.json")
    import json as _json
    import hashlib as _hashlib
    expected = {"payload.bin": _hashlib.sha256(b"hello").hexdigest()}
    (tmp_path / "payload.bin").write_bytes(b"hello")  # payload must exist on disk
    (tmp_path / "manifests" / "data.json").write_text(
        _json.dumps({"manifest": expected}), encoding="utf-8")
    # in-memory payload diverges from the published file -> fail closed
    with pytest.raises(ValueError, match="hash"):
        write_completion_marker(
            tmp_path, {"payload.bin": b"jello"}, manifest_path="manifests/data.json")
    write_completion_marker(tmp_path, payloads, manifest_path="manifests/data.json")
    marker = _json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["status"] == "PASS"
    assert marker["payload_hashes"] == expected
    # the marker binds the exact published manifest bytes
    import hashlib as _h
    manifest_bytes = (tmp_path / "manifests" / "data.json").read_bytes()
    assert marker["manifest_hash"] == _h.sha256(manifest_bytes).hexdigest()
    from slm_train.artifacts.markers import verify_completion_marker
    assert verify_completion_marker(tmp_path, ["payload.bin"]) is True


def test_completion_marker_rejects_tampered_manifest(tmp_path: Path) -> None:
    import json as _json
    import hashlib as _h
    payloads = {"payload.bin": b"hello"}
    expected = {"payload.bin": _h.sha256(b"hello").hexdigest()}
    (tmp_path / "payload.bin").write_bytes(b"hello")
    write_manifest(tmp_path, "manifests/data.json", {"manifest": expected})
    write_completion_marker(tmp_path, payloads, manifest_path="manifests/data.json")
    (tmp_path / "manifests" / "data.json").write_bytes(b"changed")
    with pytest.raises(ValueError):  # fail closed on divergence (JSON or hash)
        write_completion_marker(tmp_path, payloads, manifest_path="manifests/data.json")
    from slm_train.artifacts.markers import verify_completion_marker
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        verify_completion_marker(tmp_path, ["payload.bin"])


def test_completion_marker_rejects_tampered_payload(tmp_path: Path) -> None:
    import hashlib as _h
    payloads = {"payload.bin": b"hello"}
    expected = {"payload.bin": _h.sha256(b"hello").hexdigest()}
    (tmp_path / "payload.bin").write_bytes(b"hello")
    write_manifest(tmp_path, "manifests/data.json", {"manifest": expected})
    write_completion_marker(tmp_path, payloads, manifest_path="manifests/data.json")
    (tmp_path / "payload.bin").write_bytes(b"jello")
    from slm_train.artifacts.markers import verify_completion_marker
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_completion_marker(tmp_path, ["payload.bin"])

def _published_marker(tmp_path: Path) -> "dict[str, str]":
    """One manifest-covered payload on disk plus a passing COMPLETE.json."""
    import hashlib as _h
    import json as _json
    expected = {"payload.bin": _h.sha256(b"hello").hexdigest()}
    (tmp_path / "payload.bin").write_bytes(b"hello")
    write_manifest(tmp_path, "manifests/data.json", {"manifest": expected})
    write_completion_marker(
        tmp_path, {"payload.bin": b"hello"}, manifest_path="manifests/data.json")
    return expected


def test_completion_marker_verify_fails_closed_without_payload_hashes(tmp_path: Path) -> None:
    """M-A: a COMPLETE.json whose payload_hashes key was deleted never verifies."""
    import json as _json
    from slm_train.artifacts.markers import verify_completion_marker
    _published_marker(tmp_path)
    marker = _json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    del marker["payload_hashes"]
    (tmp_path / "COMPLETE.json").write_text(_json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="missing the payload_hashes section"):
        verify_completion_marker(tmp_path, ["payload.bin"])


def test_completion_marker_verify_fails_closed_on_empty_payload_hashes(tmp_path: Path) -> None:
    """M-A: an emptied payload map never verifies; zero payloads need the
    explicit payload-free call path."""
    import json as _json
    from slm_train.artifacts.markers import (
        verify_completion_marker,
        verify_payload_free_completion_marker,
        write_payload_free_completion_marker,
    )
    _published_marker(tmp_path)
    marker = _json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    marker["payload_hashes"] = {}
    (tmp_path / "COMPLETE.json").write_text(_json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="certifies no payloads"):
        verify_completion_marker(tmp_path, ["payload.bin"])
    assert verify_payload_free_completion_marker(tmp_path) is True  # only via the explicit path
    with pytest.raises(ValueError, match="at least one payload"):
        write_completion_marker(tmp_path, {}, manifest_path="manifests/data.json")
    # explicit zero-payload publication round-trips through its own path
    write_payload_free_completion_marker(tmp_path, "manifests/data.json")
    assert verify_payload_free_completion_marker(tmp_path) is True


def test_completion_marker_verify_fails_closed_on_omitted_expected_payload(tmp_path: Path) -> None:
    """M-A: a map that omits one payload the caller expects never verifies."""
    import hashlib as _h
    import json as _json
    from slm_train.artifacts.markers import verify_completion_marker
    expected = {
        "payload.bin": _h.sha256(b"hello").hexdigest(),
        "second.bin": _h.sha256(b"world").hexdigest(),
    }
    (tmp_path / "payload.bin").write_bytes(b"hello")
    (tmp_path / "second.bin").write_bytes(b"world")
    write_manifest(tmp_path, "manifests/data.json", {"manifest": expected})
    write_completion_marker(
        tmp_path, {"payload.bin": b"hello"}, manifest_path="manifests/data.json")
    assert verify_completion_marker(tmp_path, ["payload.bin"]) is True
    with pytest.raises(ValueError, match="second.bin"):
        verify_completion_marker(tmp_path, ["payload.bin", "second.bin"])
    with pytest.raises(ValueError, match="expected_payloads is empty"):
        verify_completion_marker(tmp_path, [])
    from slm_train.artifacts.markers import verify_payload_free_completion_marker
    with pytest.raises(ValueError, match="certifies payloads"):
        verify_payload_free_completion_marker(tmp_path)
    # a marker that also certifies an unrelated payload is not a zero-payload one
    marker = _json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    marker["payload_hashes"]["extra.bin"] = "0" * 64
    (tmp_path / "COMPLETE.json").write_text(_json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="certified payload is missing: extra.bin"):
        verify_completion_marker(tmp_path, ["payload.bin"])


def test_log_segment_scan_rejects_foreign_file_in_kind_dir(tmp_path: Path) -> None:
    """M-C: a non-segment file fails closed with a message naming the file."""
    root = tmp_path / "run"
    kind_dir = root / "logs" / "lease"
    kind_dir.mkdir(parents=True)
    (kind_dir / "segment-000001.jsonl").write_bytes(b"l1")
    (kind_dir / "segment-abc.jsonl").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="segment-abc.jsonl"):
        append_log_segment(root, kind="lease", sequence=2, payload=b"l2")
    assert not (kind_dir / "segment-000002.jsonl").exists()  # nothing published
    (kind_dir / "segment-abc.jsonl").unlink()
    (kind_dir / "segment-12.jsonl").write_bytes(b"foreign")  # wrong digit count too
    with pytest.raises(ValueError, match="segment-12.jsonl"):
        append_log_segment(root, kind="lease", sequence=2, payload=b"l2")
    (kind_dir / "segment-12.jsonl").unlink()
    append_log_segment(root, kind="lease", sequence=2, payload=b"l2")
    assert (kind_dir / "segment-000002.jsonl").read_bytes() == b"l2"

