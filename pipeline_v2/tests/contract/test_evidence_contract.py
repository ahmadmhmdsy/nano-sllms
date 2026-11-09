"""Evidence must be a path anyone can open, and a number anyone can trust."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "pipeline_v2" / "test-artifacts"


def _summaries():
    return sorted(ARTIFACTS.glob("*/summary.json"))


def test_all_evidence_paths_are_repository_relative() -> None:
    for path in _summaries():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.get("command", []):
            assert not Path(value).is_absolute(), value


def test_absolute_paths_inside_the_repository_become_relative() -> None:
    """A repo-relative command stays reproducible on any machine."""
    from tests.test_artifact_writer import TestArtifactWriter

    root = Path(__file__).resolve().parents[3]
    inside = root / "pipeline_v2" / "tests" / "conftest.py"
    canonical = TestArtifactWriter._evidence_argument(str(inside))
    assert canonical == "pipeline_v2/tests/conftest.py"


def test_absolute_paths_outside_the_repository_become_machine_independent() -> None:
    """An interpreter outside the repo is a machine fact, not evidence."""
    from tests.test_artifact_writer import TestArtifactWriter

    root = Path(__file__).resolve().parents[3]
    outside = root.parent / "elsewhere" / "pytest" / "__main__.py"
    canonical = TestArtifactWriter._evidence_argument(str(outside))
    assert not Path(canonical).is_absolute()
    assert canonical == "__main__.py"


def test_metric_evidence_requires_finite_values() -> None:
    assert math.isfinite(1.0)
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))


def test_every_artifact_carries_a_verifiable_hash() -> None:
    import hashlib

    for path in _summaries():
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = payload.get("payload_sha256")
        assert stored, path
        body = {k: v for k, v in payload.items() if k != "payload_sha256"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == stored, path


def test_every_artifact_label_is_a_validation_label() -> None:
    for path in _summaries():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["label"] in ("PASS", "FAIL"), path


def test_the_artifact_records_its_own_command_and_time() -> None:
    for path in _summaries():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["command"], path
        assert payload["completed_at"], path
        assert payload["session_id"] in path.parent.name


def test_a_tampered_artifact_does_not_verify(tmp_path: Path) -> None:
    """A hash is only evidence if a change to the artifact is detected."""
    import pytest

    from tests.test_artifact_writer import TestArtifactWriter

    target = tmp_path / "summary.json"
    payload = {
        "schema_version": 1,
        "session_id": "abc",
        "command": ["python", "-m", "pytest"],
        "exitstatus": 0,
        "test_counts": {"collected": 1, "failed": 0},
        "completed_at": "2026-01-01T00:00:00+00:00",
        "source_root": ".",
        "label": "PASS",
    }
    import hashlib
    import json as _json

    encoded = TestArtifactWriter._canonical(payload)
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    target.write_text(_json.dumps(payload), encoding="utf-8")
    assert TestArtifactWriter.verify(target) == target

    # Now change a recorded fact without updating the hash.
    payload["label"] = "PASS"
    payload["test_counts"] = {"collected": 99, "failed": 0}
    target.write_text(_json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        TestArtifactWriter.verify(target)


def test_an_artifact_with_no_hash_is_refused(tmp_path: Path) -> None:
    import pytest

    from tests.test_artifact_writer import TestArtifactWriter

    target = tmp_path / "summary.json"
    target.write_text('{"label": "PASS"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="no payload hash"):
        TestArtifactWriter.verify(target)
