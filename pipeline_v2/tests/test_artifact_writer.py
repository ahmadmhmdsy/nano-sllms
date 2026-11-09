"""A machine-readable record of what a test run actually did.

A console line is not evidence: it scrolls away, it is not reproducible, and
nobody can hash it. So every session writes a JSON artifact recording the exact
command, the counts, the completion time, and a label, and the writer reopens
that file, re-parses it, and re-checks its own hash before returning it. A
result is not a PASS until the artifact has been read back and verified.

Previous artifacts are never deleted. Each session gets its own directory, so
the history of what passed and what failed is itself the record.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes

__all__ = ["ARTIFACT_ROOT", "ARTIFACT_NAME", "TestArtifactWriter"]

ARTIFACT_ROOT = Path("pipeline_v2/test-artifacts")
ARTIFACT_NAME = "summary.json"


class TestArtifactWriter:
    """Writes, then verifies, one session summary."""

    @staticmethod
    def _evidence_argument(value: str) -> str:
        """A command argument inside the repository becomes a relative path.

        An absolute local path in a committed artifact leaks the machine layout
        and does not reproduce anywhere else, so it is canonicalized first.
        A path outside the repository cannot be made relative, so it is
        reduced to its basename: which interpreter ran the command is a
        machine fact, not evidence.
        """
        root = Path.cwd().resolve()
        candidate = Path(value)
        if not candidate.is_absolute():
            return value
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
        return candidate.name

    @staticmethod
    def _canonical(payload: dict) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def write_result(self, session_id: str, exitstatus: int, test_counts: dict) -> Path:
        if not session_id or "/" in session_id or "\\" in session_id:
            raise ValueError("a session id must be one path segment: " + repr(session_id))
        payload = {
            "schema_version": 1,
            "session_id": session_id,
            "command": [self._evidence_argument(value) for value in sys.argv],
            "exitstatus": int(exitstatus),
            "test_counts": {str(k): int(v) for k, v in dict(test_counts).items()},
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_root": ".",
            "label": "PASS" if int(exitstatus) == 0 else "FAIL",
        }
        encoded = self._canonical(payload)
        payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
        target = ARTIFACT_ROOT / session_id / ARTIFACT_NAME
        atomic_write_bytes(
            target, json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        )
        return self.verify(target)

    @staticmethod
    def verify(target: Path) -> Path:
        """Reopen, re-parse, and re-hash. A mismatch is a hard failure."""
        reopened = json.loads(Path(target).read_text(encoding="utf-8"))
        if not isinstance(reopened, dict):
            raise RuntimeError("test artifact is not a JSON object: " + str(target))
        stored = reopened.get("payload_sha256")
        if not stored:
            raise RuntimeError("test artifact carries no payload hash: " + str(target))
        body = {key: value for key, value in reopened.items() if key != "payload_sha256"}
        canonical = TestArtifactWriter._canonical(body)
        if hashlib.sha256(canonical).hexdigest() != stored:
            raise RuntimeError("test artifact hash mismatch: " + str(target))
        if reopened.get("label") not in ("PASS", "FAIL"):
            raise RuntimeError("test artifact label is not a validation label")
        return Path(target)
