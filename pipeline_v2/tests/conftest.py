"""Shared V2 test configuration."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

# The V2 package is imported by path so the suite runs from a checkout without
# requiring an install step.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

ARTIFACT_ROOT = PIPELINE_ROOT / "test-artifacts"


def pytest_configure(config) -> None:
    """Mint the session id BEFORE collection, so it exists for the whole run."""
    config.stash["v2_session_id"] = uuid.uuid4().hex[:16]


def pytest_sessionfinish(session, exitstatus) -> None:
    """Write the machine-readable summary, and never fail the run over it.

    A summary that cannot be written is worth reporting, but it must not turn a
    green suite red: the suite's own result is the authoritative one, and the
    artifact is the record of it.
    """
    session_id = session.config.stash.get("v2_session_id", None)
    if not session_id:
        return
    from tests.test_artifact_writer import TestArtifactWriter

    counts = {
        "collected": session.testscollected,
        "failed": session.testsfailed,
    }
    try:
        target = TestArtifactWriter().write_result(session_id, int(exitstatus), counts)
    except Exception as error:  # pragma: no cover - reporting path only
        print("WARNING: could not write the test artifact: " + str(error))
        return
    try:
        target.relative_to(Path.cwd().resolve())
    except ValueError:
        print("WARNING: test artifact written outside the repository: " + str(target))
        return
    print("test artifact: " + target.as_posix())
