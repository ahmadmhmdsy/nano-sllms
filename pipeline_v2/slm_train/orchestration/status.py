"""Status and diagnostic evidence, written whether the run passed or failed."""
from __future__ import annotations

from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_json
from slm_train.orchestration.stages import RunResult

STATUS_RELATIVE = ("status.json",)


def status_path(run_root: Path) -> Path:
    return Path(run_root) / STATUS_RELATIVE[0]


def write_status(run_root: Path, result: RunResult) -> Path:
    """Record the terminal status. Mutable, so a retry can replace it."""
    payload = {
        "status": result.status,
        "completed_stages": list(result.completed_stages),
        "failed_stage": result.failed_stage,
        "reason": result.reason,
    }
    target = status_path(run_root)
    atomic_write_json(target, payload, replace_existing=True)
    return target
