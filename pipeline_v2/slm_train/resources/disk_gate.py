"""Disk-space gate: refuse to train when checkpoint evidence cannot fit.

An Errno 28 discovered at checkpoint write time is the worst possible
evidence loss: the compute is spent and the checkpoint is not. The arithmetic
of what the run WILL need is known before a single micro-step, so the gate
asks for it up front. Observation and refusal only; never deletes.
"""
from __future__ import annotations

from pathlib import Path


def required_free_bytes(checkpoint_bytes: int, keep: int, margin: int) -> int:
    """Bytes one run root must be able to take: current keeps plus in-flight."""
    if checkpoint_bytes < 0 or keep < 0 or margin < 1:
        raise ValueError("checkpoint_bytes and keep must be non-negative; margin at least 2")
    return int(checkpoint_bytes) * int(keep) * int(margin)


def free_space_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding one path.

    A path that does not exist yet is legal evidence: walk up to the nearest
    existing ancestor, because the filesystem that will hold the file is
    decided by the first directory that already exists.
    """
    import shutil

    current = Path(path)
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(str(path))
        current = current.parent
    return int(shutil.disk_usage(str(current)).free)


def checkpoint_estimate(log_stub: Path | None) -> int:
    """Best effort one-checkpoint byte estimate, from evidence if available."""
    if log_stub is not None:
        try:
            import json

            manifest = json.loads(
                (Path(log_stub) / "log_stub.json").read_text("utf-8")
            )
            return int(manifest["checkpoint_bytes"])
        except Exception:
            pass
    return 900_000_000  # the measured 213M-parameter fp32 master + AdamW state


def disk_gate_decision(
    root: Path,
    *,
    checkpoint_bytes: int | None = None,
    keep: int = 3,
    margin: int = 2,
) -> dict:
    """PASS or BLOCKED (or FAIL when the arithmetic itself is unusable)."""
    from slm_train.resources.gpu_probe import UNREADABLE

    if checkpoint_bytes is None:
        checkpoint_bytes = 900_000_000
    try:
        required = required_free_bytes(
            checkpoint_bytes=int(checkpoint_bytes), keep=int(keep), margin=int(margin)
        )
    except Exception as error:
        return {
            "label": "FAIL",
            "reason": "the disk budget could not be computed: " + str(error),
        }
    if int(checkpoint_bytes) <= 0:
        return {
            "label": "FAIL",
            "reason": "the checkpoint size estimate must be positive",
        }
    free = free_space_bytes(root)
    decision = {
        "root": str(root),
        "required_free_bytes": required,
        "free_bytes": free,
        "checkpoint_bytes": int(checkpoint_bytes),
        "kept_checkpoints": int(keep),
        "margin": int(margin),
    }
    if free < required:
        decision["label"] = "BLOCKED"
        decision["reason"] = (
            "free disk space " + str(free) + " bytes is below the " + str(required)
            + " bytes this run needs for itself and its checkpoints; free space"
            + " before training starts, so no compute is wasted on a run that"
            + " must die at its first checkpoint write."
        )
        return decision
    decision["label"] = "PASS"
    return decision
