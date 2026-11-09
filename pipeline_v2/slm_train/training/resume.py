"""Exact resume: the same work, or nothing at all.

Resume is not "close enough". Every semantic identity that could change a
result is compared, including the exact command, the working directory, the
allowlisted environment, and the device. A change to any of them is a refusal,
not a warning, and the caller must start a new run in a new destination.

The automatic resume path is the exact original command with zero added flags.
This module never proposes a command; it only decides whether two identities
are the same identity.
"""
from __future__ import annotations

from typing import Mapping

__all__ = [
    "RESUME_IDENTITY_KEYS",
    "ResumeMismatch",
    "assert_exact_resume",
    "resume_identity_differences",
]

#: Every identity an exact resume must match. Order is the report order.
RESUME_IDENTITY_KEYS: tuple[str, ...] = (
    "config",
    "code",
    "data",
    "tokenizer",
    "model",
    "feature_set",
    "evaluator",
    "selection_policy",
    "command",
    "working_directory",
    "environment",
    "driver",
    "package",
    "runtime",
    "device",
)


class ResumeMismatch(ValueError):
    """A resume was requested that is not the exact same work."""


def _identity_view(values: Mapping[str, str], label: str) -> dict[str, str]:
    """A plain copy of the declared identity keys, with every gap reported."""
    if not isinstance(values, Mapping):
        raise ResumeMismatch(label + " resume identity must be a mapping")
    missing = [key for key in RESUME_IDENTITY_KEYS if key not in values]
    if missing:
        raise ResumeMismatch(label + " missing resume identity: " + ", ".join(missing))
    empties = [key for key in RESUME_IDENTITY_KEYS if not str(values[key]).strip()]
    if empties:
        raise ResumeMismatch(label + " empty resume identity: " + ", ".join(empties))
    return {key: str(values[key]) for key in RESUME_IDENTITY_KEYS}


def resume_identity_differences(
    saved: Mapping[str, str], current: Mapping[str, str]
) -> list[str]:
    """The identity keys that differ. Missing keys are reported as differences."""
    left = _identity_view(saved, "saved")
    right = _identity_view(current, "current")
    return [key for key in RESUME_IDENTITY_KEYS if left[key] != right[key]]


def assert_exact_resume(saved: Mapping[str, str], current: Mapping[str, str]) -> None:
    """Raise ResumeMismatch unless every identity key matches exactly."""
    changed = resume_identity_differences(saved, current)
    if changed:
        raise ResumeMismatch("changed resume identity: " + ", ".join(changed))
