"""Health policy: what a run concludes about itself.

The rules are fixed and none of them changes a hyperparameter:

  * loss and gradients must be finite; NaN or Inf is FAIL,
  * out of memory is BLOCKED, never a silent batch-size reduction,
  * shared-memory spill is OOM-equivalent,
  * anything the run cannot classify is FAIL, not PASS.

A health result is a decision with a reason. It never retries, never resumes,
never shrinks a batch, and never changes a precision. The caller decides what a
BLOCKED out-of-memory means for a NEW run; this module only classifies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

__all__ = [
    "HEALTH_LABELS",
    "HealthDecision",
    "classify_health",
    "is_finite_value",
]

HEALTH_LABELS = ("PASS", "FAIL", "BLOCKED", "NEEDS_USER_DECISION")

#: substrings that identify an allocator out-of-memory in a torch message
OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cuda oom",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "alloc_failed",
)

#: substrings that identify a shared-memory spill, which is OOM-equivalent
SHARED_MEMORY_MARKERS: tuple[str, ...] = (
    "shared memory",
    "shm",
    "bus error",
)


def is_finite_value(value: object) -> bool:
    """True only for a real, finite number. None, NaN, Inf, and text are not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isfinite(float(value))


@dataclass(frozen=True)
class HealthDecision:
    """A classification, with the facts it was made from."""

    label: str
    reason: str
    facts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in HEALTH_LABELS:
            listed = ", ".join(HEALTH_LABELS)
            raise ValueError(
                "health label must be one of " + listed + ": " + repr(self.label)
        )
        if not self.reason.strip():
            raise ValueError("a health decision must state a reason")

    @property
    def oom(self) -> bool:
        return self.label == "BLOCKED" and self.facts.get("kind") == "oom"


def _classify_memory(text: str) -> str | None:
    """oom, shared-memory, or None. Matched case-insensitively."""
    lowered = text.lower()
    if any(marker in lowered for marker in OOM_MARKERS):
        return "oom"
    if any(marker in lowered for marker in SHARED_MEMORY_MARKERS):
        return "shared_memory"
    return None


def classify_health(
    *,
    loss: object = None,
    gradient_norm: object = None,
    message: str = "",
    facts: dict | None = None,
    require_finite: bool = True,
) -> HealthDecision:
    """Classify a health observation. Never retries and never adjusts anything."""
    observed = dict(facts or {})
    memory = _classify_memory(message)
    if memory is not None:
        kind = "oom" if memory == "oom" else "oom"
        label = "BLOCKED"
        reason = (
            "out of memory: the run is blocked and the last valid checkpoint "
            "is preserved; no hyperparameter was changed"
        )
        observed["kind"] = kind
        observed["memory_class"] = memory
        return HealthDecision(label=label, reason=reason, facts=observed)
    if require_finite:
        for name, value in (("loss", loss), ("gradient_norm", gradient_norm)):
            if value is None:
                continue
            if not is_finite_value(value):
                observed["kind"] = "non_finite"
                observed["field"] = name
                reason = name + " is not finite: " + repr(value)
                return HealthDecision(label="FAIL", reason=reason, facts=observed)
    return HealthDecision(
        label="PASS", reason="all observed values are finite", facts=observed,
    )
