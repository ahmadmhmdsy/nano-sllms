"""Gate checks: the only way a metric becomes a PASS.

Every function here either returns None or raises GateFailure. There is no
boolean return, no warning, and no partial credit: a caller that ignores the
exception has still failed, because nothing was recorded as passing.

The checks refuse in this order, and the order matters: direction, then shape,
then finiteness, then comparison. A gate never compares a value it has not first
proved finite, so a NaN can never satisfy a bound.
"""
from __future__ import annotations

from math import isfinite

__all__ = [
    "DIRECTIONS",
    "GateFailure",
    "require_cadence",
    "require_complete_suite",
    "require_finite",
    "require_threshold",
]

DIRECTIONS = ("min", "max")


class GateFailure(ValueError):
    """A gate did not pass. The message is the evidence."""


def require_finite(metric: str, value: object) -> float:
    """Return the value as a finite float, or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reason = "non-finite metric: " + str(metric) + " is not a number: " + repr(value)
        raise GateFailure(reason)
    if not isfinite(float(value)):
        reason = "non-finite metric: " + str(metric) + " = " + repr(value)
        raise GateFailure(reason)
    return float(value)


def require_threshold(metric: str, value: object, threshold: object, direction: str) -> float:
    """Enforce one bound. The direction is checked BEFORE the comparison."""
    if direction not in DIRECTIONS:
        raise GateFailure("unsupported metric direction: " + repr(direction))
    measured = require_finite(metric, value)
    bound = require_finite(metric + " threshold", threshold)
    if direction == "min" and measured > bound:
        reason = "metric " + metric + " exceeds threshold: " + str(measured) + " > " + str(bound)
        raise GateFailure(reason)
    if direction == "max" and measured < bound:
        reason = "metric " + metric + " is below threshold: " + str(measured) + " < " + str(bound)
        raise GateFailure(reason)
    return measured


def require_complete_suite(expected, actual, *, expected_evaluator_hash: str, actual_evaluator_hash: str) -> None:
    """The suite must match exactly, and the evaluator identity must match."""
    if not isinstance(expected, (set, frozenset)) or not isinstance(actual, (set, frozenset)):
        raise GateFailure("suite membership must be compared as sets")
    # Two absent hashes compare equal, so equality alone would accept a report
    # that records no evaluator manifest at all. Absence is not identity.
    for name, value in (
        ("expected_evaluator_hash", expected_evaluator_hash),
        ("actual_evaluator_hash", actual_evaluator_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise GateFailure("evaluator manifest is absent: " + name)
    if expected_evaluator_hash != actual_evaluator_hash:
        reason = "evaluator manifest mismatch: expected " + str(expected_evaluator_hash)
        raise GateFailure(reason + ", got " + str(actual_evaluator_hash))
    missing = set(expected) - set(actual)
    if missing:
        raise GateFailure("missing required metrics: " + str(sorted(missing)))
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise GateFailure("unexpected metrics: " + str(sorted(unexpected)))


def require_cadence(metric: str, observed, expected_cadence) -> None:
    """A metric missing a configured evaluation point is FAIL, never PASS."""
    gaps = set(expected_cadence) - set(observed)
    if gaps:
        reason = "metric " + str(metric) + " missing at configured cadence: "
        raise GateFailure(reason + str(sorted(gaps)))
