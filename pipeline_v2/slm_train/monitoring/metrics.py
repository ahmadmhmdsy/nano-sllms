"""Metric values, checked for finiteness before they are ever recorded.

A metric is a name, a finite number, and nothing else. A NaN or Inf loss is
rejected here rather than written, so a corrupt value can never reach a
published segment and later be read back as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from slm_train.monitoring.health import is_finite_value

__all__ = ["MetricError", "MetricValue", "require_finite_metric"]


class MetricError(ValueError):
    """A metric value is not a finite number."""


@dataclass(frozen=True)
class MetricValue:
    """One named finite observation at one step."""

    name: str
    value: float
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise MetricError("a metric needs a non-empty name")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise MetricError("a metric step must be a non-negative integer")
        require_finite_metric(self.name, self.value)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": float(self.value), "step": self.step}


def require_finite_metric(name: str, value: object) -> float:
    """Return the value as a float, or refuse. Never rounds or substitutes."""
    if not is_finite_value(value):
        raise MetricError("metric " + str(name) + " is not finite: " + repr(value))
    return float(value)
