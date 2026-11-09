"""Evaluation contracts: the suite definition and its metric schema."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MetricSpec", "SuiteSpec", "Threshold"]


@dataclass(frozen=True)
class Threshold:
    """One bound. min means the value may not exceed it."""

    value: float
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in ("min", "max"):
            raise ValueError("threshold direction must be min or max: " + repr(self.direction))


@dataclass(frozen=True)
class MetricSpec:
    """One required metric: what it is, how it is bounded, and its cadence."""

    name: str
    definition: str
    threshold: "Threshold | None" = None
    cadence: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a metric needs a non-empty name")
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError("metric " + str(self.name) + " needs a definition")
        for step in self.cadence:
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError("cadence steps must be non-negative integers")
        if len(set(self.cadence)) != len(self.cadence):
            raise ValueError("metric " + self.name + " repeats a cadence step")


@dataclass(frozen=True)
class SuiteSpec:
    """The exact metric set an evaluator must produce, and nothing else."""

    evaluator: str
    version: str
    metrics: tuple
    split_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, str) or not self.evaluator.strip():
            raise ValueError("an evaluator needs a non-empty id")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("an evaluator needs a non-empty version")
        if not self.metrics:
            raise ValueError("a suite must declare at least one metric")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("a suite cannot declare the same metric twice")

    def names(self) -> frozenset:
        return frozenset(metric.name for metric in self.metrics)

    def spec_for(self, name: str) -> "MetricSpec":
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError("metric is not in this suite: " + str(name))

    def definitions(self) -> dict:
        return {metric.name: metric.definition for metric in self.metrics}

    def all_cadence(self) -> dict:
        return {m.name: tuple(m.cadence) for m in self.metrics if m.cadence}
