"""Typed orchestration errors.

Every stage failure is one of these. There is no generic fallback path: the
engine re-raises, and a caller decides. A BLOCKED stage is never silently
treated as a failure, and a failure is never silently treated as a skip.
"""
from __future__ import annotations


class OrchestrationError(RuntimeError):
    """Base class for every orchestration failure."""


class StageFailed(OrchestrationError):
    """A stage raised, or returned a label that is not a decided pass."""


class StageBlocked(OrchestrationError):
    """A stage is BLOCKED: something external must change before it can proceed."""


class MissingStageInput(OrchestrationError):
    """A stage declared a required input that has not been produced."""


class GraphError(ValueError):
    """The stage graph itself is malformed."""
