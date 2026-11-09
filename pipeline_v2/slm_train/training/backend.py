"""Backend Protocol for the training orchestrator (no HF types here)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from slm_train.training.spec import TrainingContext, TrainingResult

__all__ = ["TrainerBackend"]


@runtime_checkable
class TrainerBackend(Protocol):
    """Every stage consumes the typed context and returns a typed result."""

    def prepare(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def fit(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def evaluate(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def checkpoint(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def resume(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def finalize(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError