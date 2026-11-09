"""Typed training specification: context, result, and events (no HF types)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from slm_train.config.models import ConfigDocument
from slm_train.contracts.base import ValidationLabel
from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.contracts.evaluation import EvaluationReport
from slm_train.contracts.evaluator import EvaluatorManifest
from slm_train.contracts.model import ModelManifest
from slm_train.contracts.run import RunIdentityManifest
from slm_train.contracts.tokenized import TokenizedDataManifest
from slm_train.contracts.tokenizer import TokenizerManifest
from slm_train.training.events import TrainingEvent
from slm_train.training.state import TrainingCursor

__all__ = [
  "VALIDATION_LABELS", "TrainingContext", "TrainingEvent", "TrainingResult",
]

#: The single validation vocabulary (contracts/base ValidationLabel values).
VALIDATION_LABELS = ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION")


@dataclass(frozen=True)
class TrainingContext:
    """Everything the backend stage needs, resolved and typed (HF-free)."""

    config: ConfigDocument
    run_identity: RunIdentityManifest
    model_manifest: ModelManifest
    tokenizer_manifest: TokenizerManifest
    dataset_manifest: TokenizedDataManifest
    evaluator_manifest: EvaluatorManifest
    cursor: TrainingCursor
    device: str
    seed: int
    event_sink: Callable[[TrainingEvent], None]
    #: explicit repository-relative output root; required, no silent default
    output_root: str


@dataclass(frozen=True)
class TrainingResult:
    """Typed stage outcome; the orchestrator never sees HF internals.

    Construct-time contract enforcement: the label must use the exact
    validation vocabulary and event sequences must be strictly increasing
    (monotonic, no duplicates).
    """

    label: ValidationLabel
    events: tuple[TrainingEvent, ...]
    checkpoint: CheckpointManifest | None
    evaluations: tuple[EvaluationReport, ...]
    diagnostic: str | None

    def __post_init__(self) -> None:
        if self.label not in VALIDATION_LABELS:
            raise ValueError(
                f"invalid validation label: {self.label!r} "
                f"(allowed: {list(VALIDATION_LABELS)})"
            )
        sequences = [event.sequence for event in self.events]
        if any(not isinstance(seq, int) or seq < 1 for seq in sequences):
            raise ValueError("event sequences must be positive integers")
        for earlier, later in zip(sequences, sequences[1:]):
            if later <= earlier:
                raise ValueError(
                    "event sequences must be strictly increasing; got "
                    f"{sequences!r} (non-monotonic or duplicated sequence)"
                )
