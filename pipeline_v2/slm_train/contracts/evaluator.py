"""Evaluator manifest: the stored hash is publication metadata only."""
from typing import ClassVar, Mapping, Optional

from slm_train.contracts.base import SemanticManifest


class EvaluatorManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ("manifest_hash",)

    evaluator_ids: tuple[str, ...]
    evaluator_versions: Mapping[str, str] = {}
    suite_gates: tuple[str, ...] = ()
    metric_schemas: Mapping[str, str] = {}
    cadence: str = ""
    thresholds: Mapping[str, float] = {}
    threshold_directions: Mapping[str, str] = {}
    manifest_hash: Optional[str] = None
