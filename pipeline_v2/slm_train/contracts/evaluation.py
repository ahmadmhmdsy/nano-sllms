"""Evaluation report contract with exact validation vocabulary."""
from typing import ClassVar, Literal, Mapping, Optional

from pydantic import Field

from slm_train.contracts.base import SemanticManifest, ValidationLabel


class EvaluationReport(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    evaluator: str
    evaluator_version: str
    expected_suite_gates: tuple[str, ...] = ()
    actual_suite_gates: tuple[str, ...] = ()
    expected_evaluator_manifest_hash: Optional[str] = None
    actual_evaluator_manifest_hash: Optional[str] = None
    run_id: str
    checkpoint_identity: str = ""
    data_hash: str = ""
    tokenizer_hash: str = ""
    device_identity: str = ""
    seed: Optional[int] = None
    split_identity: str = ""
    metric_schema_identity: str = ""
    metric_definitions: Mapping[str, str] = {}
    metric_values: Mapping[str, float] = {}
    baseline: Optional[float] = None
    gate_threshold: Optional[float] = None
    threshold_direction: Optional[str] = None
    details: Mapping[str, object] = {}
    status: ValidationLabel
    artifact_paths: tuple[str, ...] = ()
