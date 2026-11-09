"""Evaluation: declared suites, real measurements, and gates that cannot pass.

Nothing here discovers an evaluator at runtime. The suite catalogue is a static
declaration, every metric value comes from a measurement the caller made, and the
only route to PASS is a gate that compared a finite value against an explicit bound.
"""

from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold
from slm_train.evaluation.gates import GateFailure
from slm_train.evaluation.gates import require_cadence
from slm_train.evaluation.gates import require_complete_suite
from slm_train.evaluation.gates import require_finite
from slm_train.evaluation.gates import require_threshold
from slm_train.evaluation.diacritizer import (
    DiacritizerArtifactError,
    evaluator_manifest_hash,
    report_from_artifact,
)
from slm_train.evaluation.diacritizer_suites import DIACRITIZER_SUITES, diacritizer_suite_for
from slm_train.evaluation.suites import EXTERNAL_GATE_NAMES, SUITES, suite_for

__all__ = [
    "DIACRITIZER_SUITES",
    "DiacritizerArtifactError",
    "EXTERNAL_GATE_NAMES",
    "GateFailure",
    "MetricSpec",
    "SUITES",
    "SuiteSpec",
    "Threshold",
    "diacritizer_suite_for",
    "evaluator_manifest_hash",
    "report_from_artifact",
    "require_cadence",
    "require_complete_suite",
    "require_finite",
    "require_threshold",
    "suite_for",
]
