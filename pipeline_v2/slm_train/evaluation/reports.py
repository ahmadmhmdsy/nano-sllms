"""Evaluation reports, written on success AND on failure.

A failed gate is the moment a report matters most, so the report writer never
short-circuits on failure: it builds the report from whatever was measured,
records the exact reason, and publishes it. A run that fails without a report
has failed without evidence.

Every report carries the evaluator identity, the expected and actual suite,
both evaluator manifest hashes, and the run/checkpoint/data/tokenizer/device/seed
identity, so a report can never be silently attributed to another run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from slm_train.artifacts.manifest import write_manifest
from slm_train.contracts import VALIDATION_LABELS
from slm_train.contracts.evaluation import EvaluationReport
from slm_train.evaluation.contracts import SuiteSpec
from slm_train.evaluation.gates import GateFailure, require_cadence, require_complete_suite, require_finite, require_threshold

__all__ = [
    "REPORTS_DIR",
    "REPORT_NAME",
    "build_report",
    "evaluate",
    "report_path",
    "write_report",
]

REPORTS_DIR = "evaluations"
REPORT_NAME = "report.json"

IDENTITY_KEYS = ("run_id", "data_hash", "tokenizer_hash", "device_identity")


def _identity_fields(identity: dict) -> dict:
    """The identity every report must carry, copied from the caller."""
    if not isinstance(identity, dict):
        raise ValueError("report identity must be a mapping")
    missing = [key for key in IDENTITY_KEYS if not str(identity.get(key, "")).strip()]
    if missing:
        raise ValueError("report identity is missing: " + ", ".join(missing))
    return dict(identity)


def _reportable(values: dict) -> dict:
    """Only values a report can actually hold, so a FAIL report is always buildable.

    An evaluator can hand back a bool, a string, or NaN. The gate refuses all of
    them, but the report must still record WHY, and a report carrying the bad
    value would itself be unbuildable. Dropped entries are named in the details
    rather than silently discarded.
    """
    from math import isfinite

    kept = {}
    dropped = []
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            dropped.append(name + "=" + repr(value))
        else:
            kept[name] = float(value)
    return kept, dropped


def build_report(suite: SuiteSpec, identity: dict, values: dict, *, expected_evaluator_hash: str, actual_evaluator_hash: str, status: str, reason: str = "", details: dict = None, artifact_paths: tuple = ()) -> EvaluationReport:
    """A report, whatever the outcome. Refuses only an impossible request."""
    if status not in VALIDATION_LABELS:
        raise ValueError("report status must be a validation label: " + repr(status))
    if status != "PASS" and not reason.strip():
        raise ValueError("a non-PASS report must state a reason")
    if not isinstance(values, dict):
        raise ValueError("metric values must be a mapping")
    values, dropped = _reportable(values)
    facts = _identity_fields(identity)
    recorded = dict(details or {})
    recorded["reason"] = reason
    if dropped:
        recorded["unreportable_values"] = dropped
    return EvaluationReport(
        schema_version=1,
        evaluator=suite.evaluator,
        evaluator_version=suite.version,
        expected_suite_gates=tuple(sorted(suite.names())),
        actual_suite_gates=tuple(sorted(values)),
        expected_evaluator_manifest_hash=expected_evaluator_hash,
        actual_evaluator_manifest_hash=actual_evaluator_hash,
        run_id=facts["run_id"],
        checkpoint_identity=str(facts.get("checkpoint_identity", "")),
        data_hash=str(facts["data_hash"]),
        tokenizer_hash=str(facts["tokenizer_hash"]),
        device_identity=str(facts["device_identity"]),
        seed=facts.get("seed"),
        split_identity=suite.split_identity,
        metric_schema_identity=suite.evaluator + "/" + suite.version,
        metric_definitions=suite.definitions(),
        metric_values=dict(values),
        baseline=facts.get("baseline"),
        gate_threshold=facts.get("gate_threshold"),
        threshold_direction=facts.get("threshold_direction"),
        details=recorded,
        status=status,
        artifact_paths=tuple(artifact_paths),
    )


def report_path(run_root: Path, evaluation_id: str) -> Path:
    """evaluations/<evaluation_id>/report.json. Re-evaluation uses a new id."""
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ValueError("an evaluation needs a non-empty id")
    if "/" in evaluation_id or "\\" in evaluation_id or evaluation_id in (".", ".."):
        raise ValueError("an evaluation id must be a single path segment: " + repr(evaluation_id))
    return Path(run_root) / REPORTS_DIR / evaluation_id / REPORT_NAME


def write_report(run_root: Path, evaluation_id: str, report: EvaluationReport) -> str:
    """Publish the report create-if-absent and return its relative path."""
    destination = report_path(run_root, evaluation_id)
    relative = destination.relative_to(Path(run_root)).as_posix()
    write_manifest(destination.parent, REPORT_NAME, report.model_dump(mode="json"))
    return relative


def evaluate(suite: SuiteSpec, identity: dict, values: dict, *, expected_evaluator_hash: str, actual_evaluator_hash: str, observed_cadence: dict = None, details: dict = None, thresholds: Mapping[str, float] = None) -> EvaluationReport:
    """Run every gate for a suite and return the report, pass or fail.

    Each metric is checked for presence, finiteness, cadence, and threshold. The
    FIRST failure wins and becomes the reason; later metrics are not checked, so
    the report always states one unambiguous cause. A non-PASS result is still a
    complete report, never an exception in place of one.
    """
    for metric in suite.metrics:
        if metric.name not in values:
            return build_report(
                suite, identity, values,
                expected_evaluator_hash=expected_evaluator_hash,
                actual_evaluator_hash=actual_evaluator_hash,
                status="FAIL",
                reason="missing required metric: " + metric.name,
                details=details,
            )
    try:
        require_complete_suite(
            set(suite.names()), set(values),
            expected_evaluator_hash=expected_evaluator_hash,
            actual_evaluator_hash=actual_evaluator_hash,
        )
        for metric in suite.metrics:
            require_finite(metric.name, values[metric.name])
        cadence = suite.all_cadence()
        if observed_cadence is not None and cadence:
            for name, expected_steps in cadence.items():
                require_cadence(name, set(observed_cadence.get(name, ())), set(expected_steps))
        for metric in suite.metrics:
            # A metric may declare NO threshold at all, which is how the
            # catalogue marks a measurement as reported-but-not-gated. The None
            # check has to come first: reading .value off a missing threshold
            # is an AttributeError, not a skip.
            if metric.threshold is None:
                continue
            # The catalogue owns WHICH metrics exist and which DIRECTION each
            # one improves in; the run config owns the VALUE. Binding only the
            # catalogue value would make a declared threshold decorative, so an
            # override is applied here and the direction is never negotiable.
            bound = float(thresholds[metric.name]) if (
                thresholds is not None and metric.name in thresholds
            ) else float(metric.threshold.value)
            require_threshold(
                metric.name, values[metric.name], bound, metric.threshold.direction,
            )
    except GateFailure as failure:
        return build_report(
            suite, identity, values,
            expected_evaluator_hash=expected_evaluator_hash,
            actual_evaluator_hash=actual_evaluator_hash,
            status="FAIL",
            reason=str(failure),
            details=details,
        )
    return build_report(
        suite, identity, values,
        expected_evaluator_hash=expected_evaluator_hash,
        actual_evaluator_hash=actual_evaluator_hash,
        status="PASS",
        details=details,
    )
