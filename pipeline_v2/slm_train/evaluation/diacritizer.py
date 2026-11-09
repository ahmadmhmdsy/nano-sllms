"""Bridge from the D-line metric artifact to a pipeline_v2 EvaluationReport.

diacritizer/scripts/eval_report.py writes a JSON artifact of measured numbers.
This module turns one model's row from that artifact into the same
EvaluationReport every other pipeline_v2 evaluator produces, so a diacritizer
result is a first-class governed artifact rather than a loose JSON file.

Two deliberate choices:

Scoping. pipeline_v2 requires the value set to equal the suite's metric set
exactly - a union of everything measured fails as "unexpected metrics". The
driver measures far more than the suite declares, so this bridge scopes the
row down to the declared names and lets the gate judge the rest.

Failure is a report, not an exception. A structural problem (not an artifact,
no such model) raises, because that is a caller mistake. A metric that is
missing, non-finite, or out of bounds produces a FAIL report carrying the
reason, because that is a measurement outcome and a run that fails without a
report has failed without evidence.
"""
from __future__ import annotations

import hashlib
import json

from slm_train.evaluation.diacritizer_suites import diacritizer_suite_for
from slm_train.evaluation.reports import evaluate

__all__ = ["DiacritizerArtifactError", "evaluator_manifest_hash", "report_from_artifact"]


class DiacritizerArtifactError(ValueError):
    """The artifact is not the shape this bridge reads."""


def evaluator_manifest_hash(suite) -> str:
    """A stable digest of the suite's metric schema.

    This is the evaluator identity the report carries on both sides. It is
    derived from the declared definitions rather than from the run, so a report
    scored against a changed catalogue cannot claim the old one.
    """
    payload = json.dumps(
        {"evaluator": suite.evaluator, "version": suite.version, "definitions": suite.definitions()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_model(artifact: dict, model):
    if not isinstance(artifact, dict):
        raise DiacritizerArtifactError("the artifact must be a mapping, got " + repr(type(artifact)))
    rows = artifact.get("models")
    if not isinstance(rows, list) or not rows:
        raise DiacritizerArtifactError("the artifact carries no models list")
    if model is None:
        model = artifact.get("baseline") or rows[0].get("model")
    for row in rows:
        if isinstance(row, dict) and row.get("model") == model:
            return row
    known = ", ".join(sorted(str(r.get("model")) for r in rows if isinstance(r, dict)))
    raise DiacritizerArtifactError("no such model in the artifact: " + repr(model) + "; have: " + known)


def report_from_artifact(
    artifact: dict,
    *,
    run_id: str,
    tokenizer_hash: str,
    device_identity: str,
    model=None,
    seed=None,
    suite_name: str = "diacritizer_metrics",
    thresholds=None,
):
    """Score one model row of a driver artifact against the declared suite.

    run_id, tokenizer_hash and device_identity are required and are NOT invented
    here: the driver artifact records the reference hash but knows nothing about
    which tokenizer or device produced the predictions, and a report that
    guessed them could be attributed to the wrong run.
    """
    suite = diacritizer_suite_for(suite_name)
    row = _select_model(artifact, model)
    measured = row.get("metrics")
    if not isinstance(measured, dict):
        raise DiacritizerArtifactError("model " + repr(row.get("model")) + " carries no metrics mapping")

    declared = suite.names()
    scoped = {name: measured[name] for name in declared if name in measured}

    ref = artifact.get("ref") if isinstance(artifact.get("ref"), dict) else {}
    identity = {
        "run_id": run_id,
        "data_hash": str(ref.get("sha256", "")),
        "tokenizer_hash": tokenizer_hash,
        "device_identity": device_identity,
        "seed": seed,
        "checkpoint_identity": str(row.get("pred", "")),
    }
    details = {
        "model": row.get("model"),
        "pred": row.get("pred"),
        "reference_path": ref.get("path"),
        "reference_lines": ref.get("lines"),
        "conditions": artifact.get("conditions"),
        "rank_by": artifact.get("rank_by"),
        "missing_declared_metrics": sorted(declared - set(scoped)),
    }
    digest = evaluator_manifest_hash(suite)
    return evaluate(
        suite,
        identity,
        scoped,
        expected_evaluator_hash=digest,
        actual_evaluator_hash=digest,
        details=details,
        thresholds=thresholds,
    )
