"""The diacritizer evaluator suite and the artifact-to-report bridge.

The subject here is the boundary between the D-line's own metric harness and
pipeline_v2's governed report contract: that the catalogue stays parallel to the
R1 suites, that the bridge scopes a wide artifact row down to the declared
metric set, and that a bad measurement becomes a FAIL report rather than an
exception.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "pipeline_v2")

from slm_train.adapters import registry
from slm_train.evaluation import suites
from slm_train.evaluation.diacritizer import (
    DiacritizerArtifactError,
    evaluator_manifest_hash,
    report_from_artifact,
)
from slm_train.evaluation.diacritizer_suites import DIACRITIZER_SUITES, diacritizer_suite_for

#: A row shaped like one model inside diacritizer/scripts/eval_report.py output,
#: carrying every metric the suite declares plus extras the suite does not, to
#: prove the bridge scopes rather than forwards.
ROW = {
    "der_canonical": 0.4520,
    "der_anyalt": 0.5245,
    "der_primary": 0.5094,
    "ser": 0.0,
    "cder": 0.2885,
    "macro_f1": 0.7378,
    "position_accuracy": 0.8682,
    "worst_class_recall": 0.3333,
    "case_accuracy": 0.8610,
    "density_ratio": 0.939,
    "alignment_coverage": 0.9667,
    "unclassified_units": 0.0,
    "bare_emission_rate": 0.2646,
    # not declared by the suite; must be dropped, not forwarded
    "cohen_kappa": 0.835,
    "matthews_cc": 0.82,
}


def artifact(metrics=None, model="mishkala"):
    row = dict(ROW if metrics is None else metrics)
    return {
        "artifact": "Diacritizer full-metric evaluation (WikiNews 2024)",
        "baseline": model,
        "rank_by": "der_canonical",
        "conditions": ["registered", "anyalt", "primary", "canonical", "nocase"],
        "ref": {
            "path": "models/e19/wn2024.gold.raw.pred.ref.txt",
            "sha256": "a" * 64,
            "lines": 356,
            "words": 10616,
        },
        "models": [{"model": model, "pred": "models/e19/wn2024.mishkala.pred.txt", "metrics": row}],
    }


def report(**kwargs):
    options = {
        "run_id": "diac-run-1",
        "tokenizer_hash": "t" * 64,
        "device_identity": "cpu",
    }
    options.update(kwargs.pop("options", {}))
    return report_from_artifact(kwargs.pop("artifact", artifact()), **options, **kwargs)


# --- the catalogue -----------------------------------------------------------


def test_the_catalogue_is_declared_and_well_formed():
    assert DIACRITIZER_SUITES
    for name, suite in DIACRITIZER_SUITES.items():
        assert suite.evaluator == name
        assert suite.metrics
        assert suite.split_identity


def test_the_catalogue_does_not_extend_the_r1_suite_set():
    """SUITES is pinned elsewhere; the diacritizer must stay out of it."""
    assert sorted(suites.SUITES) == ["generation", "resources", "syntax", "validation"]
    assert not set(DIACRITIZER_SUITES) & set(suites.SUITES)
    # And the R1 resolver must keep refusing it, so no R1 config can reach it
    # through the pretrain path.
    with pytest.raises(KeyError):
        suites.suite_for("diacritizer_metrics")


def test_an_unknown_diacritizer_suite_names_what_is_declared():
    with pytest.raises(KeyError, match="declared"):
        diacritizer_suite_for("wn2024_bench")


def test_the_manifest_hash_is_stable_and_covers_the_definitions():
    suite = diacritizer_suite_for("diacritizer_metrics")
    assert evaluator_manifest_hash(suite) == evaluator_manifest_hash(suite)
    assert len(evaluator_manifest_hash(suite)) == 64


# --- the bridge --------------------------------------------------------------


def test_a_complete_row_passes():
    result = report()
    assert result.status == "PASS"
    assert result.evaluator == "diacritizer_metrics"
    assert set(result.metric_values) == diacritizer_suite_for("diacritizer_metrics").names()


def test_undeclared_metrics_are_scoped_out_rather_than_failing():
    """A union of everything measured must not trip the completeness check."""
    result = report()
    assert "cohen_kappa" not in result.metric_values
    assert "matthews_cc" not in result.metric_values
    assert result.status == "PASS"


def test_a_missing_declared_metric_fails_with_a_reason():
    row = dict(ROW)
    del row["cder"]
    result = report(artifact=artifact(row))
    assert result.status == "FAIL"
    assert "cder" in result.details["reason"]


def test_a_non_finite_metric_fails():
    row = dict(ROW)
    row["macro_f1"] = float("nan")
    result = report(artifact=artifact(row))
    assert result.status == "FAIL"


def test_a_schema_violating_mark_count_fails():
    """unclassified_units above zero means the head emitted outside its schema."""
    row = dict(ROW)
    row["unclassified_units"] = 3.0
    result = report(artifact=artifact(row))
    assert result.status == "FAIL"
    assert "unclassified_units" in result.details["reason"]


def test_a_silent_model_fails_the_bare_emission_bound():
    row = dict(ROW)
    row["bare_emission_rate"] = 1.0
    result = report(artifact=artifact(row))
    assert result.status == "FAIL"
    assert "bare_emission_rate" in result.details["reason"]


def test_low_alignment_coverage_fails_the_reportability_bound():
    row = dict(ROW)
    row["alignment_coverage"] = 0.50
    result = report(artifact=artifact(row))
    assert result.status == "FAIL"
    assert "alignment_coverage" in result.details["reason"]


def test_a_run_config_may_tighten_a_bound_but_not_flip_its_direction():
    result = report(thresholds={"alignment_coverage": 0.99})
    assert result.status == "FAIL"
    assert "alignment_coverage" in result.details["reason"]


def test_the_report_carries_the_reference_identity():
    result = report()
    assert result.data_hash == "a" * 64
    assert result.split_identity == "wikinews2024_multiref"
    assert result.details["reference_lines"] == 356


def test_an_unknown_model_is_a_caller_error_not_a_failed_report():
    with pytest.raises(DiacritizerArtifactError, match="no such model"):
        report(model="not_a_model")


def test_a_non_artifact_is_refused():
    with pytest.raises(DiacritizerArtifactError):
        report(artifact={"models": []})


def test_the_bridge_does_not_invent_run_identity():
    """An empty tokenizer hash is a caller mistake, not something to guess."""
    with pytest.raises(ValueError, match="identity"):
        report(options={"tokenizer_hash": ""})


# --- the registry ------------------------------------------------------------


def test_the_registry_declares_the_diacritizer_evaluator():
    entry = registry.get("diacritizer_eval")
    assert entry.schema == "EvaluationReport"
    assert entry.maturity == "experimental"
    assert entry.release == "r1"
    assert entry.test_contract == "unit/test_diacritizer_evaluation.py"
