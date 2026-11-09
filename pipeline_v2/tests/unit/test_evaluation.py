"""Gates, evaluators, reports, and final publication. Failure is the subject."""
from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "pipeline_v2")

from slm_train.artifacts.publication import (
    FINAL_MANIFEST,
    PublicationError,
    publish_final,
    read_back_final_publication,
    write_run_complete,
)
from slm_train.evaluation import gates, suites
from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold
from slm_train.evaluation.generation import generation_metrics
from slm_train.evaluation.internal import EvaluationError, validation_metrics
from slm_train.evaluation.reports import build_report, evaluate, report_path, write_report
from slm_train.evaluation.resources import resource_metrics
from slm_train.evaluation.syntax import python_parser, syntax_metrics

IDENTITY = {
    "run_id": "run-1",
    "data_hash": "d" * 64,
    "tokenizer_hash": "t" * 64,
    "device_identity": "cpu",
}


# --- gates ----------------------------------------------------------------


def test_missing_metric_cannot_pass():
    with pytest.raises(gates.GateFailure, match="missing"):
        gates.require_complete_suite(
            {"val_loss", "generation_smoke"}, {"val_loss"},
            expected_evaluator_hash="e1", actual_evaluator_hash="e1",
        )


def test_non_finite_metric_blocks():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(gates.GateFailure, match="non-finite"):
            gates.require_finite("val_loss", bad)


def test_threshold_direction_is_enforced():
    with pytest.raises(gates.GateFailure, match="exceeds"):
        gates.require_threshold("val_loss", 2.0, 1.0, "min")
    with pytest.raises(gates.GateFailure, match="below"):
        gates.require_threshold("generation_smoke", 0.0, 1.0, "max")


def test_evaluator_manifest_mismatch_blocks():
    with pytest.raises(gates.GateFailure, match="manifest"):
        gates.require_complete_suite(
            {"val_loss"}, {"val_loss"},
            expected_evaluator_hash="e1", actual_evaluator_hash="e2",
        )


def test_an_unsupported_direction_is_refused_before_any_comparison():
    """A typo must not silently compare in one direction only."""
    with pytest.raises(gates.GateFailure, match="direction"):
        gates.require_threshold("val_loss", 0.0, 1.0, "minimum")


@pytest.mark.parametrize("value", ["1.0", None, True, [1.0], object()])
def test_a_non_number_metric_is_refused(value):
    with pytest.raises(gates.GateFailure, match="non-finite"):
        gates.require_finite("val_loss", value)


def test_a_non_finite_threshold_is_refused():
    with pytest.raises(gates.GateFailure, match="non-finite"):
        gates.require_threshold("val_loss", 1.0, float("nan"), "min")


def test_suite_membership_must_be_compared_as_sets():
    with pytest.raises(gates.GateFailure, match="sets"):
        gates.require_complete_suite(
            ["val_loss"], {"val_loss"},
            expected_evaluator_hash="e", actual_evaluator_hash="e",
        )


@pytest.mark.parametrize("absent", [None, "", "   "])
def test_two_absent_evaluator_hashes_are_not_identity(absent):
    """None == None is not evidence that the evaluator is the same."""
    with pytest.raises(gates.GateFailure, match="absent"):
        gates.require_complete_suite(
            {"val_loss"}, {"val_loss"},
            expected_evaluator_hash=absent, actual_evaluator_hash=absent,
        )


def test_an_absent_evaluator_hash_fails_a_report_even_when_both_match():
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": 1.0, "val_perplexity": 2.7},
        expected_evaluator_hash=None, actual_evaluator_hash=None,
    )
    assert report.status == "FAIL"
    assert "absent" in report.details["reason"]


def test_an_unexpected_metric_is_also_a_failure():
    with pytest.raises(gates.GateFailure, match="unexpected"):
        gates.require_complete_suite(
            {"val_loss"}, {"val_loss", "smuggled"},
            expected_evaluator_hash="e", actual_evaluator_hash="e",
        )


def test_a_cadence_gap_is_a_failure_not_a_pass():
    with pytest.raises(gates.GateFailure, match="cadence"):
        gates.require_cadence("val_loss", {1}, {1, 2})


def test_a_satisfied_cadence_returns_none():
    assert gates.require_cadence("val_loss", {1, 2}, {1, 2}) is None


def test_no_gate_returns_a_boolean():
    """A gate that returned True could be ignored; one that raises cannot."""
    for name in ("require_complete_suite", "require_cadence"):
        assert not hasattr(gates, name + "_ok")


# --- internal validation evaluator ----------------------------------------


def test_validation_metrics_derives_perplexity_from_a_real_loss():
    values = validation_metrics(1.0, 128)
    assert values["val_loss"] == 1.0
    assert 2.718 < values["val_perplexity"] < 2.719


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_loss_that_cannot_yield_a_finite_perplexity_is_refused(loss):
    with pytest.raises(EvaluationError):
        validation_metrics(loss, 128)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "8"])
def test_a_sample_count_that_was_not_measured_is_refused(count):
    with pytest.raises(EvaluationError):
        validation_metrics(1.0, count)


def test_an_astronomically_large_loss_does_not_silently_return_inf():
    with pytest.raises(EvaluationError, match="perplexity"):
        validation_metrics(100000.0, 1)


# --- generation, syntax, resources ----------------------------------------


def test_generation_smoke_passes_only_on_non_empty_text():
    assert generation_metrics(["Paris"])["generation_smoke"] == 1.0
    assert generation_metrics(["Paris", "  "])["generation_smoke"] == 0.0
    assert generation_metrics([""])["generation_smoke"] == 0.0


@pytest.mark.parametrize("samples", [[], "text", [None], [1]])
def test_generation_refuses_what_it_cannot_score(samples):
    with pytest.raises(ValueError):
        generation_metrics(samples)


def test_a_syntax_error_rate_counts_real_parser_failures():
    metrics = syntax_metrics(["x = 1", "def (:", "y = 2"], python_parser())
    assert metrics["syntax_error_rate"] == 1 / 3
    assert len(metrics["failures"]) == 1


def test_valid_python_scores_a_zero_error_rate():
    assert syntax_metrics(["x = 1"], python_parser())["syntax_error_rate"] == 0.0


def test_syntax_refuses_an_unparseable_sample_list():
    with pytest.raises(ValueError):
        syntax_metrics([], python_parser())
    with pytest.raises(ValueError):
        syntax_metrics(["x = 1"], "not callable")


def test_resource_metrics_compute_from_measurements_only():
    metrics = resource_metrics(tokens=1000, seconds=2.0, peak_vram_bytes=4096)
    assert metrics["tokens_per_second"] == 500.0
    assert metrics["peak_vram_bytes"] == 4096.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": 0, "seconds": 1.0, "peak_vram_bytes": 1},
        {"tokens": 1, "seconds": 0, "peak_vram_bytes": 1},
        {"tokens": 1, "seconds": 1.0, "peak_vram_bytes": None},
        {"tokens": 1, "seconds": 1.0, "peak_vram_bytes": -5},
        {"tokens": 1, "seconds": float("inf"), "peak_vram_bytes": 1},
    ],
)
def test_an_unmeasured_resource_is_refused_not_reported_as_zero(kwargs):
    """A fabricated zero would read as a real measurement."""
    with pytest.raises(ValueError):
        resource_metrics(**kwargs)


@pytest.mark.parametrize("bad", [True, "0.5", None, [1.0], float("nan")])
def test_an_unusable_metric_still_produces_a_report_instead_of_an_exception(bad):
    """A gate must record why it failed, not crash before the report exists."""
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": bad, "val_perplexity": 2.7},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    assert report.status == "FAIL"
    assert report.details["reason"]
    assert any(item.startswith("val_loss=") for item in report.details["unreportable_values"])


def test_an_unusable_metric_is_named_in_the_report_rather_than_dropped_silently():
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": True, "val_perplexity": 2.7},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    assert report.metric_values == {"val_perplexity": 2.7}
    assert report.details["unreportable_values"] == ["val_loss=True"]


def test_a_report_with_no_unusable_values_records_no_dropped_entries():
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": 1.0, "val_perplexity": 2.7},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    assert "unreportable_values" not in report.details


# --- suite declaration ----------------------------------------------------


def test_no_evaluator_is_discovered_at_runtime():
    assert sorted(suites.SUITES) == ["generation", "resources", "syntax", "validation"]


def test_an_unknown_suite_names_what_is_declared():
    with pytest.raises(KeyError, match="declared"):
        suites.suite_for("laya_bench")


def test_every_registered_external_gate_is_reachable_and_implemented():
    """The opposite of the old invariant, and deliberately so.

    EXTERNAL_GATE_NAMES used to be empty, which made the external_gates stage a
    trivial pass: it verified nothing and reported PASS. The stage also used to
    copy the INTERNAL suite reports and relabelled them as external, grading the
    run with the run's own report of itself. Gates are now registered by name
    and each one is implemented against the published bytes.
    """
    from slm_train.evaluation.external import EXTERNAL_GATES

    assert suites.EXTERNAL_GATE_NAMES, "an empty gate set still passes everything"
    for name in suites.EXTERNAL_GATE_NAMES:
        assert name in EXTERNAL_GATES, name + " is registered but not implemented"
    # A gate that is implemented but not registered is dead code that a run
    # can never reach, which is the failure mode the old empty set had.
    for name in EXTERNAL_GATES:
        assert name in suites.EXTERNAL_GATE_NAMES, (
            name + " is implemented but unreachable by any run"
        )


def test_a_suite_cannot_repeat_a_metric_or_declare_none():
    metric = MetricSpec(name="m", definition="d")
    with pytest.raises(ValueError, match="at least one metric"):
        SuiteSpec(evaluator="e", version="1", metrics=())
    with pytest.raises(ValueError, match="same metric twice"):
        SuiteSpec(evaluator="e", version="1", metrics=(metric, metric))


def test_a_metric_needs_a_definition_and_a_sane_cadence():
    with pytest.raises(ValueError, match="non-empty name"):
        MetricSpec(name="  ", definition="d")
    with pytest.raises(ValueError, match="needs a definition"):
        MetricSpec(name="m", definition="")
    with pytest.raises(ValueError, match="non-negative integers"):
        MetricSpec(name="m", definition="d", cadence=(-1,))
    with pytest.raises(ValueError, match="repeats a cadence"):
        MetricSpec(name="m", definition="d", cadence=(1, 1))


def test_a_threshold_direction_is_checked_when_the_threshold_is_built():
    with pytest.raises(ValueError, match="min or max"):
        Threshold(value=1.0, direction="minimum")


# --- reports --------------------------------------------------------------


def test_a_passing_report_records_the_whole_identity():
    suite = suites.suite_for("validation")
    values = {"val_loss": 1.0, "val_perplexity": 2.718}
    report = evaluate(
        suite, IDENTITY, values,
        expected_evaluator_hash="e1", actual_evaluator_hash="e1",
    )
    assert report.status == "PASS"
    assert report.run_id == "run-1"
    assert report.checkpoint_identity == ""
    assert report.metric_schema_identity == "validation/1"
    assert report.metric_definitions == suite.definitions()
    assert set(report.expected_suite_gates) == {"val_loss", "val_perplexity"}


def test_a_failing_gate_still_writes_a_complete_report():
    """A failure without a report is a failure without evidence."""
    suite = suites.suite_for("validation")
    values = {"val_loss": 9.0, "val_perplexity": 8103.0}
    report = evaluate(
        suite, IDENTITY, values,
        expected_evaluator_hash="e1", actual_evaluator_hash="e1",
    )
    assert report.status == "FAIL"
    assert "exceeds threshold" in report.details["reason"]
    assert report.metric_values == values


def test_a_missing_metric_produces_a_failure_report_naming_it():
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": 1.0},
        expected_evaluator_hash="e1", actual_evaluator_hash="e1",
    )
    assert report.status == "FAIL"
    assert "val_perplexity" in report.details["reason"]
    assert report.metric_values == {"val_loss": 1.0}


def test_a_non_finite_metric_can_never_reach_a_pass():
    suite = suites.suite_for("validation")
    values = {"val_loss": float("nan"), "val_perplexity": 2.718}
    report = evaluate(
        suite, IDENTITY, values,
        expected_evaluator_hash="e1", actual_evaluator_hash="e1",
    )
    assert report.status == "FAIL"


def test_a_cadence_gap_fails_even_when_every_value_is_good():
    suite = suites.suite_for("validation")
    values = {"val_loss": 1.0, "val_perplexity": 2.718}
    report = evaluate(
        suite, IDENTITY, values,
        expected_evaluator_hash="e1", actual_evaluator_hash="e1",
        observed_cadence={"val_loss": {1}, "val_perplexity": {1, 2}},
    )
    assert report.status == "FAIL"


def test_an_evaluator_identity_mismatch_fails():
    suite = suites.suite_for("validation")
    report = evaluate(
        suite, IDENTITY, {"val_loss": 1.0, "val_perplexity": 2.7},
        expected_evaluator_hash="e1", actual_evaluator_hash="e2",
    )
    assert report.status == "FAIL"


@pytest.mark.parametrize("missing", ["run_id", "data_hash", "tokenizer_hash", "device_identity"])
def test_a_report_without_full_identity_is_refused(missing):
    identity = dict(IDENTITY)
    identity.pop(missing)
    with pytest.raises(ValueError, match=missing):
        build_report(
            suites.suite_for("validation"), identity, {"val_loss": 1.0},
            expected_evaluator_hash="e", actual_evaluator_hash="e", status="PASS",
        )


def test_a_non_pass_report_must_state_a_reason():
    with pytest.raises(ValueError, match="reason"):
        build_report(
            suites.suite_for("validation"), IDENTITY, {"val_loss": 1.0},
            expected_evaluator_hash="e", actual_evaluator_hash="e", status="FAIL",
        )


def test_an_invented_status_is_refused():
    with pytest.raises(ValueError, match="validation label"):
        build_report(
            suites.suite_for("validation"), IDENTITY, {"val_loss": 1.0},
            expected_evaluator_hash="e", actual_evaluator_hash="e", status="GREAT",
        )


@pytest.mark.parametrize("skipped", ["SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"])
def test_every_validation_label_is_accepted_by_a_report(skipped):
    report = build_report(
        suites.suite_for("validation"), IDENTITY, {"val_loss": 1.0},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
        status=skipped, reason="evidence",
    )
    assert report.status == skipped


@pytest.mark.parametrize("bad", ["", "  ", "../escape", "a/b", ".", ".."])
def test_an_evaluation_id_must_be_one_path_segment(bad):
    with pytest.raises(ValueError):
        report_path("/run", bad)


def test_a_second_write_to_one_evaluation_id_is_refused_and_keeps_the_first(tmp_path):
    suite = suites.suite_for("generation")
    report = evaluate(
        suite, IDENTITY, {"generation_smoke": 1.0},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    write_report(tmp_path, "eval-1", report)
    with pytest.raises(FileExistsError):
        write_report(tmp_path, "eval-1", report)
    kept = json.loads((tmp_path / "evaluations/eval-1/report.json").read_text("utf-8"))
    assert kept["status"] == "PASS"


def test_a_report_is_published_create_if_absent(tmp_path):
    suite = suites.suite_for("generation")
    report = evaluate(
        suite, IDENTITY, {"generation_smoke": 1.0},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    relative = write_report(tmp_path, "eval-1", report)
    assert relative == "evaluations/eval-1/report.json"
    document = json.loads((tmp_path / relative).read_text("utf-8"))
    assert document["status"] == "PASS"
    with pytest.raises(Exception):
        write_report(tmp_path, "eval-1", report)


def test_a_second_evaluation_uses_a_new_id_and_does_not_overwrite(tmp_path):
    suite = suites.suite_for("validation")
    first = evaluate(
        suite, IDENTITY, {"val_loss": 1.0, "val_perplexity": 2.7},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    second = evaluate(
        suite, IDENTITY, {"val_loss": 5.0, "val_perplexity": 148.0},
        expected_evaluator_hash="e", actual_evaluator_hash="e",
    )
    write_report(tmp_path, "eval-1", first)
    write_report(tmp_path, "eval-2", second)
    kept = json.loads((tmp_path / "evaluations/eval-1/report.json").read_text("utf-8"))
    assert kept["status"] == "PASS"
    later = json.loads((tmp_path / "evaluations/eval-2/report.json").read_text("utf-8"))
    assert later["status"] == "FAIL"




# --- final publication ----------------------------------------------------


def _stage_bound_artifacts(tmp_path):
    """Write the two artifacts every run must bind, and their real hashes."""
    from slm_train.contracts.hash import sha256_bytes

    bound = {
        "manifests/run_identity.json": b'{"run_id": "run-1"}',
        "logs/log-manifest.json": b'{"kinds": {}}',
    }
    for name, payload in bound.items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    hashes = {name: sha256_bytes(payload) for name, payload in bound.items()}
    return bound, hashes


def _publish(tmp_path, **overrides):
    bound, hashes = _stage_bound_artifacts(tmp_path)
    arguments = {
        "run_id": "run-1",
        "selected_checkpoint": "step-00000010",
        "model_payloads": {"model.safetensors": b"weights"},
        "tokenizer_payloads": {"tokenizer.json": b"tokens"},
        "required_hashes": hashes,
    }
    arguments.update(overrides)
    publish_final(tmp_path, **arguments)
    return bound


def test_publication_writes_the_manifest_after_reading_the_staged_files_back(tmp_path):
    _publish(tmp_path)
    assert (tmp_path / FINAL_MANIFEST).is_file()
    manifest = read_back_final_publication(tmp_path)
    assert manifest.read_back_passed is True
    assert manifest.run_id == "run-1"
    assert {record.path for record in manifest.final_files} == {
        "final/model/model.safetensors",
        "final/tokenizer/tokenizer.json",
    }


def test_publication_binds_the_staged_files_it_did_not_receive_hashes_for(tmp_path):
    """A caller cannot stage a model file and forget to bind it."""
    _publish(tmp_path)
    manifest = read_back_final_publication(tmp_path)
    assert "final/model/model.safetensors" in manifest.payload_hashes
    assert "final/tokenizer/tokenizer.json" in manifest.payload_hashes
    assert "manifests/run_identity.json" in manifest.payload_hashes
    assert "logs/log-manifest.json" in manifest.payload_hashes


def test_a_staged_file_that_changes_after_staging_fails_read_back(tmp_path, monkeypatch):
    import slm_train.artifacts.publication as publication

    real = publication._read_back

    def sabotage(root, record):
        (root / record.path).write_bytes(b"tampered")
        return real(root, record)

    monkeypatch.setattr(publication, "_read_back", sabotage)
    with pytest.raises(PublicationError, match="read-back"):
        _publish(tmp_path)


def test_publication_refuses_to_stage_nothing(tmp_path):
    with pytest.raises(PublicationError, match="no artifacts"):
        _publish(tmp_path, model_payloads={}, tokenizer_payloads={})


def test_publication_refuses_a_staged_name_that_escapes_the_staging_directory(tmp_path):
    with pytest.raises(PublicationError, match="single name"):
        _publish(tmp_path, model_payloads={"../escape.pt": b"x"})


def test_publication_refuses_an_empty_run_id_or_checkpoint(tmp_path):
    with pytest.raises(PublicationError, match="run id"):
        _publish(tmp_path, run_id="  ")
    with pytest.raises(PublicationError, match="selected checkpoint"):
        _publish(tmp_path, selected_checkpoint="")


def test_no_complete_marker_exists_before_the_marker_step(tmp_path):
    _publish(tmp_path)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_the_complete_marker_is_written_last_and_binds_the_publication(tmp_path):
    bound = _publish(tmp_path)
    write_run_complete(tmp_path, bound)
    marker = json.loads((tmp_path / "COMPLETE.json").read_text("utf-8"))
    assert marker["manifest"] == FINAL_MANIFEST
    assert "manifests/run_identity.json" in marker["payload_hashes"]
    assert "logs/log-manifest.json" in marker["payload_hashes"]


def test_a_payload_the_publication_never_certified_cannot_be_bound(tmp_path):
    _publish(tmp_path)
    stranger = tmp_path / "not/certified.json"
    stranger.parent.mkdir(parents=True, exist_ok=True)
    stranger.write_bytes(b"x")
    with pytest.raises(ValueError, match="does not cover"):
        write_run_complete(tmp_path, {"not/certified.json": b"x"})


def test_a_bound_payload_missing_from_disk_is_refused(tmp_path):
    _publish(tmp_path)
    with pytest.raises(PublicationError, match="missing on disk"):
        write_run_complete(tmp_path, {"manifests/never_written.json": b"identity"})


def test_the_marker_refuses_bytes_that_disagree_with_the_published_file(tmp_path):
    bound = _publish(tmp_path)
    with pytest.raises(ValueError, match="hash read-back failed"):
        write_run_complete(tmp_path, dict(bound, **{"logs/log-manifest.json": b"different"}))


def test_the_marker_refuses_when_a_bound_artifact_changed_after_publication(tmp_path):
    bound = _publish(tmp_path)
    (tmp_path / "logs/log-manifest.json").write_bytes(b"edited-after-publication")
    with pytest.raises(ValueError, match="hash read-back failed"):
        write_run_complete(tmp_path, bound)


def test_write_run_complete_refuses_an_empty_binding(tmp_path):
    _publish(tmp_path)
    with pytest.raises(PublicationError, match="needs the artifacts"):
        write_run_complete(tmp_path, {})


def test_write_run_complete_refuses_when_the_publication_was_never_made(tmp_path):
    with pytest.raises(Exception):
        write_run_complete(tmp_path, {"anything.json": b"x"})
