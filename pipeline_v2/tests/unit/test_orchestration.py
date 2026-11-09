"""Task 8 unit tests: the stage graph, the CPU-only doctor, and the handoff."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.orchestration.bootstrap import BootstrapIdentity, bootstrap_attempt_id
from slm_train.orchestration.engine import Engine, StageOutcome
from slm_train.orchestration.errors import (
    MissingStageInput,
    StageBlocked,
    StageFailed,
)
from slm_train.orchestration.graph import StageGraph
from slm_train.orchestration.identity_handoff import IdentityHandoff, HandoffConflict


def _bootstrap(**overrides) -> BootstrapIdentity:
    values = {
        "logical_experiment": "exp",
        "config_hash": "c" * 64,
        "source_identity_hash": "a" * 64,
        "tokenizer_identity_hash": "b" * 64,
        "launch_profile_hash": "d" * 64,
        "requested_runtime_hash": "e" * 64,
        "code_dependency_hash": "f" * 64,
        "command_hash": "0" * 64,
    }
    return BootstrapIdentity(**{**values, **overrides})


# --- graph contract ---------------------------------------------------------


def test_graph_order_matches_v2_contract():
    graph = StageGraph.r1()
    order = graph.order
    assert order[0] == "validate"
    assert order.index("prepare") < order.index("tokenize") < order.index("train")
    assert order[-2:] == ("external_gates", "publish_final")
    produced = {path for spec in graph.specs() for path in spec.outputs}
    assert {
        "manifests/data.json",
        "manifests/tokenizer.json",
        "manifests/tokenized_data.json",
        "manifests/evaluator.json",
        "manifests/run_identity.json",
    } <= produced


def test_graph_order_is_exactly_the_r1_sequence():
    assert StageGraph.r1().order == (
        "validate",
        "bootstrap",
        "gpu_preflight",
        "prepare",
        "tokenize",
        "dataset_validation",
        "model_validation",
        "evaluator_validation",
        "resolve_final_identity",
        "train",
        "internal_evaluation",
        "external_gates",
        "publish_final",
    )


def test_every_stage_requires_only_artifacts_an_earlier_stage_produces():
    graph = StageGraph.r1()
    produced_so_far: set[str] = set()
    for spec in graph.specs():
        missing = set(spec.required_inputs) - produced_so_far
        assert not missing, f"{spec.name} requires unproduced inputs: {sorted(missing)}"
        produced_so_far |= set(spec.outputs)


def test_every_stage_has_a_completion_marker_named_after_itself():
    for spec in StageGraph.r1().specs():
        assert spec.completion_marker == f"{spec.name}.COMPLETE.json"


def test_stage_graph_rejects_a_cycle_and_a_reordering():
    with pytest.raises(ValueError):
        StageGraph(order=("tokenize", "prepare", "validate"))
    with pytest.raises(ValueError):
        StageGraph(order=("prepare", "prepare", "tokenize"))
    with pytest.raises(ValueError):
        StageGraph(order=())


def test_stage_graph_is_immutable():
    graph = StageGraph.r1()
    with pytest.raises(Exception):
        graph.order = ("validate",)


# --- bootstrap identity -----------------------------------------------------


def test_bootstrap_digest_is_deterministic_and_pure():
    first = _bootstrap()
    second = _bootstrap()
    assert first.digest() == second.digest()
    assert first.digest() != _bootstrap(config_hash="9" * 64).digest()


def test_bootstrap_digest_does_not_include_the_attempt():
    """The attempt id must be excluded from the bootstrap identity digest."""
    identity = _bootstrap()
    assert "attempt" not in identity.digest().lower()
    assert set(identity.as_dict()) == {
        "logical_experiment",
        "config_hash",
        "source_identity_hash",
        "tokenizer_identity_hash",
        "launch_profile_hash",
        "requested_runtime_hash",
        "code_dependency_hash",
        "command_hash",
    }


def test_bootstrap_rejects_an_empty_or_short_field():
    for bad in ({"logical_experiment": ""}, {"config_hash": "short"}, {"command_hash": "0" * 63}, {"config_hash": "z" * 64}):
        with pytest.raises(ValueError):
            _bootstrap(**bad)


def test_attempt_id_is_pure_and_excludes_nothing_but_its_three_inputs():
    a = bootstrap_attempt_id("boot", "nonce-1", 123, 1.5)
    b = bootstrap_attempt_id("boot", "nonce-1", 123, 1.5)
    assert a == b
    assert a != bootstrap_attempt_id("boot", "nonce-2", 123, 1.5)
    assert a != bootstrap_attempt_id("boot", "nonce-1", 124, 1.5)
    assert a != bootstrap_attempt_id("boot", "nonce-1", 123, 1.6)
    assert a != bootstrap_attempt_id("other", "nonce-1", 123, 1.5)


# --- doctor -----------------------------------------------------------------


def test_doctor_never_calls_gpu(monkeypatch, tmp_path):
    import slm_train.resources.doctor as doctor

    def explode():
        raise AssertionError("GPU touched")

    monkeypatch.setattr(doctor, "probe_gpu", explode, raising=False)
    result = doctor.run_doctor(root=tmp_path)
    assert result["gpu_touched"] is False


def test_doctor_never_imports_torch_cuda(monkeypatch, tmp_path):
    import sys

    import slm_train.resources.doctor as doctor

    monkeypatch.delitem(sys.modules, "torch.cuda", raising=False)
    doctor.run_doctor(root=tmp_path)
    assert "torch.cuda" not in sys.modules


def test_doctor_reports_every_check_with_a_label(tmp_path):
    import slm_train.resources.doctor as doctor

    result = doctor.run_doctor(root=tmp_path)
    assert result["status"] in {"PASS", "FAIL", "BLOCKED", "NEEDS_USER_DECISION", "SKIPPED"}
    for check in ("venv", "package_metadata", "imports", "writable_roots", "disk", "config_paths"):
        assert check in result["checks"], check
        assert result["checks"][check]["label"] in {
            "PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION",
        }


def test_doctor_reports_a_missing_root_as_blocked_not_pass(tmp_path):
    import slm_train.resources.doctor as doctor

    result = doctor.run_doctor(root=tmp_path / "does-not-exist")
    assert result["status"] != "PASS"

