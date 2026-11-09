"""Task 8 unit tests, part 2: the identity handoff and the stage engine."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.orchestration.bootstrap import BootstrapIdentity
from slm_train.orchestration.engine import Engine, StageOutcome
from slm_train.orchestration.errors import (
    MissingStageInput,
    StageBlocked,
    StageFailed,
)
from slm_train.orchestration.graph import StageGraph
from slm_train.orchestration.identity_handoff import HandoffConflict, IdentityHandoff


HASH_H = "a1" * 32
OTHER_HASH = "b2" * 32


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


# --- identity handoff (CAS) -------------------------------------------------


def test_handoff_creates_then_refuses_a_different_identity(tmp_path):
    handoff = IdentityHandoff(tmp_path)
    written = handoff.publish(_bootstrap(), run_id="run-1", identity_hash=HASH_H)
    assert written.exists()
    again = handoff.publish(_bootstrap(), run_id="run-1", identity_hash=HASH_H)
    assert again == written, "an identical identity must be idempotent"
    with pytest.raises(HandoffConflict):
        handoff.publish(_bootstrap(), run_id="run-1", identity_hash=OTHER_HASH)


def test_handoff_never_overwrites_a_foreign_run(tmp_path):
    handoff = IdentityHandoff(tmp_path)
    handoff.publish(_bootstrap(), run_id="run-1", identity_hash=HASH_H)
    with pytest.raises(HandoffConflict):
        handoff.publish(_bootstrap(), run_id="run-1", identity_hash=OTHER_HASH)
    target = tmp_path / "run-1" / "manifests" / "run_identity.json"
    assert json.loads(target.read_text("utf-8"))["identity_hash"] == HASH_H


@pytest.mark.parametrize("bad", ["", None, "..", "../escape", "/abs", "C:win", "a/b"])
def test_handoff_refuses_a_placeholder_or_traversal_run_id(tmp_path, bad) -> None:
    handoff = IdentityHandoff(tmp_path)
    with pytest.raises(ValueError):
        handoff.publish(_bootstrap(), run_id=bad, identity_hash=HASH_H)


def test_handoff_round_trips_the_published_identity(tmp_path):
    handoff = IdentityHandoff(tmp_path)
    identity = _bootstrap()
    handoff.publish(identity, run_id="run-1", identity_hash=HASH_H)
    loaded = handoff.load("run-1")
    assert loaded["identity_hash"] == HASH_H
    assert loaded["bootstrap"]["config_hash"] == identity.config_hash


# --- engine -----------------------------------------------------------------


def _handlers(graph: StageGraph, log: list[str]) -> dict:
    handlers = {}
    for spec in graph.specs():
        def make(name):
            def handler(context):
                log.append(name)
                for path in context.graph.spec_by_name(name).outputs:
                    target = context.run_root / name / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("{}", encoding="utf-8")
                return StageOutcome(label="PASS")
            return handler
        handlers[spec.name] = make(spec.name)
    return handlers


def test_engine_runs_stages_in_order_and_writes_status(tmp_path):
    graph = StageGraph.r1()
    log: list[str] = []
    engine = Engine(graph=graph, handlers=_handlers(graph, log), run_root=tmp_path)
    result = engine.run()
    assert log == list(graph.order)
    assert result.status == "PASS"
    assert result.completed_stages == tuple(graph.order)


def test_engine_stops_at_the_failing_stage_and_never_falls_back(tmp_path):
    graph = StageGraph.r1()
    log: list[str] = []
    handlers = _handlers(graph, log)
    failing = graph.order[3]

    def boom(context):
        log.append(failing)
        raise StageFailed("stage " + failing + " failed")

    handlers[failing] = boom
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageFailed):
        engine.run()
    assert log == list(graph.order[:4]), "no stage after the failure may run"


def test_engine_reports_a_missing_declared_input(tmp_path):
    """A declared input that is not on disk is reported, not assumed present."""
    graph = StageGraph.r1()
    Engine(graph=graph, handlers=_handlers(graph, []), run_root=tmp_path).run()
    engine = Engine(graph=graph, handlers=_handlers(graph, []), run_root=tmp_path)
    consumer = graph.spec_by_name("tokenize")
    assert engine._missing_inputs(consumer) == []
    (tmp_path / "prepare" / "manifests" / "data.json").unlink()
    assert engine._missing_inputs(consumer) == ["manifests/data.json"]


def test_engine_refuses_a_graph_with_an_unknown_handler(tmp_path):
    graph = StageGraph.r1()
    handlers = _handlers(graph, [])
    del handlers["train"]
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(MissingStageInput):
        engine.run()


def test_engine_records_a_blocked_label_without_falling_back(tmp_path):
    graph = StageGraph.r1()
    log: list[str] = []
    handlers = _handlers(graph, log)
    def blocked(context):
        log.append("train")
        return StageOutcome(label="BLOCKED", reason="lease busy")

    handlers["train"] = blocked
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageBlocked):
        engine.run()
    assert log == list(graph.order[: graph.order.index("train") + 1])


def test_engine_rejects_a_blocked_label_with_no_reason(tmp_path):
    graph = StageGraph.r1()
    handlers = _handlers(graph, [])
    handlers["train"] = lambda context: StageOutcome(label="BLOCKED")
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageFailed):
        engine.run()


def test_engine_rejects_an_unknown_label(tmp_path):
    graph = StageGraph.r1()
    handlers = _handlers(graph, [])
    handlers["train"] = lambda context: StageOutcome(label="bogus")
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageFailed):
        engine.run()


def test_engine_records_the_actual_label_and_never_flattens_it(tmp_path):
    """SKIPPED and NEEDS_USER_DECISION must survive into the recorded status."""
    for label in ("SKIPPED", "NEEDS_USER_DECISION", "FAIL"):
        root = tmp_path / label
        root.mkdir()
        graph = StageGraph.r1()
        handlers = _handlers(graph, [])
        handlers["train"] = lambda context: StageOutcome(label=label, reason="why")
        engine = Engine(graph=graph, handlers=handlers, run_root=root)
        with pytest.raises(StageFailed):
            engine.run()
        recorded = json.loads((root / "status.json").read_text("utf-8"))
        assert recorded["status"] == label
        assert recorded["reason"] == "why"


def test_engine_refuses_a_directory_where_a_file_output_is_declared(tmp_path):
    """A declared output must be a real file, not a same-named directory."""
    graph = StageGraph.r1()
    handlers = _handlers(graph, [])

    def as_directories(context):
        for relative in context.spec.outputs:
            (context.run_root / context.spec.name / relative).mkdir(
                parents=True, exist_ok=True,
            )
        return StageOutcome(label="PASS")

    handlers["validate"] = as_directories
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageFailed) as caught:
        engine.run()
    assert "did not produce" in str(caught.value)


def test_engine_refuses_a_handler_that_returns_a_bare_pass_string(tmp_path):
    """Only a StageOutcome is accepted; a truthy stand-in is a failure."""
    graph = StageGraph.r1()
    handlers = _handlers(graph, [])
    handlers["train"] = lambda context: "PASS"
    engine = Engine(graph=graph, handlers=handlers, run_root=tmp_path)
    with pytest.raises(StageFailed) as caught:
        engine.run()
    assert "StageOutcome" in str(caught.value)


CONCURRENT_PUBLISH = """
import sys, pathlib
sys.path.insert(0, "pipeline_v2")
from slm_train.orchestration.identity_handoff import IdentityHandoff
from slm_train.orchestration.bootstrap import BootstrapIdentity
b = BootstrapIdentity(logical_experiment="e", config_hash="c"*64,
    source_identity_hash="a"*64, tokenizer_identity_hash="b"*64,
    launch_profile_hash="d"*64, requested_runtime_hash="e"*64,
    code_dependency_hash="f"*64, command_hash="0"*64)
h = IdentityHandoff(pathlib.Path(sys.argv[1]))
h.publish(b, run_id="run-1", identity_hash="f"*64)
print("OK")
"""


def test_concurrent_identical_publishes_all_succeed_and_never_clobber(tmp_path):
    """A lost create-if-absent race with the same identity stays idempotent."""
    import subprocess
    import sys

    workers = [
        subprocess.Popen(
            [sys.executable, "-c", CONCURRENT_PUBLISH, str(tmp_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    results = [worker.communicate(timeout=180) for worker in workers]
    assert all("OK" in out for out, _err in results), results
    target = tmp_path / "run-1" / "manifests" / "run_identity.json"
    payload = json.loads(target.read_text("utf-8"))
    assert payload["identity_hash"] == "f" * 64
    assert payload["bootstrap"]["config_hash"] == "c" * 64


def test_a_concurrent_publish_of_a_different_identity_is_refused(tmp_path):
    """Racing a DIFFERENT identity must lose, even if it arrives second."""
    import subprocess
    import sys

    handoff = IdentityHandoff(tmp_path)
    handoff.publish(_bootstrap(), run_id="run-1", identity_hash=HASH_H)
    foreign = CONCURRENT_PUBLISH.replace(chr(34) + "f" + chr(34) + "*64",
                                       chr(34) + "9" + chr(34) + "*64")
    worker = subprocess.run([sys.executable, "-c", foreign, str(tmp_path)],
                            capture_output=True, text=True, timeout=180)
    assert "OK" not in worker.stdout, "a different identity must never win the race"
    assert "HandoffConflict" in worker.stderr, worker.stderr
    payload = json.loads(
        (tmp_path / "run-1" / "manifests" / "run_identity.json").read_text("utf-8"),
    )
    assert payload["identity_hash"] == HASH_H, "the first identity survives"
