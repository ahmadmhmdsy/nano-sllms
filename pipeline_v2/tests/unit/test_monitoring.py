"""Task 9 unit tests, part 2: event streams, health policy, and metrics."""
from __future__ import annotations

import json

import pytest

from slm_train.artifacts.log_index import (
    LOG_KINDS,
    published_segment_numbers,
    verify_all_streams,
    verify_stream,
)
from slm_train.monitoring.bus import (
    RUN_EVENT_NAMES,
    STREAM_FOR_EVENT,
    EventBus,
    EventBusError,
)
from slm_train.monitoring.health import (
    HEALTH_LABELS,
    HealthDecision,
    classify_health,
    is_finite_value,
)
from slm_train.monitoring.metrics import (
    MetricError,
    MetricValue,
    require_finite_metric,
)


def records_for(run_root, kind) -> list:
    from slm_train.artifacts.log_index import segment_path

    out = []
    for number in published_segment_numbers(run_root, kind):
        path = segment_path(run_root, kind, number)
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


# --- event bus ------------------------------------------------------------


def test_the_designed_run_events_are_exactly_the_ones_the_bus_accepts():
    assert set(RUN_EVENT_NAMES) == {
        "run_started", "stage_started", "step_completed", "metric_emitted",
        "evaluation_completed", "checkpoint_completed", "health_warning",
        "health_failure", "stage_completed", "run_completed",
        # operator-diagnostics additions (2026-09-27 friction plan)
        "vram_probe", "oom_accounting", "event_refused",
    }


@pytest.mark.parametrize("name", RUN_EVENT_NAMES)
def test_every_run_event_maps_to_a_designed_log_stream(name):
    assert STREAM_FOR_EVENT[name] in LOG_KINDS


def test_a_run_event_carries_a_monotonic_record_sequence_and_the_run_identity(tmp_path):
    bus = EventBus(tmp_path, "run-42")
    for step in range(3):
        bus.emit("step_completed", {"step": step})
    bus.flush()
    published = records_for(tmp_path, "metrics")
    assert [item["record"] for item in published] == [1, 2, 3]
    assert {item["run_id"] for item in published} == {"run-42"}


def test_an_unflushed_event_is_buffered_and_never_published(tmp_path):
    bus = EventBus(tmp_path, "run-1")
    bus.emit("run_started", {"stage": "validate"})
    assert bus.pending("monitor") == 1
    assert published_segment_numbers(tmp_path, "monitor") == ()
    assert not (tmp_path / "logs" / "monitor").exists()


def test_a_reopened_bus_continues_the_published_record_sequence(tmp_path):
    first = EventBus(tmp_path, "run-1")
    first.emit("run_started", {})
    first.flush()
    second = EventBus(tmp_path, "run-1")
    second.emit("stage_started", {"stage": "validate"})
    second.flush()
    published = records_for(tmp_path, "monitor")
    assert [item["record"] for item in published] == [1, 2]


def test_segment_numbers_never_repeat_across_flushes(tmp_path):
    bus = EventBus(tmp_path, "run-1", max_records=2)
    for step in range(5):
        bus.emit("step_completed", {"step": step})
    bus.flush()
    numbers = published_segment_numbers(tmp_path, "metrics")
    assert numbers == (1, 2, 3)
    assert len(records_for(tmp_path, "metrics")) == 5


def test_an_unknown_run_event_is_refused_rather_than_written_somewhere(tmp_path):
    bus = EventBus(tmp_path, "run-1")
    with pytest.raises(EventBusError, match="unknown run event"):
        bus.emit("not_a_real_event", {})
    assert bus.pending("monitor") == 0


def test_a_payload_that_cannot_be_encoded_does_not_burn_a_record_number(tmp_path):
    """A refused emit must leave the stream contiguous, not poisoned with a gap."""
    bus = EventBus(tmp_path, "run-1")
    bus.emit("step_completed", {"step": 1})
    with pytest.raises(TypeError):
        bus.emit("step_completed", {"bad": object()})
    bus.emit("step_completed", {"step": 2})
    bus.flush()
    published = records_for(tmp_path, "metrics")
    assert [item["record"] for item in published] == [1, 2]
    assert verify_stream(tmp_path, "metrics").ok


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_the_bus_refuses_an_empty_run_identity(bad, tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        EventBus(tmp_path, bad)


# --- stream verification --------------------------------------------------


def test_every_designed_stream_including_the_lease_stream_verifies(tmp_path):
    bus = EventBus(tmp_path, "run-1")
    bus.emit("run_started", {})
    bus.emit("step_completed", {"step": 1})
    bus.flush()
    report = verify_all_streams(tmp_path)
    assert set(report) == set(LOG_KINDS)
    for kind, result in report.items():
        assert result.ok, (kind, result.problems)


def test_a_tampered_segment_is_detected_by_hash(tmp_path):
    bus = EventBus(tmp_path, "run-1")
    bus.emit("step_completed", {"step": 1})
    bus.flush()
    from slm_train.artifacts.log_index import segment_path

    target = segment_path(tmp_path, "metrics", 1)
    target.write_bytes(target.read_bytes() + b"{\"record\":99}\n")
    result = verify_stream(tmp_path, "metrics")
    assert not result.ok
    assert any("hash mismatch" in problem for problem in result.problems)


def test_a_deleted_published_segment_is_detected(tmp_path):
    bus = EventBus(tmp_path, "run-1")
    bus.emit("step_completed", {"step": 1})
    bus.flush()
    from slm_train.artifacts.log_index import segment_path

    segment_path(tmp_path, "metrics", 1).unlink()
    result = verify_stream(tmp_path, "metrics")
    assert not result.ok
    assert any("missing published segment" in problem for problem in result.problems)


def test_two_buses_on_one_run_root_never_publish_a_duplicate_record(tmp_path):
    """Concurrent writers must renumber under the flush lock, not collide."""
    first = EventBus(tmp_path, "run-1")
    second = EventBus(tmp_path, "run-1")
    first.emit("step_completed", {"who": 1})
    second.emit("step_completed", {"who": 2})
    first.flush()
    second.flush()
    published = records_for(tmp_path, "metrics")
    assert [item["record"] for item in published] == [1, 2]
    assert {item["payload"]["who"] for item in published} == {1, 2}
    assert verify_stream(tmp_path, "metrics").ok


def test_ten_buses_on_one_run_root_publish_one_contiguous_sequence(tmp_path):
    buses = [EventBus(tmp_path, "run-1") for _ in range(10)]
    for index, bus in enumerate(buses):
        bus.emit("step_completed", {"who": index})
    for bus in buses:
        bus.flush()
    published = records_for(tmp_path, "metrics")
    assert [item["record"] for item in published] == list(range(1, 11))
    assert verify_stream(tmp_path, "metrics").ok


def test_a_record_gap_is_detected_even_when_every_hash_matches(tmp_path):
    from slm_train.artifacts.log_index import append_log_segment

    bus = EventBus(tmp_path, "run-1")
    bus.emit("step_completed", {"step": 1})
    bus.flush()
    # a second segment whose first record skips ahead, with a correct hash
    payload = (json.dumps({"stream": "metrics", "record": 7, "event": "step_completed",
                       "run_id": "run-1", "payload": {}}, sort_keys=True) + "\n")
    append_log_segment(tmp_path, "metrics", 2, payload.encode("utf-8"))
    result = verify_stream(tmp_path, "metrics")
    assert not result.ok
    assert any("record gap" in problem for problem in result.problems)


def test_an_empty_stream_verifies_as_zero_records_rather_than_failing(tmp_path):
    result = verify_stream(tmp_path, "monitor")
    assert result.ok
    assert result.records == 0


# --- health policy --------------------------------------------------------


@pytest.mark.parametrize("value, finite", [
    (1.0, True), (0, True), (-3, True),
    (float("nan"), False), (float("inf"), False), (float("-inf"), False),
    (True, False), (None, False), ("1.0", False), ([1.0], False),
])
def test_finiteness_is_decided_by_value_not_by_intent(value, finite):
    assert is_finite_value(value) is finite


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_loss_is_a_failure(value):
    decision = classify_health(loss=value, gradient_norm=1.0)
    assert decision.label == "FAIL"
    assert "loss" in decision.reason


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_gradient_norm_is_a_failure(value):
    decision = classify_health(loss=1.0, gradient_norm=value)
    assert decision.label == "FAIL"
    assert "gradient_norm" in decision.reason


@pytest.mark.parametrize("message", [
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: CUDA error: out of memory",
    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED",
])
def test_out_of_memory_is_blocked_and_preserves_the_last_valid_checkpoint(message):
    decision = classify_health(message=message)
    assert decision.label == "BLOCKED"
    assert decision.oom
    assert "preserved" in decision.reason


@pytest.mark.parametrize("message", [
    "DataLoader worker (pid 123) is killed by signal: Bus error",
    "unable to write to shared memory /dev/shm",
])
def test_a_shared_memory_spill_is_oom_equivalent(message):
    decision = classify_health(message=message)
    assert decision.label == "BLOCKED"
    assert decision.facts["memory_class"] == "shared_memory"


def test_a_healthy_step_passes():
    decision = classify_health(loss=0.5, gradient_norm=1.0)
    assert decision.label == "PASS"
    assert decision.reason


def test_health_facts_are_recorded_verbatim_and_never_rewritten():
    facts = {"vram_used_bytes": 100, "temperature_c": 80.0}
    decision = classify_health(loss=0.5, gradient_norm=1.0, facts=facts)
    assert decision.facts["vram_used_bytes"] == 100
    assert decision.facts["temperature_c"] == 80.0


@pytest.mark.parametrize("label", ["PASS", "OK", "pass", "", None])
def test_a_health_decision_refuses_a_label_outside_the_vocabulary(label):
    if label in HEALTH_LABELS:
        pytest.skip("not outside the vocabulary")
    with pytest.raises(ValueError, match="health label"):
        HealthDecision(label=label, reason="because")


def test_a_health_decision_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="reason"):
        HealthDecision(label="PASS", reason="   ")


# --- metrics --------------------------------------------------------------


def test_a_finite_metric_round_trips():
    assert MetricValue("loss", 0.25, 7).to_dict() == {
        "name": "loss", "value": 0.25, "step": 7,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_metric_is_refused_before_it_is_recorded(value):
    with pytest.raises(MetricError, match="loss"):
        MetricValue("loss", value, 1)
    with pytest.raises(MetricError, match="loss"):
        require_finite_metric("loss", value)


@pytest.mark.parametrize("value", [True, None, "0.5", [0.5]])
def test_a_non_numeric_metric_is_refused(value):
    with pytest.raises(MetricError):
        MetricValue("loss", value, 1)


@pytest.mark.parametrize("step", [-1, 1.5, True, "3"])
def test_a_metric_step_must_be_a_non_negative_integer(step):
    with pytest.raises(MetricError, match="step"):
        MetricValue("loss", 0.5, step)


# --- Phase 1 operator diagnostics (2026-09-26 plan) -------------------------


def test_the_vram_probe_and_oom_accounting_events_are_declared(tmp_path):
    assert "vram_probe" in RUN_EVENT_NAMES
    assert "oom_accounting" in RUN_EVENT_NAMES
    assert "event_refused" in RUN_EVENT_NAMES
    assert STREAM_FOR_EVENT["vram_probe"] == "monitor"
    assert STREAM_FOR_EVENT["oom_accounting"] == "monitor"
    assert STREAM_FOR_EVENT["event_refused"] == "monitor"


def test_emit_or_quarantine_publishes_a_refused_name_instead_of_raising(tmp_path):
    bus = EventBus(tmp_path, "run-quarantine")
    record = bus.emit_or_quarantine("vram_probe_the_unknown", {"size": 1})
    assert record["event"] == "event_refused"
    bus.flush()
    published = records_for(tmp_path, "monitor")
    assert published
    assert published[-1]["event"] == "event_refused"
    assert published[-1]["payload"]["refused_event"] == "vram_probe_the_unknown"
    assert published[-1]["payload"]["payload"] == {"size": 1}


def test_emit_or_quarantine_passes_through_declared_events(tmp_path):
    bus = EventBus(tmp_path, "run-passthrough")
    record = bus.emit_or_quarantine("vram_probe", {"step": 4, "reserved_bytes": 8})
    assert record["event"] == "vram_probe"
    bus.flush()
    published = records_for(tmp_path, "monitor")
    assert published[-1]["event"] == "vram_probe"
    assert published[-1]["payload"]["step"] == 4


def test_an_unknown_name_still_raises_on_the_strict_path(tmp_path):
    bus = EventBus(tmp_path, "run-strict")
    with pytest.raises(EventBusError, match="unknown run event"):
        bus.emit("not_a_declared_event", {})


def test_flush_on_del_publishes_a_buffered_record(tmp_path):
    bus = EventBus(tmp_path, "run-finalizer")
    bus.emit("vram_probe", {"step": 1, "reserved_bytes": 8})
    del bus  # finalizer: the buffered record survives the collector
    published = records_for(tmp_path, "monitor")
    assert any(r["event"] == "vram_probe" and r["payload"]["step"] == 1 for r in published)
