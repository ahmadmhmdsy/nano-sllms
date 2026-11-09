"""Task 8 unit tests: the repository GPU lease, its recovery, and the handoff CAS."""
from __future__ import annotations

import json

import pytest

from slm_train.resources.lease import (
    LeaseConflict,
    LeaseManager,
    LeaseState,
)


def lease_extra(pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "process_start_time": "start-" + str(pid),
        "command": "pipeline_v2 run",
        "device": "cuda:0",
        "config_hash": "config-hash",
        "timestamp": "2026-01-01T00:00:00Z",
    }


def test_second_lease_is_blocked(tmp_path):
    manager = LeaseManager(tmp_path)
    first = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot", "attempt-2", lease_extra(2))
    first.release()


def test_final_handoff_is_cas(tmp_path):
    lease = LeaseManager(tmp_path).acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    lease.bind_final("run-1", "identity-1")
    with pytest.raises(LeaseConflict):
        lease.bind_final("run-2", "identity-2")
    lease.release()


def test_rebinding_the_same_identity_is_idempotent(tmp_path):
    lease = LeaseManager(tmp_path).acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    lease.bind_final("run-1", "identity-1")
    lease.bind_final("run-1", "identity-1")
    assert lease.metadata.final_run_id == "run-1"
    lease.release()


def test_bind_final_refuses_a_mismatched_identity_on_the_same_run(tmp_path):
    lease = LeaseManager(tmp_path).acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    lease.bind_final("run-1", "identity-1")
    with pytest.raises(LeaseConflict):
        lease.bind_final("run-1", "other-identity")
    lease.release()


@pytest.mark.parametrize("bad", ["", None, "..", "../x", "/abs"])
def test_bind_final_refuses_an_unsafe_run_id(tmp_path, bad) -> None:
    lease = LeaseManager(tmp_path).acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    with pytest.raises(ValueError):
        lease.bind_final(bad, "identity-1")
    lease.release()


def test_lease_is_reusable_after_release(tmp_path):
    manager = LeaseManager(tmp_path)
    first = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    first.release()
    second = manager.acquire_bootstrap("boot", "attempt-2", lease_extra(2))
    assert second.metadata.attempt_id == "attempt-2"
    second.release()


def test_release_is_idempotent(tmp_path):
    manager = LeaseManager(tmp_path)
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    handle.release()
    handle.release()
    assert manager.state() is LeaseState.FREE


def test_metadata_is_written_atomically_and_readable(tmp_path):
    manager = LeaseManager(tmp_path)
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    on_disk = json.loads((tmp_path / "run.lock").read_text("utf-8"))
    assert on_disk["bootstrap_id"] == "boot"
    assert on_disk["attempt_id"] == "attempt-1"
    assert on_disk["pid"] == 1
    assert on_disk["final_run_id"] is None
    handle.release()


def test_state_reports_held_while_acquired(tmp_path):
    manager = LeaseManager(tmp_path)
    assert manager.state() is LeaseState.FREE
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    assert manager.state() is LeaseState.HELD
    handle.release()
    assert manager.state() is LeaseState.FREE


def test_acquire_rejects_malformed_extra_facts(tmp_path):
    manager = LeaseManager(tmp_path)
    for missing in ("pid", "process_start_time", "command", "device", "config_hash", "timestamp"):
        extra = lease_extra(1)
        extra.pop(missing)
        with pytest.raises(ValueError):
            manager.acquire_bootstrap("boot", "attempt-1", extra)
    assert manager.state() is LeaseState.FREE
