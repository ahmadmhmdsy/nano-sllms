"""Task 8 unit tests, part 2: the two-process race and the no-kill recovery rule."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.resources.lease import LeaseConflict, LeaseManager, LeaseState


def lease_extra(pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "process_start_time": "start-" + str(pid),
        "command": "pipeline_v2 run",
        "device": "cuda:0",
        "config_hash": "config-hash",
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _stale_metadata(pid: int) -> dict[str, object]:
    values = lease_extra(pid)
    values["bootstrap_id"] = "boot"
    values["attempt_id"] = "attempt-1"
    values["final_run_id"] = None
    values["identity_hash"] = None
    return values


RACE_SCRIPT = "\n".join(
    [
        "import sys, pathlib",
        "sys.path.insert(0, \"pipeline_v2\")",
        "from slm_train.resources.lease import LeaseManager, LeaseConflict",
        "m = LeaseManager(pathlib.Path(sys.argv[1]))",
        "extra = dict(pid=99, process_start_time=\"start-99\",",
        "    command=\"pipeline_v2 run\", device=\"cuda:0\",",
        "    config_hash=\"h\", timestamp=\"2026-01-01T00:00:00Z\")",
        "print(\"ACQUIRED\" if m.acquire_bootstrap(\"boot2\", \"attempt-9\", extra) else \"NO\")",
    ]
)


def test_two_process_race_admits_exactly_one_holder(tmp_path):
    import subprocess
    import sys

    manager = LeaseManager(tmp_path)
    first = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    completed = subprocess.run(
        [sys.executable, "-c", RACE_SCRIPT, str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert "ACQUIRED" not in completed.stdout, "a second process must not take the lease"
    assert "LeaseConflict" in completed.stderr, completed.stderr
    first.release()


def test_a_stale_sidecar_alone_never_yields_a_free_lease(tmp_path):
    manager = LeaseManager(tmp_path)
    (tmp_path / "run.lock").write_text(json.dumps(_stale_metadata(999999)), encoding="utf-8")
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot2", "attempt-2", lease_extra(2))


def test_recovery_never_kills_a_foreign_process(tmp_path, monkeypatch):
    import os as os_module

    manager = LeaseManager(tmp_path)
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(os_module.getpid()))
    monkeypatch.setattr(
        os_module, "kill",
        lambda *a, **k: pytest.fail("lease code must never kill a process"),
        raising=False,
    )
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot2", "attempt-2", lease_extra(2))
    handle.release()


def test_lease_module_never_kills_a_process() -> None:
    """Signal 0 is a liveness probe; any real termination signal is forbidden."""
    import re

    import slm_train.resources.lease as lease_module

    source = Path(lease_module.__file__).read_text(encoding="utf-8")
    for banned in ("SIGKILL", "SIGTERM", "SIGINT", "taskkill", ".terminate()", "psutil.Process"):
        assert banned not in source, "lease code must never kill: " + banned
    for match in re.findall(r"[.]kill[(]([^)]*)[)]", source):
        assert match.replace(" ", "") == "pid,0", "kill may only probe with signal 0: " + match


def test_a_live_owner_is_never_evicted_by_a_stale_sidecar(tmp_path):
    import os as os_module

    manager = LeaseManager(tmp_path)
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(os_module.getpid()))
    (tmp_path / "run.lock").write_text(
        json.dumps(_stale_metadata(os_module.getpid())), encoding="utf-8"
    )
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot2", "attempt-2", lease_extra(2))
    handle.release()


def test_malformed_sidecar_never_silently_yields_a_free_lease(tmp_path):
    manager = LeaseManager(tmp_path)
    (tmp_path / "run.lock").write_text("{not json", encoding="utf-8")
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    assert manager.state() is LeaseState.BLOCKED


def test_recover_stale_clears_a_sidecar_whose_owner_is_gone(tmp_path):
    manager = LeaseManager(tmp_path)
    (tmp_path / "run.lock").write_text(json.dumps(_stale_metadata(999999)), encoding="utf-8")
    assert manager.inspect_metadata()["pid"] == 999999
    assert manager.recover_stale() is True
    assert manager.state() is LeaseState.FREE
    assert not (tmp_path / "run.lock").exists()
    handle = manager.acquire_bootstrap("boot2", "attempt-2", lease_extra(2))
    assert manager.state() is LeaseState.HELD
    handle.release()


def test_recover_stale_refuses_while_the_owner_is_alive(tmp_path):
    import os as os_module

    manager = LeaseManager(tmp_path)
    handle = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(os_module.getpid()))
    assert manager.recover_stale() is False, "a live owner is never evicted"
    assert manager.state() is LeaseState.BLOCKED
    assert (tmp_path / "run.lock").exists(), "the sidecar must survive a refused recovery"
    handle.release()


def test_recover_stale_refuses_while_the_lock_is_busy(tmp_path):
    """A dead recorded owner is not enough: the lock itself must also be free."""
    other = LeaseManager(tmp_path)
    held = other.acquire_bootstrap("other", "attempt-7", lease_extra(999999))
    manager = LeaseManager(tmp_path)
    try:
        assert manager.inspect_metadata()["pid"] == 999999
        assert manager.recover_stale() is False, "a busy lock is never stolen"
        assert manager.state() is LeaseState.BLOCKED
        assert (tmp_path / "run.lock").exists(), "a refused recovery keeps the sidecar"
    finally:
        held.release()


def test_recover_stale_tolerates_a_malformed_sidecar(tmp_path):
    manager = LeaseManager(tmp_path)
    (tmp_path / "run.lock").write_text("not json at all", encoding="utf-8")
    assert manager.recover_stale() is True, "a malformed sidecar has no provable owner"
    assert not (tmp_path / "run.lock").exists()


def test_a_fresh_manager_may_take_a_released_lease(tmp_path):
    first = LeaseManager(tmp_path)
    handle = first.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    handle.release()
    assert first.state() is LeaseState.FREE
    second = LeaseManager(tmp_path)
    taken = second.acquire_bootstrap("boot2", "attempt-2", lease_extra(2))
    assert second.state() is LeaseState.HELD
    taken.release()
