"""Repository-level GPU lease.

One protected GPU job at a time, enforced across processes by a file lock held
for the whole GPU stage, and re-checked in-process so one manager can never
hold two leases either.

Fail-closed policy: a busy, malformed, stale, or unverifiable lease is BLOCKED.
The lease is never stolen and never force-cleared, and this module never kills a
process. Replacing a stale lease is an explicit recovery step that must first
prove the recorded owner is gone and the GPU lock is free; there is no implicit
repair on the acquisition path.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from slm_train.artifacts.atomic import atomic_write_bytes


class LeaseConflict(RuntimeError):
    """The GPU lease could not be taken. Always fail closed, never force."""


class LeaseState:
    """The three observable states of a repository lease."""

    FREE = "FREE"
    HELD = "HELD"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class LeaseMetadata:
    """The facts about the holder, recorded once at acquisition."""

    bootstrap_id: str
    attempt_id: str
    pid: int
    process_start_time: str
    command: str
    device: str
    config_hash: str
    timestamp: str
    final_run_id: Optional[str] = None
    identity_hash: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


REQUIRED_FACTS: tuple[str, ...] = (
    "pid",
    "process_start_time",
    "command",
    "device",
    "config_hash",
    "timestamp",
)


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lease fact " + label + " must be a non-empty string")
    return value


def _require_run_component(value: object, label: str) -> str:
    """A run id is a single safe path component, never a path or a traversal."""
    text = _require_text(value, label)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(label + " must be a single path component")
    if ":" in text or text.startswith("~"):
        raise ValueError(label + " must not be drive-qualified or home-relative")
    return text


def process_start_time_of(pid: int) -> Optional[float]:
    """The moment the OS says this pid's process started, or None.

    A recycled pid gets a NEW start time: comparing it against the recorded
    one is the difference between 'the owner is alive' and 'a stranger reused
    the pid'. Read-only: this sends no signal and changes nothing.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        import psutil

        factory = getattr(psutil, "Process")
        return float(factory(pid).create_time())
    except Exception:
        return None


def _owner_start_matches(pid: int, recorded: object) -> Optional[bool]:
    """True/False when both start times are known; None when unknowable."""
    actual = process_start_time_of(pid)
    if actual is None:
        return None
    try:
        expected = float(recorded)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return abs(actual - expected) < 5.0


def process_is_alive(pid: int) -> bool:
    """Is the recorded owner still running? Read-only: this sends no signal."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        return _pid_alive_without_psutil(pid)


def _pid_alive_without_psutil(pid: int) -> bool:
    """POSIX fallback. Signal 0 is an existence probe, not a termination."""
    import os

    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True



@dataclass
class LeaseHandle:
    """A live lease. The lock stays held for as long as this object exists."""

    manager: "LeaseManager"
    metadata: LeaseMetadata
    _released: bool = False

    def bind_final(self, final_run_id: str, identity_hash: str) -> None:
        """Bind the lease to a final run identity, once, and only to the same one."""
        if self._released:
            raise LeaseConflict("lease is already released")
        run_id = _require_run_component(final_run_id, "final_run_id")
        identity = _require_text(identity_hash, "identity_hash")
        current = self.metadata
        if current.final_run_id is not None and current.final_run_id != run_id:
            raise LeaseConflict("lease is already bound to another run")
        if current.identity_hash is not None and current.identity_hash != identity:
            raise LeaseConflict("lease identity does not match")
        updated = LeaseMetadata(
            **{
                **current.as_dict(),
                "final_run_id": run_id,
                "identity_hash": identity,
            }
        )
        self.manager._write_metadata(updated)
        self.metadata = updated

    def release(self) -> None:
        """Release the lock. Idempotent, so a failure path can always release."""
        if self._released:
            return
        self._released = True
        self.manager._release()


class LeaseManager:
    """Acquires, binds, releases, and explicitly recovers the GPU lease."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / "gpu.lock"), timeout=0)
        self.metadata_path = self.root / "run.lock"
        self._state = LeaseState.FREE
        self._handle: Optional[LeaseHandle] = None

    def state(self) -> str:
        return self._state

    def acquire_bootstrap(
        self, bootstrap_id: str, attempt_id: str, extra: dict[str, object]
    ) -> LeaseHandle:
        """Take the lease for a bootstrap attempt, or raise LeaseConflict."""
        bootstrap = _require_text(bootstrap_id, "bootstrap_id")
        attempt = _require_text(attempt_id, "attempt_id")
        if not isinstance(extra, dict):
            raise ValueError("lease facts must be a mapping")
        missing = [key for key in REQUIRED_FACTS if key not in extra]
        if missing:
            raise ValueError("missing lease facts: " + ", ".join(missing))
        pid = extra["pid"]
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("lease fact pid must be a positive integer")
        if self._handle is not None and not self._handle._released:
            self._state = LeaseState.BLOCKED
            raise LeaseConflict("this manager already holds the repository GPU lease")
        prior = self.inspect_metadata()
        if prior is not None:
            owner = prior.get("pid")
            if isinstance(owner, int) and not isinstance(owner, bool) and process_is_alive(owner):
                self._state = LeaseState.BLOCKED
                raise LeaseConflict("a prior lease holder is still running: pid " + str(owner))
            self._state = LeaseState.BLOCKED
            raise LeaseConflict("a prior lease sidecar is still present; recover it explicitly")
        try:
            self.lock.acquire()
        except Timeout as exc:
            self._state = LeaseState.BLOCKED
            raise LeaseConflict("repository GPU lease is busy") from exc
        try:
            metadata = LeaseMetadata(
                bootstrap_id=bootstrap,
                attempt_id=attempt,
                pid=pid,
                process_start_time=_require_text(
                    extra["process_start_time"], "process_start_time",
                ),
                command=_require_text(extra["command"], "command"),
                device=_require_text(extra["device"], "device"),
                config_hash=_require_text(extra["config_hash"], "config_hash"),
                timestamp=_require_text(extra["timestamp"], "timestamp"),
            )
            self._write_metadata(metadata)
            handle = LeaseHandle(self, metadata)
            self._handle = handle
            self._state = LeaseState.HELD
            return handle
        except Exception:
            self._release()
            raise

    def inspect_metadata(self) -> Optional[dict]:
        """The recorded sidecar, or None. Unreadable sidecars raise, never hide."""
        if not self.metadata_path.exists():
            return None
        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._state = LeaseState.BLOCKED
            raise LeaseConflict("lease sidecar is unreadable; recover it explicitly") from exc
        if not isinstance(raw, dict):
            self._state = LeaseState.BLOCKED
            raise LeaseConflict("lease sidecar is malformed; recover it explicitly")
        return raw

    def recover_stale(self) -> bool:
        """Replace a stale sidecar, but only after proving owner and GPU work are gone.

        Returns False and leaves the lease BLOCKED whenever any proof fails. This
        never signals the recorded owner; it only observes it.
        """
        try:
            prior = self.inspect_metadata()
        except LeaseConflict:
            prior = {}
        if prior is not None:
            owner = prior.get("pid")
            if isinstance(owner, int) and not isinstance(owner, bool) and process_is_alive(owner):
                # A live pid is the previous owner only if its START TIME also
                # matches the recorded one; otherwise the OS recycled the pid
                # and the recorded owner is gone. The default stays fail-closed
                # (BLOCKED) whenever the start time cannot be proven.
                start_match = _owner_start_matches(owner, prior.get("process_start_time"))
                if start_match is not False:
                    self._state = LeaseState.BLOCKED
                    return False
        try:
            self.lock.acquire()
        except Timeout:
            self._state = LeaseState.BLOCKED
            return False
        try:
            if self.metadata_path.exists():
                self.metadata_path.unlink()
        except OSError:
            self._state = LeaseState.BLOCKED
            try:
                self.lock.release()
            except Exception:
                pass
            return False
        try:
            self.lock.release()
        except Exception:
            pass
        self._state = LeaseState.FREE
        return True

    def _write_metadata(self, metadata: LeaseMetadata) -> None:
        payload = json.dumps(metadata.as_dict(), sort_keys=True).encode("utf-8")
        atomic_write_bytes(self.metadata_path, payload, replace_existing=True)

    def _release(self) -> None:
        # releasing removes only the sidecar this manager wrote; a foreign one is
        # left untouched so recovery still has something to prove
        if self._handle is not None:
            try:
                if self.metadata_path.exists():
                    self.metadata_path.unlink()
            except OSError:
                pass
        self._handle = None
        try:
            self.lock.release()
        finally:
            self._state = LeaseState.FREE

