"""Create-if-absent handoff from a bootstrap attempt to an immutable run.

The handoff is the point where provisional work becomes a run. It publishes
manifests/run_identity.json exactly once per run id: an identical republication
is idempotent, and a DIFFERENT identity at an existing destination is BLOCKED and
never overwritten. A run id is a single safe path component, never a path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.orchestration.bootstrap import BootstrapIdentity

RUN_IDENTITY_RELATIVE = ("manifests", "run_identity.json")


class HandoffConflict(RuntimeError):
    """A different identity already occupies this run id. Never overwrite."""


def _require_run_component(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_id must be a non-empty string")
    text = value.strip()
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError("run_id must be a single path component")
    if ":" in text or text.startswith("~"):
        raise ValueError("run_id must not be drive-qualified or home-relative")
    return text


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(label + " must be a 64-character digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(label + " must be a lowercase hex digest")
    return value


class IdentityHandoff:
    """Publishes the run identity once, and refuses to replace a different one."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / _require_run_component(run_id)

    def identity_path(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        for part in RUN_IDENTITY_RELATIVE:
            path = path / part
        return path

    def publish(
        self, identity: BootstrapIdentity, run_id: str, identity_hash: str
    ) -> Path:
        """Create-if-absent. Same identity twice is a no-op; a new one conflicts."""
        if not isinstance(identity, BootstrapIdentity):
            raise ValueError("identity must be a BootstrapIdentity")
        run = _require_run_component(run_id)
        digest = _require_digest(identity_hash, "identity_hash")
        target = self.identity_path(run)
        payload = {
            "run_id": run,
            "identity_hash": digest,
            "bootstrap_digest": identity.digest(),
            "bootstrap": identity.as_dict(),
        }
        existing = self.load(run, missing_ok=True) if target.exists() else None
        if existing is not None:
            self._refuse_if_foreign(existing, run, digest)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        # immutable publication: an exclusive link fails if anyone raced us here
        try:
            atomic_write_bytes(
                target,
                json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
                replace_existing=False,
            )
        except FileExistsError as exc:
            # we lost the race. Re-read what actually landed: an identical identity
            # is still idempotent, a different one is still a conflict.
            winner = self.load(run)
            self._refuse_if_foreign(winner, run, digest)
            return target
        return target

    @staticmethod
    def _refuse_if_foreign(existing: dict, run: str, digest: str) -> None:
        """Refuse a different identity at an existing destination. Never overwrite."""
        if existing.get("identity_hash") != digest:
            raise HandoffConflict(
                "run " + run + " is already bound to a different identity; "
                "refusing to overwrite"
            )

    def load(self, run_id: str, missing_ok: bool = False) -> Optional[dict]:
        """Read back what is actually on disk, never what was intended."""
        target = self.identity_path(run_id)
        if not target.exists():
            if missing_ok:
                return None
            raise HandoffConflict("run identity is not published: " + str(run_id))
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffConflict("run identity is unreadable: " + str(run_id)) from exc
        if not isinstance(raw, dict) or "identity_hash" not in raw:
            raise HandoffConflict("run identity is malformed: " + str(run_id))
        return raw
