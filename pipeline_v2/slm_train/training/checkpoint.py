"""Atomic checkpoint publication.

A checkpoint is published in an order that makes a partial one unselectable:
every state file is written to a temporary sibling and fsynced, each is verified
against the hash and size the manifest will claim, the manifest is published
immutably, and only then is COMPLETE.json written. A crash at any point leaves
a directory with no completion marker, and a checkpoint without a completion
marker is never selected.

Files are written through the log-segment style create-if-absent publish, so a
second attempt at the same checkpoint number fails instead of overwriting.
"""
from __future__ import annotations

from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.artifacts.manifest import write_manifest
from slm_train.artifacts.markers import MARKER_NAME, write_completion_marker
from slm_train.contracts.checkpoint import CheckpointManifest, ResumeFile
from slm_train.contracts.hash import sha256_bytes
from slm_train.training.state import CheckpointState

CHECKPOINTS_DIR = "checkpoints"
MANIFEST_NAME = "manifest.json"
STATE_DIR = "state"


class CheckpointIncomplete(RuntimeError):
    """A checkpoint could not be published completely. Never selectable."""


def checkpoint_dir(run_root: Path, number: int) -> Path:
    """The destination for one checkpoint. A number, never a caller string."""
    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        raise ValueError("checkpoint number must be a non-negative integer")
    return Path(run_root) / CHECKPOINTS_DIR / ("step-" + str(number).zfill(8))


def is_complete(run_root: Path, number: int) -> bool:
    """A checkpoint is complete only when its marker is on disk."""
    return (checkpoint_dir(run_root, number) / MARKER_NAME).is_file()


def _write_and_verify(target: Path, payload: bytes, role: str) -> ResumeFile:
    """Publish one state file immutably, then read it back and verify it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, payload)
    written = target.read_bytes()
    if written != payload:
        reason = "state file did not read back as written: " + str(role)
        raise CheckpointIncomplete(reason)
    return ResumeFile(
        path=target.name,
        size_bytes=len(written),
        sha256=sha256_bytes(written),
        role=role,
    )


def publish_checkpoint(
    run_root: Path,
    number: int,
    state: CheckpointState,
    payloads: dict,
    manifest: CheckpointManifest,
) -> Path:
    """Publish a complete checkpoint. Returns its directory, or raises.

    A raised CheckpointIncomplete always leaves the destination WITHOUT a
    completion marker, so an interrupted publish can never be selected.
    """
    if not isinstance(state, CheckpointState):
        raise ValueError("state must be a CheckpointState")
    if not isinstance(payloads, dict):
        raise ValueError("payloads must map every state role to its bytes")
    wanted = state.files()
    absent = sorted(set(wanted) - set(payloads))
    if absent:
        raise ValueError("missing state payloads: " + ", ".join(absent))
    extra = sorted(set(payloads) - set(wanted))
    if extra:
        raise ValueError("undeclared state payloads: " + ", ".join(extra))
    destination = checkpoint_dir(run_root, number)
    # An interrupted publish leaves a destination without a completion marker
    # (the write order guarantees it) -- that debris is not user evidence, it
    # is the crash residue of a failed write, and a rerun must be able to
    # rebuild its own checkpoint. A complete destination is never touched.
    if destination.exists() and not (destination / MARKER_NAME).is_file():
        import shutil

        shutil.rmtree(destination)
    state_dir = destination / STATE_DIR
    files = []
    certified = {}
    for role, name in wanted.items():
        relative = STATE_DIR + "/" + name
        files.append(_write_and_verify(state_dir / name, payloads[role], role))
        certified[relative] = payloads[role]
    published = manifest.model_copy(update={"files": tuple(files), "complete": True})
    hashes = {
        relative: sha256_bytes(payload) for relative, payload in sorted(certified.items())
    }
    write_manifest(destination, MANIFEST_NAME, published.model_dump(mode="json"))
    write_completion_marker(destination, certified, MANIFEST_NAME, hashes)
    if not (destination / MARKER_NAME).is_file():
        reason = "completion marker was not written: " + str(destination)
        raise CheckpointIncomplete(reason)
    return destination
