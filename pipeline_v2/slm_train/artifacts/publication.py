"""Final publication: the last write of a successful run.

The order below is fixed, and each step is verified before the next one runs:

    complete checkpoint
      -> best selection
      -> stage merged/exported model if required
      -> stage tokenizer
      -> read-back staged model and tokenizer
      -> publish manifests/final_publication.json
      -> read back and schema-validate final_publication.json
      -> publish COMPLETE.json

Nothing is published optimistically. Every staged file is written with
create-if-absent semantics, read back, and hashed from the bytes actually on
disk. COMPLETE.json is written last and binds the final publication hash, the
model and tokenizer hashes, the required checkpoint/evaluation/report hashes,
the run identity hash, and the log index hash. A run with no COMPLETE.json is
not a successful run, and nothing here ever implies otherwise.
"""
from __future__ import annotations

from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.artifacts.manifest import write_manifest
from slm_train.artifacts.markers import write_completion_marker
from slm_train.contracts.final import FinalPublicationManifest, PublishedFile
from slm_train.contracts.hash import sha256_file

__all__ = [
    "COMPLETE_NAME",
    "FINAL_MANIFEST",
    "PublicationError",
    "artifact_record",
    "publish_final",
    "read_back_final_publication",
    "stage_artifact",
    "write_run_complete",
]

COMPLETE_NAME = "COMPLETE.json"
FINAL_MANIFEST = "manifests/final_publication.json"
MODEL_DIR = "final/model"
TOKENIZER_DIR = "final/tokenizer"


class PublicationError(RuntimeError):
    """Publication did not complete. Nothing was marked successful."""


def stage_artifact(run_root: Path, directory: str, name: str, payload: bytes) -> PublishedFile:
    """Stage one file create-if-absent and record it from the bytes on disk."""
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise PublicationError("a staged file must be a single name: " + repr(name))
    destination = Path(run_root) / directory / name
    atomic_write_bytes(destination, payload)
    if not destination.is_file():
        raise PublicationError("staged file was not written: " + str(destination))
    return PublishedFile(
        schema_version=1,
        path=directory + "/" + name,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
    )


def _read_back(run_root: Path, record: PublishedFile) -> PublishedFile:
    """Re-stat and re-hash a staged file. A mismatch is a publication failure."""
    destination = Path(run_root) / record.path
    if not destination.is_file():
        raise PublicationError("staged artifact is missing on read-back: " + record.path)
    size = destination.stat().st_size
    if size != record.size_bytes:
        reason = "staged artifact changed size on read-back: " + record.path
        raise PublicationError(reason)
    digest = sha256_file(destination)
    if digest != record.sha256:
        raise PublicationError("staged artifact changed hash on read-back: " + record.path)
    return PublishedFile(
        schema_version=1, path=record.path, size_bytes=size, sha256=digest,
    )


def read_back_final_publication(run_root: Path) -> FinalPublicationManifest:
    """Load, schema-validate, and re-verify every file the manifest claims."""
    # model_validate_json, not model_validate: the strict object mode rejects a
    # JSON array for a tuple field, and a manifest on disk is exactly that.
    document = (Path(run_root) / FINAL_MANIFEST).read_bytes()
    manifest = FinalPublicationManifest.model_validate_json(document)
    for record in manifest.final_files:
        _read_back(run_root, record)
    return manifest


def publish_final(
    run_root: Path,
    *,
    run_id: str,
    selected_checkpoint: str,
    model_payloads: dict,
    tokenizer_payloads: dict,
    required_hashes: dict,
) -> str:
    """Stage, read back, and publish final_publication.json. Not COMPLETE.json.

    This is step six of seven. The marker is written by write_run_complete, and
    only that function may do it, so no caller can mark a run successful without
    passing through the read-back and the COMPLETE binding.
    """
    if not run_id or not run_id.strip():
        raise PublicationError("publication needs a run id")
    if not selected_checkpoint or not selected_checkpoint.strip():
        raise PublicationError("publication needs the selected checkpoint")
    staged = []
    for name, payload in model_payloads.items():
        staged.append(stage_artifact(run_root, MODEL_DIR, name, payload))
    for name, payload in tokenizer_payloads.items():
        staged.append(stage_artifact(run_root, TOKENIZER_DIR, name, payload))
    if not staged:
        raise PublicationError("publication staged no artifacts at all")
    verified = tuple(_read_back(run_root, record) for record in staged)
    manifest = FinalPublicationManifest(
        schema_version=1,
        run_id=run_id,
        evidence_root=str(run_id),
        selected_checkpoint=selected_checkpoint,
        final_files=verified,
        read_back_passed=True,
        payload_hashes=dict(required_hashes),
    )
def publish_final(
    run_root: Path,
    *,
    run_id: str,
    selected_checkpoint: str,
    model_payloads: dict,
    tokenizer_payloads: dict,
    required_hashes: dict,
) -> str:
    """Stage, read back, and publish final_publication.json. Not COMPLETE.json.

    This is step six of seven. The marker is written by write_run_complete, and
    only that function may do it, so no caller can mark a run successful without
    passing through the read-back and the COMPLETE binding.

    The staged file hashes are added to the required set here, so a caller cannot
    publish a model file and then forget to bind it.
    """
    if not run_id or not run_id.strip():
        raise PublicationError("publication needs a run id")
    if not selected_checkpoint or not selected_checkpoint.strip():
        raise PublicationError("publication needs the selected checkpoint")
    if not isinstance(required_hashes, dict):
        raise PublicationError("required hashes must be a mapping")
    staged = []
    for name, payload in model_payloads.items():
        staged.append(stage_artifact(run_root, MODEL_DIR, name, payload))
    for name, payload in tokenizer_payloads.items():
        staged.append(stage_artifact(run_root, TOKENIZER_DIR, name, payload))
    if not staged:
        raise PublicationError("publication staged no artifacts at all")
    verified = tuple(_read_back(run_root, record) for record in staged)
    payload_hashes = dict(required_hashes)
    for record in verified:
        payload_hashes[record.path] = record.sha256
    manifest = FinalPublicationManifest(
        schema_version=1,
        run_id=run_id,
        evidence_root=str(run_id),
        selected_checkpoint=selected_checkpoint,
        final_files=verified,
        read_back_passed=True,
        payload_hashes=payload_hashes,
    )
    write_manifest(Path(run_root), FINAL_MANIFEST, manifest.model_dump(mode="json"))
    read_back_final_publication(run_root)
    return FINAL_MANIFEST


def write_run_complete(run_root: Path, payloads: dict) -> Path:
    """Publish COMPLETE.json last, binding every artifact the run depends on.

    The payloads mapping is a run-relative path to the exact bytes that path
    holds, and must cover the run identity, the log index, and every required
    checkpoint, evaluation, and report artifact. A payload the published final
    manifest does not certify is refused, so COMPLETE.json can never claim to
    bind something the publication step never recorded.
    """
    destination = Path(run_root)
    manifest = read_back_final_publication(destination)
    if not manifest.read_back_passed:
        raise PublicationError("final publication was not read back")
    if not isinstance(payloads, dict) or not payloads:
        raise PublicationError("a completion marker needs the artifacts it binds")
    required = {}
    for name, payload in payloads.items():
        published = destination / name
        if not published.is_file():
            raise PublicationError("bound payload is missing on disk: " + str(name))
        required[name] = payload
    return write_completion_marker(
        destination, required, FINAL_MANIFEST, certified=manifest.payload_hashes,
    )
