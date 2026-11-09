from pydantic import Field
"""Final publication manifest: immutable bundle identity."""
from typing import ClassVar, Mapping

from slm_train.contracts.base import SemanticManifest


class PublishedFile(SemanticManifest):
    schema_version: int = Field(default=1, ge=1)
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()
    path: str
    size_bytes: int
    sha256: str


class FinalPublicationManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str
    evidence_root: str
    selected_checkpoint: str
    final_files: tuple[PublishedFile, ...]
    read_back_passed: bool
    payload_hashes: Mapping[str, str] = {}
