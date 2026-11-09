"""Pinned Hugging Face repository source reader (declared, never inferred).

from_repository requires a nonempty pinned revision AND an explicit nonempty
file list, and never performs network I/O here. Exact file hashes come only
from a recorded SourceManifest (recorded offline or by the user-approved
opt-in network lane, which must use a pinned revision and no token).
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from slm_train.data.sources.base import (
    FreshnessError,
    require_declared_fields,
    OptionalSourceDecision,
    SourceRole,
    SourceValidationError,
    freshness_label,
    require_role,
)
from slm_train.identity.source import SourceIdentity


@dataclass(frozen=True)
class HFSourceReader:
    """A source pinned by an HF repository revision plus exact file hashes."""

    repository: str
    revision: str
    files: tuple
    role: SourceRole
    manifest: SourceIdentity | None = None
    license_note: str = ""
    provenance_note: str = ""
    required: bool = True
    absent_source: Optional[OptionalSourceDecision] = None
    freshness_expires_after: Optional[str] = None
    verified_revision: Optional[str] = None
    verified_file_hashes: Optional[Mapping] = None
    declared_source_id: str = field(kw_only=True)
    expected_row_budget: int = field(kw_only=True)
    expected_token_budget: int = field(kw_only=True)

    def __post_init__(self) -> None:
        """Shared construction gate for EVERY HF constructor path."""
        require_role(self.role)
        if not self.required and self.absent_source is None:
            raise ValueError(
                "a source cannot become optional implicitly: an optional source "
                "requires an explicit profile decision, an explicit absent-content "
                "decision, and a recorded reason")
        if self.absent_source is not None and self.required:
            raise ValueError("an absent-content decision is only valid for an optional source")
        require_declared_fields(self)
        if self.manifest is not None:
            declared = sorted(set(str(value) for value in self.files))
            pinned = sorted(str(value) for value in self.manifest.file_digests)
            if declared != pinned:
                raise ValueError(
                    "recorded manifest file set does not match the declared file "
                    f"set: FAIL (declared {declared}, manifest pins {pinned}) "
                    f"for {self.repository!r} @ {self.revision!r}")

    @classmethod
    def from_repository(
        cls,
        repository: str,
        revision: str,
        files,
        manifest: SourceIdentity | None = None,
        *,
        role: SourceRole,
        declared_source_id: str,
        expected_row_budget: int,
        expected_token_budget: int,
        license_note: str = "",
        provenance_note: str = "",
        required: bool = True,
        absent_source: Optional[OptionalSourceDecision] = None,
        freshness_expires_after: Optional[str] = None,
        verified_revision: Optional[str] = None,
        verified_file_hashes: Optional[Mapping] = None,
    ) -> "HFSourceReader":
        """Declare a pinned HF source; argument validation only, no network."""
        if revision is None or not str(revision).strip() or revision == "":
            raise SourceValidationError(
                "a HF source requires a nonempty pinned revision: got "
                f"{revision!r} (repository {repository!r})")
        if isinstance(revision, bytes):
            raise SourceValidationError("pinned revision must be a string: bytes are not accepted")
        listed = tuple(str(value).strip() for value in files)
        if not listed:
            raise SourceValidationError(
                "a HF source requires an explicit nonempty file list: got "
                f"{listed!r} (roles are inferred from declared files, never "
                "from a repository directory listing)")
        if any(not value for value in listed):
            raise SourceValidationError("HF file list contains empty file names")
        if repository is None or not str(repository).strip():
            raise SourceValidationError("HF source requires a repository id")
        if not required and absent_source is None:
            raise ValueError(
                "a source cannot become optional implicitly: an optional source "
                "requires an explicit profile decision, an explicit absent-content "
                "decision, and a recorded reason")
        if manifest is not None:
            if manifest.kind != "hf":
                raise ValueError("HF source requires an hf-kind SourceManifest")
            if manifest.repo_id != str(repository) or manifest.revision != str(revision):
                raise ValueError(
                    "recorded manifest repository/revision do not match the declaration")
            if sorted(manifest.file_digests) != sorted(set(str(value) for value in listed)):
                raise ValueError("Files in manifest do not match the declared file list: FAIL")
            if not manifest.root_hash:
                raise ValueError("recorded HF manifest lacks a pinned root hash")
        require_role(role)
        return cls(
            repository=str(repository),
            revision=str(revision),
            files=listed,
            role=role,
            manifest=manifest,
            license_note=license_note,
            provenance_note=provenance_note,
            required=required,
            absent_source=absent_source,
            freshness_expires_after=freshness_expires_after,
            verified_revision=verified_revision,
            verified_file_hashes=verified_file_hashes,
            declared_source_id=declared_source_id,
            expected_row_budget=expected_row_budget,
            expected_token_budget=expected_token_budget,
        )

    @staticmethod
    def collect_file_hashes(revision, repository, files) -> "dict[str, str]":
        """Opt-in network lane (user-approved only, pinned revision, no token).

        Performs live network I/O and is therefore excluded from the unit lane
        and from default agent runs; it persists nothing. Argument validation
        happens before any network import: an unpinned revision never reaches
        the hub and no hash is ever invented for an absent declared file.
        """
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                "collect_file_hashes requires an explicit nonempty pinned "
                f"revision: got {revision!r}")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError(f"collect_file_hashes requires a repository id: got {repository!r}")
        declared = tuple(str(value) for value in files)
        if not declared or any(not value for value in declared):
            raise ValueError("collect_file_hashes requires an explicit nonempty file list")
        from huggingface_hub import hf_hub_download, list_repo_files
        listed = list_repo_files(repository, revision=revision, token=False)
        hashes: "dict[str, str]" = {}
        for name in declared:
            if name not in set(listed):
                raise ValueError(f"declared file is absent from pinned revision: {name}")
            local = hf_hub_download(repository, name, revision=revision, token=False)
            from slm_train.contracts.hash import sha256_file
            hashes[name] = sha256_file(local)
        return {name: hashes[name] for name in sorted(hashes)}

    def source_id(self) -> str:
        """The DECLARED stable source id, independent of current file content."""
        return self.declared_source_id

    def identity(self) -> Mapping[str, object]:
        """pinned identity: exact repo revision plus every declared file hash."""
        if self.manifest is None:
            raise SourceValidationError(
                "an HF source without recorded exact file hashes is not pinned: "
                "record an hf-kind SourceManifest (offline) before consumption")
        return self.manifest.model_dump()

    def provenance(self) -> Mapping[str, object]:
        from slm_train.data.provenance import provenance_record
        return provenance_record(self)

    def semantic_fingerprint(self) -> str:
        return f"hf:{self.repository}:{self.revision}"

    def revalidate(self) -> Mapping[str, object]:
        """Fail-closed HF revalidation: never self-certify PASS offline.

        A pinned manifest alone is an identity record, not a verification:
        without an explicit verified evidence record (file hashes recorded
        against this exact pinned revision) the report is SKIPPED, never
        PASS. A recorded evidence hash that disagrees with the pinned
        manifest is rejected outright.
        """
        if self.manifest is None:
            raise SourceValidationError(
                "cannot revalidate an unpinned HF source: no recorded file hashes")
        freshness = freshness_label(self.freshness_expires_after)
        if freshness == "BLOCKED":
            raise FreshnessError(
                f"freshness policy expired for source {self.source_id()}: BLOCKED")
        skipped = {
            "source_id": self.source_id(),
            "label": "SKIPPED",
            "freshness": "SKIPPED",
            "semantic_fingerprint": self.manifest.root_hash,
            "required": self.required,
            "reason": "offline: no verification evidence recorded against the pinned revision",
        }
        if self.verified_revision != self.revision or self.verified_file_hashes is None:
            return skipped
        evidence = {str(key): str(value) for key, value in sorted(self.verified_file_hashes.items())}
        declared = sorted(set(str(value) for value in self.files))
        if sorted(evidence) != declared:
            raise SourceValidationError(
                "verification evidence does not match the declared file set "
                f"for {self.source_id()}: verified evidence {sorted(evidence)} "
                f"vs declared {declared}")
        recorded = dict(self.manifest.file_digests)
        if evidence != {key: str(value) for key, value in sorted(recorded.items())}:
            raise SourceValidationError(
                "recorded HF verification evidence disagrees with the pinned "
                f"manifest: {self.source_id()}")
        return {
            "source_id": self.source_id(),
            "label": "PASS",
            "freshness": freshness,
            "semantic_fingerprint": self.manifest.root_hash,
            "required": self.required,
            "verified_revision": self.revision,
            "file_digests": evidence,
        }
