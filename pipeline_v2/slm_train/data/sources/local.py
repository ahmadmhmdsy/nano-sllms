"""Local pinned source reader: exact file set, Merkle manifest, revalidation.

from_path rejects an unmanifested path outright, from_file_set may build a
manifest only from the exact listed files, and revalidation always re-hashes
every pinned file from disk. Verification time and freshness status are
mutable evidence; the semantic fingerprint never includes them.
"""
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from slm_train.contracts.hash import canonical_sha256, sha256_file
from slm_train.contracts.validation import require_repository_relative
from slm_train.data.sources.base import (
    FreshnessError,
    require_declared_fields,
    OptionalSourceDecision,
    SourceRole,
    SourceValidationError,
    freshness_label,
    require_role,
)
from slm_train.identity.source import SourceIdentity, local_merkle_manifest
from slm_train.data.provenance import verify_published_generation


def _validate_published_generation(published, reader) -> None:
    verify_published_generation(
        published, reader.source_id(), reader.semantic_fingerprint())


def _common_root(files) -> Path:
    return Path(os.path.commonpath([str(path.parent) for path in files]))


@dataclass(frozen=True)
class LocalSourceReader:
    """A source pinned by an exact file-set Merkle manifest."""

    root: Path
    files: tuple
    role: SourceRole
    license_note: str = ""
    provenance_note: str = ""
    required: bool = True
    absent_source: Optional[OptionalSourceDecision] = None
    freshness_expires_after: Optional[str] = None
    recorded_manifest: Optional[SourceIdentity] = None
    declared_source_id: str = field(kw_only=True)
    expected_row_budget: int = field(kw_only=True)
    expected_token_budget: int = field(kw_only=True)
    manifest: SourceIdentity = field(init=False)
    file_sizes: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        # Shared construction gate: EVERY constructor enforces the explicit
        # role vocabulary and the explicit optional-source decision contract.
        require_role(self.role)
        require_declared_fields(self)
        if not self.files:
            raise ValueError("a local source requires an explicit nonempty file set")
        rels = [str(Path(path).relative_to(self.root)).replace(os.sep, "/") for path in self.files]
        for rel in rels:
            require_repository_relative(rel)
        if self.recorded_manifest is None:
            for rel, path in zip(rels, self.files):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"manifest input is missing: {rel}")
            object.__setattr__(self, "manifest", local_merkle_manifest(self.root, rels))
        elif isinstance(self.recorded_manifest, dict):
            object.__setattr__(self, "manifest", SourceIdentity.model_validate(self.recorded_manifest))
        else:
            object.__setattr__(self, "manifest", self.recorded_manifest)
        sizes = {}
        for rel, path in zip(rels, self.files):
            path = Path(path)
            if path.is_file():
                sizes[rel] = path.stat().st_size
        object.__setattr__(self, "file_sizes", sizes)
        if self.absent_source is not None and self.required:
            raise ValueError("an absent-content decision is only valid for an optional source")
        if not self.required and self.absent_source is None:
            raise ValueError(
                "a source cannot become optional implicitly: an optional source "
                "requires an explicit profile decision, an explicit absent-content "
                "decision, and a recorded reason")
        # Fail closed when a recorded manifest does not pin the exact declared file set.
        if self.recorded_manifest is not None:
            digest_map = self.manifest.file_digests
            for rel in rels:
                if rel not in digest_map:
                    raise ValueError(
                        "recorded manifest does not pin the declared exact "
                        "file set: " + rel)

    @classmethod
    def from_path(
        cls,
        path,
        manifest: Optional[SourceIdentity] = None,
        *,
        role: Optional[SourceRole] = None,
        declared_source_id: str = "",
        expected_row_budget: Optional[int] = None,
        expected_token_budget: Optional[int] = None,
        license_note: str = "",
        provenance_note: str = "",
        required: bool = True,
        absent_source: Optional[OptionalSourceDecision] = None,
        freshness_expires_after: Optional[str] = None,
    ) -> "LocalSourceReader":
        """Adopt a recorded manifest for one exact file.

        An unmanifested path is never accepted: without a recorded manifest
        there is no pinned identity, so the caller must record one first.
        """
        if manifest is None:
            raise ValueError(
                "an unmanifested path is not a pinned source: record a manifest "
                "first (from_file_set) and pass it explicitly")
        if role is None:
            raise ValueError(
                "source role must be declared explicitly; roles are never "
                "inferred from filenames")
        return cls(
            root=Path(path).parent,
            files=(Path(path),),
            role=role,
            license_note=license_note,
            provenance_note=provenance_note,
            required=required,
            absent_source=absent_source,
            freshness_expires_after=freshness_expires_after,
            recorded_manifest=manifest,
            declared_source_id=declared_source_id,
            expected_row_budget=expected_row_budget,
            expected_token_budget=expected_token_budget,
        )

    @classmethod
    def from_file_set(
        cls,
        files,
        role: SourceRole,
        manifest: Optional[SourceIdentity] = None,
        *,
        declared_source_id: str,
        expected_row_budget: int,
        expected_token_budget: int,
        license_note: str = "",
        provenance_note: str = "",
        required: bool = True,
        absent_source: Optional[OptionalSourceDecision] = None,
        freshness_expires_after: Optional[str] = None,
    ) -> "LocalSourceReader":
        """Pin an exact, explicitly declared file set under one root."""
        require_role(role)
        listed = [Path(value) for value in files]
        if not listed:
            raise SourceValidationError("an empty required file set cannot define a source: FAIL")
        if not required and absent_source is None:
            raise ValueError(
                "a source cannot become optional implicitly: an optional source "
                "requires an explicit profile decision, an explicit absent-content "
                "decision, and a recorded reason")
        if absent_source is not None and required:
            raise ValueError("an absent-content decision is only valid for an optional source")
        instance = cls(
            root=_common_root(listed),
            files=tuple(listed),
            role=role,
            license_note=license_note,
            provenance_note=provenance_note,
            required=required,
            absent_source=absent_source,
            freshness_expires_after=freshness_expires_after,
            recorded_manifest=manifest,
            declared_source_id=declared_source_id,
            expected_row_budget=expected_row_budget,
            expected_token_budget=expected_token_budget,
        )
        return instance

    def source_id(self) -> str:
        """The DECLARED stable source id, independent of current file content."""
        return self.declared_source_id

    def identity(self) -> Mapping[str, object]:
        return self.manifest.model_dump()

    def provenance(self) -> Mapping[str, object]:
        from slm_train.data.provenance import provenance_record
        return provenance_record(self)

    def relative_files(self):
        rels = [str(Path(path).relative_to(self.root)).replace(os.sep, "/") for path in self.files]
        return tuple(zip(rels, self.files))

    def semantic_fingerprint(self) -> str:
        return canonical_sha256({
            "kind": self.manifest.kind,
            "merkle_root": self.manifest.root_hash,
            "repo_id": self.manifest.repo_id,
            "revision": self.manifest.revision,
            "role": self.role,
        })

    def revalidate(self, *, published_generation=None) -> Mapping[str, object]:
        """Reopen and re-hash every pinned file; evaluate freshness fail-closed.

        The verification time and freshness status are mutable evidence in the
        returned report; the semantic fingerprint never includes them. A failed
        revalidation raises; a BLOCKED freshness policy raises FreshnessError.
        When published_generation is declared, the completion marker and the
        manifest/schema identity are reopened and validated alongside the
        content checks (design 9.1), not taken on trust.
        """
        # Freshness policy is evaluated before content verification: a
        # changed-then-expired source surfaces BLOCKED, not a content FAIL.
        freshness = freshness_label(self.freshness_expires_after)
        if freshness == "BLOCKED":
            raise FreshnessError(
                f"freshness policy expired for source {self.source_id()}: BLOCKED")
        digests = {}
        sizes = {}
        for relative, path in self.relative_files():
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"pinned source file is absent: {relative} ({self.source_id()})")
            digests[relative] = sha256_file(path)
            sizes[relative] = Path(path).stat().st_size
        if digests != dict(self.manifest.file_digests):
            raise SourceValidationError(
                f"pinned source file changed after pinning: {self.source_id()}")
        if self.file_sizes and sizes != self.file_sizes:
            raise SourceValidationError(f"pinned source file size changed: {self.source_id()}")
        if published_generation is not None:
            _validate_published_generation(Path(published_generation), self)
        return {
            "source_id": self.source_id(),
            "semantic_fingerprint": self.semantic_fingerprint(),
            "label": "PASS",
            "freshness": freshness,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "file_digests": digests,
            "file_sizes": sizes,
            "required": self.required,
        }

    def iter_rows(self) -> Iterator[Mapping[str, object]]:
        self.revalidate()
        index = 0
        for relative, path in self.relative_files():
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"source file {relative}: row {line_index} is unreadable ({error})") from error
                    if not isinstance(row, dict):
                        raise ValueError(f"source file {relative}: row {line_index} is not an object")
                    record = dict(row)
                    record["_row_index"] = index
                    record["_source_file"] = relative
                    index += 1
                    yield record
        if index == 0:
            raise FileNotFoundError(f"required source set is empty: {self.source_id()}")
