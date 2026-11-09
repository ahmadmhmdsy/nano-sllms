"""Pinned tokenizer identity and token-id validation.

A tokenizer is pinned either to an exact local file-set identity (Merkle
over the listed files) or to a pinned HF repository revision plus exact
file hashes. Nothing here performs network I/O or touches the hub.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from slm_train.contracts.hash import canonical_sha256, sha256_bytes
from slm_train.contracts.tokenizer import TokenizerManifest
from slm_train.errors import PipelineError
from slm_train.identity.source import SourceIdentity, hf_source_identity, local_merkle_manifest


class TokenizerError(PipelineError, ValueError):
    """Fail-closed tokenizer identity/validation failure, labeled FAIL."""

    def __init__(self, message: str, label: str = "FAIL") -> None:
        super().__init__(message)
        self.label = label


@dataclass(frozen=True)
class TokenizerIdentity:
    """Immutable pinned tokenizer identity plus its live record fields."""

    manifest: TokenizerManifest
    source: SourceIdentity
    record: "dict[str, object]"

    @property
    def label(self) -> str:
        return str(self.record.get("label", "PASS"))

    @property
    def context_length(self) -> Optional[int]:
        value = self.record.get("context_length")
        return int(value) if value is not None else None


def manifest_json(identity: TokenizerIdentity) -> "dict[str, object]":
    """Tokenizer identities publish their manifest and live record."""
    value = identity.manifest.model_dump()
    value["extra_record"] = dict(identity.record)
    return value


def tokenizer_from_record(record: "Mapping[str, object]") -> "tuple[TokenizerManifest, dict]":
    """Validate a published tokenizer record fail-closed: the extra record
    is separated (never an extra field of the contract) and the special
    token list is coerced to its tuple form before validation. Any schema
    failure is a labeled TokenizerError, never a raw pydantic error."""
    from pydantic import ValidationError
    value = dict(record)
    extra_record = value.pop("extra_record", None)
    value["special_tokens"] = tuple(value.get("special_tokens", ()) or ())
    try:
        manifest = TokenizerManifest.model_validate(value)
    except ValidationError as error:
        raise TokenizerError(
            f"the recorded tokenizer manifest is invalid: FAIL ({error})") from error
    return manifest, (dict(extra_record) if isinstance(extra_record, dict) else {})


def reopen_identity(manifest: TokenizerManifest, extra_record: "Optional[dict]" = None) -> TokenizerIdentity:
    """Reopen a tokenizer identity that carries its live record."""
    extra = dict(extra_record or {})
    nested = extra.get("extra_record")
    if isinstance(nested, dict):
        for key, value in nested.items():
            extra.setdefault(key, value)
    record = dict(extra)
    kind = "hf" if manifest.repository_identity.startswith("hf:") else "local_merkle"
    record["kind"] = kind
    if "revision" not in record and kind == "hf":
        # the pinned revision is part of the recorded file identity
        file_identity = str(manifest.file_identity)
        for part in file_identity.split(";"):
            if part.startswith("revision:"):
                record["revision"] = part.split(":", 1)[1]
    return TokenizerIdentity(
        manifest=manifest,
        source=SourceIdentity(
            schema_version=1,
            kind=kind,
            root_hash=manifest.file_identity,
            file_digests={"tokenizer": manifest.content_hash},
            repo_id=(manifest.repository_identity[len("hf:"):]
                     if manifest.repository_identity.startswith("hf:") else ""),
            revision=str(record.get("revision", "")),
        ),
        record=record,
    )


def _validated_specials(
    vocabulary_size: int,
    max_token_id: int,
    bos: Optional[int],
    eos: Optional[int],
    pad: Optional[int],
    special_tokens: Iterable[str],
) -> "dict[str, int]":
    if vocabulary_size <= 0:
        raise TokenizerError("the tokenizer vocabulary must be positive: FAIL")
    if max_token_id < 0 or max_token_id >= vocabulary_size:
        raise TokenizerError("the tokenizer maximum token id is out of range: FAIL")
    specials: "dict[str, int]" = {}
    for name, token_id in (("bos", bos), ("eos", eos), ("pad", pad)):
        if token_id is None:
            continue
        if not 0 <= int(token_id) <= max_token_id:
            raise TokenizerError(f"special token {name} id is out of range: FAIL")
        specials[name] = int(token_id)
    seen: "dict[int, str]" = {}
    for name, token_id in specials.items():
        if token_id in seen:
            raise TokenizerError(
                f"duplicate special token id for {seen[token_id]}/{name}: FAIL")
        seen[token_id] = name
    specials_map = dict(specials)
    for index, value in enumerate(special_tokens):
        if not str(value):
            raise TokenizerError("special-token names must be non-empty: FAIL")
        specials_map[f"special:{index}"] = -1 - index
    return specials_map


def _check_pinned(kind: str, repository_identity: str, revision: str) -> None:
    if kind == "hf":
        if not repository_identity or not revision:
            raise TokenizerError(
                "an HF tokenizer identity is unpinned: it requires the pinned "
                "repository and revision: FAIL")
    elif kind == "local_merkle":
        if not repository_identity:
            raise TokenizerError(
                "a local tokenizer identity is unpinned: it requires the pinned "
                "file identity: FAIL")
    else:
        raise TokenizerError(f"an unpinned tokenizer identity is FAIL: {kind!r}")


def _manifest(
    repository_identity: str,
    file_identity: str,
    content_hash: str,
    vocabulary_size: int,
    max_token_id: int,
    bos: Optional[int],
    eos: Optional[int],
    pad: Optional[int],
    special_tokens: Iterable[str],
    context_length: Optional[int],
    offset_behavior: str,
    package_version: str,
) -> TokenizerManifest:
    return TokenizerManifest(
        schema_version=1,
        
        repository_identity=repository_identity,
        file_identity=file_identity,
        content_hash=content_hash,
        vocabulary_size=int(vocabulary_size),
        max_token_id=int(max_token_id),
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        special_tokens=tuple(str(v) for v in special_tokens),
        context_behavior="truncate",
        offset_behavior=str(offset_behavior),
        package_version=str(package_version),
    )


def pin_local_tokenizer(
    files: "Iterable[Union[str, Path]]",
    *,
    vocabulary_size: int,
    max_token_id: int,
    package_version: str,
    bos: Optional[int] = None,
    eos: Optional[int] = None,
    pad: Optional[int] = None,
    special_tokens: Iterable[str] = (),
    context_length: Optional[int] = None,
    offset_behavior: str = "none",
    truncation_counts: "Optional[dict]" = None,
) -> TokenizerIdentity:
    """Pin a tokenizer to an exact local file set through its Merkle identity."""
    paths = [Path(value) for value in files]
    if not paths:
        raise TokenizerError("pinning a local tokenizer requires an exact non-empty file set")
    parent = paths[0].parent
    if any(path.parent != parent for path in paths):
        raise TokenizerError("pinned local tokenizer files must share one root")
    names = tuple(path.name for path in paths)
    specials = _validated_specials(
        int(vocabulary_size), int(max_token_id), bos, eos, pad, special_tokens)
    source = local_merkle_manifest(parent, names)
    manifest = _manifest(
        f"local:{parent.name}",
        f"merkle:{source.root_hash}",
        canonical_sha256(source.file_digests),
        int(vocabulary_size), int(max_token_id),
        bos, eos, pad, special_tokens, context_length, offset_behavior, package_version,
    )
    return TokenizerIdentity(
        manifest=manifest,
        source=source,
        record={
            "label": "PASS",
            "kind": "local_merkle",
            "name": str(names[0]),
            "files": ",".join(names),
            "root_hash": source.root_hash,
            "special_token_map": specials,
            "context_length": context_length,
            "truncation_counts": dict(truncation_counts or {}),
            "package_version": str(package_version),
        },
    )


def pin_hf_tokenizer(
    repo_id: str,
    revision: str,
    file_sha256: "Mapping[str, str]",
    *,
    vocabulary_size: int,
    max_token_id: int,
    package_version: str,
    bos: Optional[int] = None,
    eos: Optional[int] = None,
    pad: Optional[int] = None,
    special_tokens: Iterable[str] = (),
    context_length: Optional[int] = None,
    offset_behavior: str = "none",
    truncation_counts: "Optional[dict]" = None,
) -> TokenizerIdentity:
    """Pin an HF tokenizer to a pinned revision plus exact file hashes."""
    if not file_sha256:
        raise TokenizerError("an HF tokenizer pin requires exact file hashes: FAIL")
    if any(not str(v) for v in file_sha256.values()):
        raise TokenizerError("an HF tokenizer pin rejects empty file hashes: FAIL")
    specials = _validated_specials(
        int(vocabulary_size), int(max_token_id), bos, eos, pad, special_tokens)
    source = hf_source_identity(repo_id, revision, dict(file_sha256))
    manifest = _manifest(
        f"hf:{repo_id}", f"revision:{revision};merkle:{source.root_hash}",
        canonical_sha256(source.file_digests),
        int(vocabulary_size), int(max_token_id),
        bos, eos, pad, special_tokens, context_length, offset_behavior, package_version,
    )
    return TokenizerIdentity(
        manifest=manifest,
        source=source,
        record={
            "label": "PASS",
            "kind": "hf",
            "revision": str(revision),
            "root_hash": source.root_hash,
            "special_token_map": specials,
            "context_length": context_length,
            "truncation_counts": dict(truncation_counts or {}),
            "package_version": str(package_version),
        },
    )


def validate_token_ids(tokens: "Iterable[int]", identity: TokenizerIdentity) -> "dict[str, object]":
    """Every token id is in [0, max_token_id]; out-of-range ids are FAIL.

    Never silently truncated, replaced by padding, or wrapped around.
    """
    manifest = identity.manifest
    vocabulary_size = int(manifest.vocabulary_size)
    max_token_id = int(manifest.max_token_id)
    seen = 0
    highest = -1
    for value in tokens:
        token_id = int(value)
        if token_id < 0 or token_id >= vocabulary_size:
            raise TokenizerError(
                f"token id {token_id} is out of range for the recorded vocabulary: FAIL")
        if token_id > max_token_id:
            raise TokenizerError(
                f"token id {token_id} exceeds the recorded maximum token id: FAIL")
        seen += 1
        highest = max(highest, token_id)
    return {"label": "PASS", "validated_tokens": seen, "max_token_seen": max(highest, 0)}


def assert_unpinned(identity: Optional[TokenizerIdentity]) -> None:
    """Fail closed when a tokenizer is not exactly pinned."""
    if identity is None:
        raise TokenizerError("an unpinned tokenizer is FAIL")
    _check_pinned(
        str(identity.record.get("kind", "")),
        identity.manifest.repository_identity,
        str(identity.record.get("revision", "")),
    )
