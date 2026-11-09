"""Exact source identity: local Merkle file-set manifest or pinned HF repo.

SourceIdentity is the contract shape (contracts/source.py SourceManifest)
specialized for identity computation.
"""
from pathlib import Path
from typing import Iterable, Mapping

from slm_train.contracts.hash import sha256_bytes, sha256_file
from slm_train.contracts.source import SourceManifest


class SourceIdentity(SourceManifest):
    pass


def _merkle_root(digests: "dict[str, str]") -> str:
    return sha256_bytes(
        "".join(f"{path}\x00{digest}\n" for path, digest in sorted(digests.items())).encode("utf-8"))


def local_merkle_manifest(root: Path, files: Iterable[str]) -> SourceIdentity:
    """Hash an exact file set under root; every listed file must exist."""
    digests: "dict[str, str]" = {}
    for rel in files:
        path = Path(root) / rel
        if not path.is_file():
            raise KeyError(f"manifest input is missing: {rel}")
        digests[rel.replace("\\", "/")] = sha256_file(path)
    return SourceIdentity(
        schema_version=1, kind="local_merkle", root_hash=_merkle_root(digests),
        file_digests=digests)


def hf_source_identity(
    repo_id: str,
    revision: str,
    file_sha256: Mapping[str, str],
) -> SourceIdentity:
    """Pinned HF repository revision plus exact file hashes (no network here)."""
    if not repo_id or not revision:
        raise ValueError("HF source identity requires repo_id and a pinned revision")
    digests = dict(file_sha256)
    return SourceIdentity(
        schema_version=1, kind="hf", repo_id=repo_id, revision=revision,
        root_hash=_merkle_root(digests), file_digests=digests,
    )
