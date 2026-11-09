"""Source contract shape: local Merkle file set or pinned HF revision.

This is the single source-of-truth shape for source identity; the identity
layer subclasses it (identity/source.py SourceIdentity).
"""
from typing import Mapping, Optional

from slm_train.contracts.base import SemanticManifest


class SourceManifest(SemanticManifest):
    kind: str  # "local_merkle", "hf", or "code"
    root_hash: str = ""
    file_digests: Mapping[str, str] = {}
    repo_id: str = ""
    revision: str = ""
    file_digests_path: Optional[str] = None
