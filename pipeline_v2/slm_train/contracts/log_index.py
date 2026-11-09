"""Log-index contract shapes (mutable operational index, not semantic identity).

This is the single source of truth for the serialized log-manifest shape;
artifacts/log_index.py consumes it directly.
"""
from typing import Mapping

from pydantic import Field

from slm_train.contracts.status import MutableContractModel


class KindLogIndex(MutableContractModel):
    """Per-stream index state: its own generation, chain, and sequences."""

    schema_version: int = Field(default=1, ge=1)

    generation: int = 0
    previous_index_hash: str = ""
    segment_hashes: list[str] = []
    first_sequence: int = Field(default=-1, ge=-1)
    last_sequence: int = Field(default=-1, ge=-1)
    index_hash: str = ""


class LogIndexContract(MutableContractModel):
    schema_version: int = Field(default=1, ge=1)
    kinds: Mapping[str, KindLogIndex] = {}
