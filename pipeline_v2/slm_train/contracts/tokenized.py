"""Tokenized-data manifest (immutable)."""
from typing import ClassVar, Optional

from pydantic import Field, model_validator

from slm_train.contracts.base import ContractModel, SemanticManifest
from slm_train.contracts.validation import require_repository_relative


class TokenizedShard(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @model_validator(mode="after")
    def _relative_path(self) -> "TokenizedShard":
        require_repository_relative(self.path)
        return self


class TokenizedDataManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    shards: tuple[TokenizedShard, ...]
    mask_path: Optional[str] = None
    sequence_length: int = Field(gt=0)
    packing_mode: str = "none"
    usable_blocks: int = Field(default=0, ge=0)
    dropped_tail_tokens: int = Field(default=0, ge=0)
    overflow_blocks: int = Field(default=0, ge=0)
    token_range: Optional[tuple[int, int]] = None
    document_boundaries_path: Optional[str] = None
    source_data_hash: str

    @model_validator(mode="after")
    def _mask_required_for_document_masked(self) -> "TokenizedDataManifest":
        if self.packing_mode == "document_masked":
            if not self.mask_path:
                raise ValueError("mask_path is required when packing_mode is document_masked")
            require_repository_relative(self.mask_path)
        return self
