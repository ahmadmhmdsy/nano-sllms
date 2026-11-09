"""Tokenizer manifest (immutable)."""
from typing import ClassVar, Optional

from pydantic import Field

from slm_train.contracts.base import SemanticManifest


class TokenizerManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    repository_identity: str
    file_identity: str = ""
    content_hash: str
    vocabulary_size: int = Field(gt=0)
    max_token_id: int = Field(ge=0)
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    special_tokens: tuple[str, ...] = ()
    context_behavior: str = "truncate"
    offset_behavior: str = "none"
    package_version: str
