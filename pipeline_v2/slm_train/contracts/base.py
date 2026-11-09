"""Base contract envelopes."""
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from slm_train.contracts.hash import canonical_sha256

ValidationLabel = Literal["PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"]


class ContractModel(BaseModel):
    """Strict, frozen, versioned envelope."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: int = Field(ge=1)

    def semantic_digest(self, omit: "tuple[str, ...]" = ()) -> str:
        """Canonical hash of the full contract with named fields omitted."""
        payload = self.model_dump()
        for key in omit:
            payload.pop(key, None)
        return canonical_sha256(payload)


class SemanticManifest(ContractModel):
    """Immutable manifest: hashes exclude mutable verification state."""

    #: subclasses may name fields excluded from the semantic identity
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    def semantic_manifest_hash(self) -> str:
        return self.semantic_digest(omit=self.MUTABLE_FIELDS)
