"""Source protocol, explicit role vocabulary, and optional-source decisions.

A source is valid only with a pinned immutable identity (an exact local
file-set Merkle manifest or a pinned HF repository revision plus exact file
hashes). A role is declared explicitly; it is never inferred from a filename
or from a file merely existing inside a directory. A source cannot become
optional implicitly: an optional source requires an explicit profile
decision, an explicit absent-content decision, and a recorded reason.
"""
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Literal, Optional, Protocol

from pydantic import ConfigDict, Field

from slm_train.contracts.base import ContractModel, ValidationLabel

SourceRole = Literal["train", "validation", "test", "external-gate"]
VALID_ROLES: "tuple[str, ...]" = ("train", "validation", "test", "external-gate")


def require_role(role: object) -> str:
    """Fail closed unless the role is declared explicitly."""
    if not isinstance(role, str) or role not in VALID_ROLES:
        raise ValueError(
            f"source role must be declared explicitly as one of {VALID_ROLES}; "
            f"got {role!r} (roles are never inferred from filenames)")
    return role


def freshness_label(expires_after: Optional[str]) -> str:
    """Evaluate a freshness deadline fail-closed: expired is BLOCKED."""
    if not expires_after:
        return "PASS"
    try:
        deadline = datetime.fromisoformat(expires_after.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"freshness deadline is not an ISO-8601 instant: {error}") from error
    if datetime.now(timezone.utc) >= deadline:
        return "BLOCKED"
    return "PASS"


def require_declared_fields(reader) -> None:
    """Fail closed unless the DECLARED source identity fields are recorded.

    Every source declaration records a stable, content-independent source id
    plus its expected row and token budgets (design 9.1); an absent budget is
    rejected, never defaulted.
    """
    declared = getattr(reader, "declared_source_id", None)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(
            "a stable declared source_id must be recorded explicitly "
            f"(not derived from file content): got {declared!r}")
    for name in ("expected_row_budget", "expected_token_budget"):
        value = getattr(reader, name, None)
        if value is None:
            raise ValueError(
                f"{name} must be declared explicitly and is never defaulted: "
                "an absent budget is not a source declaration")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer: got {value!r}")


class OptionalSourceDecision(ContractModel):
    """An explicit, reasoned absent-content decision (never implicit)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = 1
    absent_content: Literal["expected", "needs_user_decision"] = "expected"
    reason: str = Field(min_length=1)


class SourceError(ValueError):
    """A fail-closed source error carrying a validation label from the vocabulary."""

    def __init__(self, message: str, label: ValidationLabel = "FAIL") -> None:
        super().__init__(message)
        self.label: ValidationLabel = label


SourceValidationError = SourceError


class FreshnessError(SourceError):
    """Freshness policy expired: the consuming stage is BLOCKED."""

    def __init__(self, message: str) -> None:
        super().__init__(message, label="BLOCKED")


class SourceReader(Protocol):
    """Contract every pinned source reader satisfies."""

    def source_id(self) -> str:
        raise NotImplementedError

    def identity(self) -> Mapping[str, object]:
        raise NotImplementedError

    def provenance(self) -> Mapping[str, object]:
        raise NotImplementedError

    def iter_rows(self) -> Iterator[Mapping[str, object]]:
        raise NotImplementedError

    def revalidate(self) -> Mapping[str, object]:
        raise NotImplementedError
