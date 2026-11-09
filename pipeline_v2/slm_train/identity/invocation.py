"""Canonical invocation identity: command, working directory, allowlisted env."""
import re
from typing import ClassVar, Mapping

from slm_train.contracts.base import SemanticManifest
from slm_train.contracts.validation import require_repository_relative

_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _allowlisted_environment(environment: Mapping[str, str]) -> "dict[str, str]":
    """Allow only syntactically delimited names that reveal no secrets."""
    allowlisted: "dict[str, str]" = {}
    for name, value in environment.items():
        if not _ENV_NAME.fullmatch(name):
            continue
        if any(marker in name for marker in _SECRET_MARKERS):
            continue
        allowlisted[name] = value
    return allowlisted


def _validate_working_directory(working_directory: str) -> str:
    """Require an exact, clean, repository-relative working directory:
    no NUL byte, no "." or ".." component, no empty or trailing component."""
    if not working_directory or not working_directory.strip():
        raise ValueError(
            "working_directory must be repository-relative and inside the repository")
    if "\x00" in working_directory:
        raise ValueError(
            f"working_directory must not contain a NUL byte: {working_directory!r}")
    require_repository_relative(working_directory)
    components = re.split(r"[\\/]", working_directory)
    if any(part in (".", "..") for part in components):
        raise ValueError(
            f"working_directory must not contain '.' or '..' components: {working_directory!r}")
    if not all(components):
        raise ValueError(
            f"working_directory must not contain an empty or trailing separator: {working_directory!r}")
    return "/".join(components) if "\\" in working_directory else working_directory


class InvocationIdentity(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    exact_command: "tuple[str, ...]"
    working_directory: str
    environment: Mapping[str, str] = {}

    def invocation_hash(self) -> str:
        return self.semantic_manifest_hash()


def invocation_identity(
    command: "tuple[str, ...] | list[str]",
    working_directory: str,
    environment: Mapping[str, str] | None = None,
) -> InvocationIdentity:
    return InvocationIdentity(
        schema_version=1,
        exact_command=tuple(command),
        working_directory=_validate_working_directory(working_directory),
        environment=_allowlisted_environment(dict(environment or {})),
    )
