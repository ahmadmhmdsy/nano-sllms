"""Source-controlled feature registry and the R1 capability gate.

Release 1 carries exactly one feature capability: pretrain. Every R2 capability
(sft, lora, kd, gdn, mounting) is out of scope and is refused here rather than
left for a later stage to discover. The registry is static: no entry points, no
directory scan, no remote lookup, no dynamic import.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from slm_train.config.models import R1_CAPABILITIES, R2_CAPABILITIES

R1_RELEASE = "r1"


@dataclass(frozen=True)
class FeatureEntry:
    """One static feature declaration; every field is required and validated."""

    name: str
    version: str
    release: str
    capabilities: tuple[str, ...]
    protocol: str
    maturity: str
    test_contract: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "release", "protocol", "test_contract"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("FeatureEntry." + field_name + " must be a non-empty string")
        if self.release != R1_RELEASE:
            raise ValueError("FeatureEntry.release must be r1 in this registry")
        if self.maturity not in {"experimental", "validated", "retired"}:
            raise ValueError("FeatureEntry.maturity is not a known maturity")
        if not isinstance(self.capabilities, tuple) or not self.capabilities:
            raise ValueError("FeatureEntry.capabilities must be a non-empty tuple")
        for capability in self.capabilities:
            if not isinstance(capability, str) or not capability.strip():
                raise ValueError("capability entries must be non-empty strings")
            if capability not in R1_CAPABILITIES:
                raise ValueError("capability is outside release R1: " + repr(capability))


_ENTRIES: tuple[FeatureEntry, ...] = (
    FeatureEntry(
        name="r1_pretrain_loop",
        version="1",
        release=R1_RELEASE,
        capabilities=("pretrain",),
        protocol="SelectionPolicy",
        maturity="validated",
        test_contract="unit/test_model_factory.py",
    ),
    FeatureEntry(
        name="r1_health_monitor",
        version="1",
        release=R1_RELEASE,
        capabilities=("pretrain",),
        protocol="HealthMonitor",
        maturity="experimental",
        test_contract="unit/test_evaluation.py",
    ),
)

_BY_NAME: dict[str, FeatureEntry] = {entry.name: entry for entry in _ENTRIES}

#: release-2 features are declared here as an explicitly empty boundary
_BY_NAME_R2: dict[str, FeatureEntry] = {}

if len(_BY_NAME) != len(_ENTRIES):
    raise ValueError("feature registry names must be unique")


def entries() -> tuple[FeatureEntry, ...]:
    """The static R1 feature declarations, in declaration order."""
    return _ENTRIES


def get(name: str) -> FeatureEntry:
    """Resolve a declared feature by name, or raise for anything undeclared."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("feature name must be a non-empty string")
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError("unknown feature: " + repr(name)) from None


def r2_boundary() -> tuple[str, ...]:
    """Declared release-2 features. Deliberately empty: R2 is not implemented."""
    return tuple(sorted(_BY_NAME_R2))


def r2_capabilities() -> tuple[str, ...]:
    """Every R2 capability, listed so the boundary is explicit rather than absent."""
    return tuple(sorted(R2_CAPABILITIES))


def validate_capabilities(release: str, capabilities: Sequence[str]) -> None:
    """Fail closed unless the release is r1 and the set is exactly the pretrain capability."""
    if not isinstance(release, str) or release != R1_RELEASE:
        raise ValueError("Release 1 only accepts release=" + repr(R1_RELEASE))
    if isinstance(capabilities, (str, bytes)):
        raise ValueError("R1 requires exactly the pretrain capability")
    try:
        requested = tuple(capabilities)
    except TypeError:
        raise ValueError("R1 requires exactly the pretrain capability") from None
    if len(requested) != 1 or requested[0] not in R1_CAPABILITIES:
        refused = sorted({
            item for item in requested
            if isinstance(item, str) and item in R2_CAPABILITIES
        })
        detail = "; out of scope for R1: " + ", ".join(refused) if refused else ""
        raise ValueError("R1 requires exactly the pretrain capability" + detail)
