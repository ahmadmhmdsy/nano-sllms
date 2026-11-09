"""Source-controlled adapter registry (release 1).

The registry is a static Python declaration. There is no entry-point scan, no
directory walk, no remote lookup, and no dynamic import: an adapter exists here
only because someone wrote it down, and an unknown name fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Maturity = Literal["experimental", "validated", "retired"]
Release = Literal["r1"]

MATURE_VALUES = frozenset({"experimental", "validated", "retired"})
RELEASE_VALUES = frozenset({"r1"})


@dataclass(frozen=True)
class RegistryEntry:
    """One static adapter declaration; every field is required and validated."""

    name: str
    version: str
    schema: str
    stage: str
    maturity: Maturity
    compatibility: tuple[str, ...]
    release: Release
    test_contract: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "schema", "stage", "test_contract"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("RegistryEntry." + field_name + " must be a non-empty string")
        if self.maturity not in MATURE_VALUES:
            raise ValueError("RegistryEntry.maturity must be one of " + str(sorted(MATURE_VALUES)))
        if self.release not in RELEASE_VALUES:
            raise ValueError("RegistryEntry.release must be one of " + str(sorted(RELEASE_VALUES)))
        if not isinstance(self.compatibility, tuple) or not self.compatibility:
            raise ValueError("RegistryEntry.compatibility must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.compatibility):
            raise ValueError("RegistryEntry.compatibility entries must be non-empty strings")


_ENTRIES: tuple[RegistryEntry, ...] = (
    RegistryEntry(
        "local_source", "1", "SourceManifest", "prepare", "validated",
        ("r1",), "r1", "unit/test_data.py",
    ),
    RegistryEntry(
        "hf_source", "1", "SourceManifest", "prepare", "validated",
        ("r1",), "r1", "network/test_hf_source.py",
    ),
    RegistryEntry(
        "concat_packer", "1", "TokenizedDataManifest", "tokenize", "validated",
        ("r1",), "r1", "unit/test_data.py",
    ),
    RegistryEntry(
        "document_masked_packer", "1", "TokenizedDataManifest", "tokenize",
        "experimental", ("r1",), "r1", "unit/test_data.py",
    ),
    RegistryEntry(
        "llama_gqa", "1", "ModelManifest", "model_validation", "validated",
        ("r1",), "r1", "unit/test_model_factory.py",
    ),
    RegistryEntry(
        "hf_causal_lm", "1", "CheckpointManifest", "train", "validated",
        ("r1",), "r1", "unit/test_model_factory.py",
    ),
    RegistryEntry(
        "r1_evaluators", "1", "EvaluationReport", "internal_evaluation", "validated",
        ("r1",), "r1", "unit/test_evaluation.py",
    ),
    RegistryEntry(
        "r1_gates", "1", "GateResult", "external_gates", "validated",
        ("r1",), "r1", "unit/test_evaluation.py",
    ),
    RegistryEntry(
        "best_validation_loss", "1", "CheckpointManifest", "publish_final", "validated",
        ("r1",), "r1", "unit/test_evaluation.py",
    ),
    # The D-line evaluator. Declared experimental on purpose: it is reachable
    # through its own catalogue (evaluation/diacritizer_suites.py) and produces
    # a governed EvaluationReport, but it is not a member of the R1 pretrain
    # SUITES and no R1 stage graph resolves it, so it must not be labelled
    # validated alongside the suites that the R1 acceptance run exercises.
    RegistryEntry(
        "diacritizer_eval", "1", "EvaluationReport", "internal_evaluation", "experimental",
        ("r1",), "r1", "unit/test_diacritizer_evaluation.py",
    ),
)

_BY_NAME: dict[str, RegistryEntry] = {entry.name: entry for entry in _ENTRIES}

if len(_BY_NAME) != len(_ENTRIES):
    raise ValueError("adapter registry names must be unique")


def entries() -> tuple[RegistryEntry, ...]:
    """The static R1 adapter declarations, in declaration order."""
    return _ENTRIES


def get(name: str) -> RegistryEntry:
    """Resolve a declared adapter by name, or raise for anything undeclared."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("adapter name must be a non-empty string")
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError("unknown adapter: " + repr(name)) from None

