"""Run identity manifest (immutable, published once, never edited).

Mutable stage state, lease state, heartbeat, and recovery text live in the
mutable status contract, never here. The semantic hash excludes only
created_at.
"""
from typing import ClassVar, Mapping, Optional

from pydantic import Field, field_validator

from slm_train.contracts.base import SemanticManifest
from slm_train.contracts.validation import require_repository_relative


def validate_evidence_root(value: str) -> str:
    return require_repository_relative(value)


class PromotionPredecessorIdentity(SemanticManifest):
    schema_version: int = Field(default=1, ge=1)
    """Identity of the run/checkpoint this one is promoted from."""

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    predecessor_run_id: str = ""
    predecessor_checkpoint_identity: str = ""
    predecessor_data_hash: str = ""
    predecessor_evaluator_hash: str = ""
    predecessor_gate_set: str = ""


class RunIdentityManifest(SemanticManifest):
    """Immutable semantic run identity."""

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ("created_at",)

    run_id: str
    experiment_id: str
    release: str = "r1"
    config_path: str
    config_hash: str
    data_hash: str = ""
    tokenizer_hash: str = ""
    tokenized_hash: str = ""
    model_hash: str = ""
    evaluator_hash: str = ""
    code_hash: str
    dependency_hash: str = ""
    git_commit: Optional[str] = None
    git_worktree_state: Optional[Mapping[str, str]] = None
    python_identity: Optional[Mapping[str, str]] = None
    package_identity: Optional[Mapping[str, str]] = None
    device_identity: Optional[Mapping[str, str]] = None
    driver_identity: Optional[Mapping[str, str]] = None
    runtime_fingerprint: str = ""
    working_directory: str
    launch_profile: str = ""
    exact_command: tuple[str, ...] = ()
    created_at: Optional[str] = None
    parent_model_identity: Optional[str] = None
    parent_checkpoint_identity: Optional[str] = None
    promotion_predecessor: Optional[PromotionPredecessorIdentity] = None
    evidence_root: str

    @field_validator("evidence_root")
    @classmethod
    def _evidence_root_must_be_relative(cls, value: str) -> str:
        return validate_evidence_root(value)

