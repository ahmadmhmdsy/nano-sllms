"""Checkpoint manifest: exact-resume hashes for every state file."""
from typing import ClassVar, Literal, Mapping, Optional

from pydantic import Field

from slm_train.contracts.base import ContractModel, SemanticManifest


class ResumeFile(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str
    role: str = "model"


class CheckpointManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = ""
    config_hash: str
    data_hash: str = ""
    tokenizer_hash: str
    tokenized_hash: str = ""
    model_hash: str
    evaluator_hash: str = ""
    code_hash: str
    dependency_hash: str = ""
    global_step: int = Field(ge=0)
    epoch: Optional[float] = None
    model_state_identity: str = ""
    optimizer_state_identity: str = ""
    scheduler_state_identity: str = ""
    scaler_state_identity: str = ""
    rng_state_identity: str = ""
    sampler_state_identity: str = ""
    files: tuple[ResumeFile, ...]
    complete: bool
    evaluation_state: str = ""
    selection_state: str = ""
