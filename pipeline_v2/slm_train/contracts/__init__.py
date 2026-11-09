"""Versioned V2 contracts: strict, frozen immutable envelopes."""
from slm_train.contracts.base import ContractModel, SemanticManifest, ValidationLabel
from slm_train.contracts.status import (
    FreshnessContract,
    HeartbeatContract,
    LeaseContract,
    MutableContractModel,
    RecoveryContract,
    StatusContract,
)
from slm_train.contracts.log_index import KindLogIndex, LogIndexContract
from slm_train.contracts.source import SourceManifest
from slm_train.contracts.run import RunIdentityManifest
from slm_train.contracts.data import DataManifest
from slm_train.contracts.tokenizer import TokenizerManifest
from slm_train.contracts.tokenized import TokenizedDataManifest
from slm_train.contracts.model import ModelManifest
from slm_train.contracts.evaluator import EvaluatorManifest
from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.contracts.evaluation import EvaluationReport
from slm_train.contracts.final import FinalPublicationManifest

VALIDATION_LABELS = ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION")

__all__ = [
    "ContractModel", "SemanticManifest", "ValidationLabel", "VALIDATION_LABELS",
    "StatusContract", "FreshnessContract", "LeaseContract", "HeartbeatContract",
    "RecoveryContract", "MutableContractModel", "KindLogIndex", "LogIndexContract", "RunIdentityManifest", "DataManifest",
    "TokenizerManifest", "TokenizedDataManifest", "ModelManifest", "EvaluatorManifest",
    "CheckpointManifest", "EvaluationReport", "FinalPublicationManifest",
]
