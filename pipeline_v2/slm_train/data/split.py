"""Deterministic salted split, derived from generation and dataset identity.

The assigned split for one document is a pure function of
(dataset identity, generation identity, document id, split_salt); overlap is
always checked and any overlap is FAIL.
"""
import hashlib

from slm_train.errors import PipelineError
from slm_train.contracts.hash import canonical_json_bytes

TRAIN = "train"
VALIDATION = "validation"
# Sources whose declared role is test or external-gate can never enter the
# train or validation split: a held-out corpus a trainer never sees is the
# whole meaning of the role (A11). Their documents fall into this split.
HELD_OUT = "held-out"

HELD_OUT_ROLES = ("test", "external-gate")


def role_forces_held_out(role: object) -> bool:
    """True when a source role legally pins every document out of training."""
    return str(role) in HELD_OUT_ROLES


def split_hash(dataset_id: str, generation_id: str, doc_id: str, split_salt) -> str:
    payload = {
        "dataset_id": str(dataset_id),
        "generation_id": str(generation_id),
        "doc_id": str(doc_id),
        "split_salt": int(split_salt),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def assign_split(dataset_id: str, generation_id: str, doc_id: str, split_salt, validation_fraction) -> str:
    if not 0 <= float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    digest = split_hash(dataset_id, generation_id, doc_id, split_salt)
    window = int(digest[:12], 16) / 281474976710655.0
    return VALIDATION if window < float(validation_fraction) else TRAIN


def overlap_between(train_ids, validation_ids) -> "list[str]":
    common = set(train_ids) & set(validation_ids)
    common = sorted(common)
    return common


class SplitOverlapError(PipelineError):
    """A document assigned to both train and validation is FAIL."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.label = "FAIL"


def require_no_overlap(train_ids, validation_ids) -> None:
    """A document assigned to both train and validation is FAIL."""
    common = overlap_between(train_ids, validation_ids)
    if common:
        raise SplitOverlapError(
            f"overlapping train/validation assignment is FAIL: {len(common)} overlapping document ids")
