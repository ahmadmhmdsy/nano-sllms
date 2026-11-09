"""Mutable operational contracts (never part of semantic identity)."""
from pydantic import ConfigDict, Field

from slm_train.contracts.base import ContractModel, ValidationLabel


class MutableContractModel(ContractModel):
    """Operational state envelope: mutable, still strict and versioned."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=False)


class StatusContract(MutableContractModel):
    label: ValidationLabel


class FreshnessContract(MutableContractModel):
    checked_at: str = ""
    fresh: bool = False


class LeaseContract(MutableContractModel):
    holder: str = ""
    expires_at: str = ""


class HeartbeatContract(MutableContractModel):
    last_beat_at: str = ""
    beat_count: int = Field(default=0, ge=0)


class RecoveryContract(MutableContractModel):
    note: str = ""
