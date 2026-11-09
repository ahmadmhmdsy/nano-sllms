"""Bootstrap identity, computed before any GPU work.

The bootstrap identity answers "what was this run asked to do" and is computed
from configuration, source, tokenizer, launch profile, requested runtime, code
dependencies, and command. It deliberately excludes the attempt: the attempt id
changes on every launch, while the bootstrap identity must be stable across
retries of the same work so an exact re-run can be recognised.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from slm_train.config.canonical import semantic_hash

HASH_FIELDS: tuple[str, ...] = (
    "config_hash",
    "source_identity_hash",
    "tokenizer_identity_hash",
    "launch_profile_hash",
    "requested_runtime_hash",
    "code_dependency_hash",
    "command_hash",
)


@dataclass(frozen=True)
class BootstrapIdentity:
    """What was requested, before anything was measured on hardware."""

    logical_experiment: str
    config_hash: str
    source_identity_hash: str
    tokenizer_identity_hash: str
    launch_profile_hash: str
    requested_runtime_hash: str
    code_dependency_hash: str
    command_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_experiment, str) or not self.logical_experiment.strip():
            raise ValueError("logical_experiment must be a non-empty string")
        for name in HASH_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(name + " must be a 64-character digest")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(name + " must be a lowercase hex digest")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        """A pure function of the request. The attempt id is not part of it."""
        return semantic_hash(self.as_dict())


def bootstrap_attempt_id(
    bootstrap_id: str, launch_nonce: str, pid: int, process_start_time: float
) -> str:
    """A stable id for one launch attempt of one bootstrap identity.

    Pure in its four inputs: no clock, no randomness, no os.getpid, so the same
    launch always yields the same attempt id and a retry is distinguishable.
    """
    if not isinstance(bootstrap_id, str) or not bootstrap_id.strip():
        raise ValueError("bootstrap_id must be a non-empty string")
    if not isinstance(launch_nonce, str) or not launch_nonce.strip():
        raise ValueError("launch_nonce must be a non-empty string")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if not isinstance(process_start_time, (int, float)) or isinstance(process_start_time, bool):
        raise ValueError("process_start_time must be a number")
    return semantic_hash(
        {
            "bootstrap_id": bootstrap_id,
            "launch_nonce": launch_nonce,
            "pid": pid,
            "process_start_time": float(process_start_time),
        }
    )[:32]

