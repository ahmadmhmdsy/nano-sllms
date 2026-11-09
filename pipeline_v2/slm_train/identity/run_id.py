"""Final run IDs: safe experiment id + semantic input fingerprint."""
from dataclasses import asdict, dataclass
from typing import Literal
import re

from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256, sha256_bytes


@dataclass(frozen=True)
class InputFingerprint:
    config_hash: str
    data_hash: str
    tokenizer_hash: str
    tokenized_hash: str
    model_hash: str
    evaluator_hash: str
    code_hash: str
    release: Literal["r1"]
    launch_profile_hash: str
    runtime_hash: str
    invocation_hash: str

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


SAFE_RUN_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def final_run_id(experiment_id: str, fingerprint: InputFingerprint) -> str:
    if not SAFE_RUN_COMPONENT.fullmatch(experiment_id):
        raise ValueError("experiment_id is not a safe run-id component")
    return experiment_id + "-" + fingerprint.digest()


def bootstrap_run_id(experiment_id: str, bootstrap_sequence: int) -> str:
    """Operational bootstrap identity: never part of final identity."""
    if not SAFE_RUN_COMPONENT.fullmatch(experiment_id):
        raise ValueError("experiment_id is not a safe run-id component")
    return f"{experiment_id}-bootstrap-{int(bootstrap_sequence)}"


def attempt_id(prefix: str, launch_nonce: str, pid: int, process_start_time: float) -> str:
    """Operational attempt id (design 10.3): derived from the launch nonce,
    PID, and process start time. Pure and deterministic; excluded from the
    final run identity."""
    if not SAFE_RUN_COMPONENT.fullmatch(prefix):
        raise ValueError("attempt prefix is not a safe run-id component")
    if not str(launch_nonce).strip() or int(pid) <= 0 or float(process_start_time) < 0:
        raise ValueError("attempt id requires a launch nonce, pid, and process start time")
    digest = sha256_bytes(canonical_json_bytes({
        "launch_nonce": str(launch_nonce),
        "pid": int(pid),
        "process_start_time": float(process_start_time),
    }))
    return f"{prefix}-attempt-{digest[:12]}"
