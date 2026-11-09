"""Manifest publication with mandatory read-back."""
import json
from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.contracts.hash import canonical_json_bytes


def write_manifest(root: Path, relative_path: str, value: object) -> str:
    """Serialize value, publish it, and verify by byte compare + full re-parse.
    The reopened bytes are re-parsed as JSON and re-serialized bit-exactly
    before the write is declared successful."""
    if not isinstance(value, (dict, list)):
        raise ValueError("manifest read-back precondition failed: manifest must be JSON")
    try:
        payload = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"manifest read-back precondition failed: not canonical JSON ({error})") from error
    target = Path(root) / relative_path
    atomic_write_bytes(target, payload)
    read_back = target.read_bytes()
    if read_back != payload:
        raise ValueError("manifest read-back mismatch after publication")
    try:
        reparsed = json.loads(read_back.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"manifest re-parse validation failed after publication ({error})") from error
    if canonical_json_bytes(reparsed) != payload:
        raise ValueError("manifest re-parse validation failed after publication")
    return read_back.decode("utf-8")


def load_manifest(root: Path, relative_path: str) -> object:
    data = (Path(root) / relative_path).read_bytes()
    return json.loads(data.decode("utf-8"))
