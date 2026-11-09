import hashlib
import json


CANONICAL_FORMAT_VERSION = "canonical-json-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
