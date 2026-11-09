"""Immutable per-kind log segments plus a mutable, hash-chained log-manifest.

Design 8 layout: logs/<kind>/segment-NNNNNN.jsonl (lease, metrics, monitor),
with its own monotonic record sequence and index chain per kind. The
serialized shape is the contracts/log_index.py contract.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pydantic import ValidationError

from slm_train.artifacts.atomic import atomic_write_bytes, replace_mutable_bytes
from slm_train.contracts.hash import canonical_json_bytes, sha256_bytes
from slm_train.contracts.log_index import KindLogIndex, LogIndexContract

LOG_KINDS = ("lease", "metrics", "monitor")
LOGS_DIR = "logs"
INDEX_NAME = "log-manifest.json"
SEGMENT_TEMPLATE = "segment-{:06d}.jsonl"  # design 8: 1-based, six digits
_SEGMENT_NAME = re.compile(r"segment-(\d{6})\.jsonl")


def _published_segment_sequences(kind_dir: Path) -> "list[int]":
    """Exact segment naming only. A file in the kind directory that is not a
    numbered segment is a fail-closed condition, not a silently ignored stray."""
    sequences: "list[int]" = []
    for path in sorted(kind_dir.glob("segment-*.jsonl")):
        if not path.is_file():
            continue
        match = _SEGMENT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"foreign file in log segment directory {kind_dir.name!r}, "
                f"expected a segment-NNNNNN.jsonl name: {path.name}")
        sequences.append(int(match.group(1)))
    return sorted(sequences)


def _fresh_kind_state() -> KindLogIndex:
    return KindLogIndex(schema_version=1, generation=0, previous_index_hash="",
                        segment_hashes=[], first_sequence=-1, last_sequence=-1,
                        index_hash="")


def _kind_index_body(state: KindLogIndex) -> dict:
    return {
        "generation": state.generation,
        "previous_index_hash": state.previous_index_hash,
        "segment_hashes": list(state.segment_hashes),
        "first_sequence": state.first_sequence,
        "last_sequence": state.last_sequence,
    }


def kind_index_hash(state: KindLogIndex) -> str:
    """Chain hash of one kind's index (previous hash included in the body)."""
    return sha256_bytes(canonical_json_bytes(_kind_index_body(state)))


def _index_bytes(kinds: Mapping[str, KindLogIndex]) -> bytes:
    return canonical_json_bytes({
        "kinds": {name: state.model_dump() for name, state in sorted(kinds.items())},
    })


class LogIndexState:
    """Parsed, verified view of logs/log-manifest.json (mutable hash chain;
    not semantic identity). Sequence/index semantics are per kind."""

    def __init__(self, kinds: Mapping[str, KindLogIndex] | None = None) -> None:
        self._kinds: "dict[str, KindLogIndex]" = dict(kinds or {})

    def state_for(self, kind: str) -> KindLogIndex:
        """Per-kind state; an unpublished kind is a fresh generation-0 state."""
        if kind not in LOG_KINDS:
            raise ValueError(f"log stream kind is not a design stream: {kind!r}")
        return self._kinds.get(kind) or _fresh_kind_state()

    def to_json(self) -> bytes:
        return _index_bytes(self._kinds)

    @classmethod
    def load_from(cls, run_root: Path) -> "LogIndexState":
        path = Path(run_root) / LOGS_DIR / INDEX_NAME
        if not path.is_file():
            return cls()
        data = json.loads(path.read_bytes().decode("utf-8"))
        try:
            contract = LogIndexContract.model_validate(data)
        except ValidationError as error:
            raise ValueError(f"log-manifest.json failed schema validation: {error}") from error
        for name, state in contract.kinds.items():
            if kind_index_hash(state) != state.index_hash:
                raise ValueError(f"log index hash chain tampered for stream {name!r}")
        return cls(dict(contract.kinds))


LogIndex = LogIndexState  # stable public name


def segment_path(run_root: Path, kind: str, sequence: int) -> Path:
    if kind not in LOG_KINDS:
        raise ValueError(f"log stream kind is not a design stream: {kind!r}")
    return Path(run_root) / LOGS_DIR / kind / SEGMENT_TEMPLATE.format(int(sequence))


def append_log_segment(run_root: Path, kind: str, sequence: int, payload: bytes) -> Path:
    """Append one immutable, numbered log segment for a stream kind; never
    overwrite, never reuse a number, and never skip within a kind."""
    if kind not in LOG_KINDS:
        raise ValueError(f"log stream kind is not a design stream: {kind!r}")
    sequence = int(sequence)
    if sequence < 1:
        raise ValueError(f"log segment sequence starts at 1: {sequence}")
    root = Path(run_root)
    index = LogIndexState.load_from(root)
    state = index.state_for(kind)
    kind_dir = root / LOGS_DIR / kind
    if state.generation > 0:
        if sequence <= state.last_sequence:
            raise FileExistsError(
                f"log segment number already used within indexed range: {sequence}")
        if sequence != state.last_sequence + 1:
            raise ValueError(f"log gap: expected {state.last_sequence + 1}, got {sequence}")
    elif kind_dir.is_dir():
        published = _published_segment_sequences(kind_dir)
        if published and sequence not in (published[0], max(published) + 1):
            raise ValueError(f"log gap: segments exist but {sequence} does not continue them")
    target = segment_path(root, kind, sequence)
    if target.exists():
        raise FileExistsError(f"log segment was already published: {target}")
    segments = root / LOGS_DIR / kind
    segments.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, payload)
    previous_hash = "" if state.generation == 0 else state.index_hash
    new_state = KindLogIndex(
        schema_version=1,
        generation=state.generation + 1,
        previous_index_hash=previous_hash,
        segment_hashes=list(state.segment_hashes) + [sha256_bytes(payload)],
        first_sequence=state.first_sequence if state.first_sequence >= 0 else sequence,
        last_sequence=sequence,
        index_hash="",
    )
    new_state.index_hash = kind_index_hash(new_state)
    index._kinds[kind] = new_state
    replace_mutable_bytes(root / LOGS_DIR / INDEX_NAME, index.to_json())
    return target


RECOVERY_DIR = "_recovery"


def _state_for(run_root: Path, kind: str) -> KindLogIndex:
    if kind not in LOG_KINDS:
        raise ValueError(f"log stream kind is not a design stream: {kind!r}")
    return LogIndexState.load_from(Path(run_root)).state_for(kind)


def next_segment_number(run_root: Path, kind: str) -> int:
    """The next segment number a stream may publish. Never reuses a number."""
    state = _state_for(run_root, kind)
    if state.generation > 0:
        return state.last_sequence + 1
    kind_dir = Path(run_root) / LOGS_DIR / kind
    published = _published_segment_sequences(kind_dir) if kind_dir.is_dir() else []
    return published[0] if published else 1


def published_segment_numbers(run_root: Path, kind: str) -> tuple[int, ...]:
    """Exactly the segment numbers the index names. Never a directory listing."""
    state = _state_for(run_root, kind)
    if state.generation == 0:
        return ()
    count = state.last_sequence - state.first_sequence + 1
    if count != len(state.segment_hashes):
        raise ValueError(f"log index segment count disagrees with its hashes: {kind!r}")
    return tuple(range(state.first_sequence, state.last_sequence + 1))


def _stream_records(run_root: Path, kind: str) -> list[dict]:
    """Every published record of a stream, read only from indexed segments."""
    records: list[dict] = []
    for number in published_segment_numbers(run_root, kind):
        path = segment_path(Path(run_root), kind, number)
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def published_last_record(run_root: Path, kind: str) -> int:
    """The highest record sequence already published on this stream."""
    records = _stream_records(run_root, kind)
    return max((int(item["record"]) for item in records), default=0)


@dataclass(frozen=True)
class StreamVerification:
    """One stream reopened and checked for gaps and hash mismatches."""

    kind: str
    records: int
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_stream(run_root: Path, kind: str) -> StreamVerification:
    """Reopen one stream and check contiguity, ordering, and every hash."""
    root = Path(run_root)
    state = _state_for(root, kind)
    if state.generation == 0:
        return StreamVerification(kind=kind, records=0, problems=())
    problems: list[str] = []
    numbers = published_segment_numbers(root, kind)
    for offset, number in enumerate(numbers):
        path = segment_path(root, kind, number)
        if not path.is_file():
            problems.append(f"missing published segment {SEGMENT_TEMPLATE.format(number)}")
            continue
        payload = path.read_bytes()
        if sha256_bytes(payload) != state.segment_hashes[offset]:
            problems.append(f"segment hash mismatch: {SEGMENT_TEMPLATE.format(number)}")
    expected = 0
    for record in _stream_records(root, kind):
        value = int(record["record"])
        if value != expected + 1:
            problems.append(f"record gap: expected {expected + 1}, got {value}")
        expected = value
    return StreamVerification(
        kind=kind, records=expected, problems=tuple(problems),
    )


def verify_all_streams(run_root: Path) -> dict[str, StreamVerification]:
    """Every designed stream, including the lease stream, checked before PASS."""
    return {kind: verify_stream(Path(run_root), kind) for kind in LOG_KINDS}
