"""The run event bus.

Every run event carries a monotonic (stream, segment, record) sequence and the
run identity, and is written as an immutable numbered JSONL segment. The index
is updated atomically after the segment lands, so a segment that is not in the
index is not published, and an index that names a segment means that segment is
complete on disk.

The bus is CPU-only and has no torch import: emitting a health warning must never
be the thing that pulls CUDA onto a CPU path.
"""
from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock, Timeout

from slm_train.artifacts.log_index import (
    next_segment_number,
    published_last_record,
)

RUN_EVENT_NAMES: tuple[str, ...] = (
    "run_started",
    "stage_started",
    "step_completed",
    "metric_emitted",
    "evaluation_completed",
    "checkpoint_completed",
    "health_warning",
    "health_failure",
    "stage_completed",
    "run_completed",
    # Operator diagnostics (2026-09-26 plan, Phase 1): a bounded per-group
    # VRAM accounting event and the OOM refusal record. Both stream into the
    # monitor segment so a BLOCKED run carries its own memory evidence.
    "vram_probe",
    "oom_accounting",
    "event_refused",
)

#: Run events map onto the three designed log streams. A name outside this map
#: is a programming error and is refused, not written somewhere unrecoverable.
STREAM_FOR_EVENT: dict[str, str] = {
    "run_started": "monitor",
    "stage_started": "monitor",
    "step_completed": "metrics",
    "metric_emitted": "metrics",
    "evaluation_completed": "metrics",
    "checkpoint_completed": "monitor",
    "health_warning": "monitor",
    "health_failure": "monitor",
    "stage_completed": "monitor",
    "run_completed": "monitor",
    "vram_probe": "monitor",
    "oom_accounting": "monitor",
    "event_refused": "monitor",
}

MAX_RECORDS_PER_SEGMENT = 1000

#: one exclusive lock per run root, held only for the publication of a flush
FLUSH_LOCK_NAME = ".flush.lock"
FLUSH_LOCK_TIMEOUT = 30


class EventBusError(RuntimeError):
    """A run event could not be published. Never swallowed."""


class EventBus:
    """Publish run events to immutable JSONL segments under logs/<stream>/."""

    def __init__(self, run_root, run_id: str, max_records: int = MAX_RECORDS_PER_SEGMENT):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(max_records, int) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.max_records = max_records
        self._buffer: dict[str, list[dict]] = {}
        self._records: dict[str, int] = {}

    def _next_record(self, stream: str) -> int:
        """Continue the stream record sequence from what is already published."""
        if stream not in self._records:
            self._records[stream] = published_last_record(self.run_root, stream)
        self._records[stream] += 1
        return self._records[stream]

    def emit(self, name: str, payload: dict) -> dict:
        """Buffer one run event. It is published on flush, not on emit."""
        if name not in RUN_EVENT_NAMES:
            listed = ", ".join(RUN_EVENT_NAMES)
            raise EventBusError("unknown run event: " + repr(name) + "; expected one of " + listed)
        if not isinstance(payload, dict):
            raise EventBusError("run event payload must be a mapping: " + name)
        stream = STREAM_FOR_EVENT[name]
        # Serialize before consuming a record number. A payload that cannot be
        # encoded must not burn a sequence value, or the stream would carry a
        # permanent gap that no later record could close.
        json.dumps(payload, sort_keys=True)
        record = {
            "stream": stream,
            "record": self._next_record(stream),
            "event": name,
            "run_id": self.run_id,
            "payload": payload,
        }
        self._buffer.setdefault(stream, []).append(record)
        return record

    def emit_or_quarantine(self, name: str, payload: dict) -> dict:
        """Emit, or if the name is not a declared event, publish the refusal.

        A forgotten event name must never cost its evidence: the refused
        payload is published as a quarantine record instead of vanishing.
        Raise-free path: encoding failures still raise (a payload that cannot
        serialize is a caller bug, not an operator event).
        """
        if name not in RUN_EVENT_NAMES:
            return self.emit("event_refused", {
                "refused_event": str(name),
                "payload": payload,
                "known": list(RUN_EVENT_NAMES),
            })
        return self.emit(name, payload)

    def pending(self, stream: str) -> int:
        """Buffered but unpublished records. These are never complete."""
        return len(self._buffer.get(stream, ()))

    def __del__(self) -> None:
        """Flush-on-close: a crashed stage still carries its own evidence.

        A run that dies between an emit and an explicit flush used to lose the
        events it had produced, which is exactly the failure mode the events
        exist to diagnose. Best effort only: a finalizer must never raise.
        """
        try:
            self.flush()
        except Exception:
            pass

    def flush(self) -> list:
        """Publish every buffered record as immutable numbered segments.

        The renumbering and the publication happen under one exclusive per-run
        lock. Without it, two buses on the same run root each compute the next
        record number independently and publish the same value, which corrupts
        the stream permanently and cannot be repaired afterwards.
        """
        from slm_train.artifacts.log_index import append_log_segment

        lock_path = self.run_root / "logs" / FLUSH_LOCK_NAME
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(lock_path), timeout=FLUSH_LOCK_TIMEOUT):
                return self._publish_locked()
        except Timeout as error:
            raise EventBusError(
                "another writer holds the log flush lock: " + str(lock_path)
            ) from error

    def _publish_locked(self) -> list:
        from slm_train.artifacts.log_index import append_log_segment

        published: list = []
        for stream in sorted(self._buffer):
            records = self._buffer[stream]
            if not records:
                continue
            base = published_last_record(self.run_root, stream)
            for offset, record in enumerate(records):
                record["record"] = base + offset + 1
            self._records[stream] = base + len(records)
            lines = [
                json.dumps(record, sort_keys=True) for record in records
            ]
            number = next_segment_number(self.run_root, stream)
            for start in range(0, len(lines), self.max_records):
                chunk = lines[start:start + self.max_records]
                payload = ("\n".join(chunk) + "\n").encode("utf-8")
                path = append_log_segment(self.run_root, stream, number, payload)
                published.append((stream, number, path))
                number += 1
            self._buffer[stream] = []
        return published
