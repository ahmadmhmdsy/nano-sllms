"""Typed training events emitted through an injected sink."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = ["TRAINING_EVENT_NAMES", "EventEmitter", "SharedSequenceSource", "TrainingEvent", "TrainingEventName"]


TrainingEventName = Literal[
    "prepare",
    "step",
    "evaluation",
    "checkpoint",
    "health",
    "finalization",
]

TRAINING_EVENT_NAMES = (
    "prepare", "step", "evaluation", "checkpoint", "health", "finalization",
)


@dataclass(frozen=True)
class TrainingEvent:
    name: str
    sequence: int
    payload: Mapping[str, object]

    @property
    def kind(self) -> str:
        return self.name


class SharedSequenceSource:
    """One monotonic sequence counter, optionally shared across emitters."""

    def __init__(self) -> None:
        self._sequence = 0

    def next(self) -> int:
        self._sequence += 1
        return self._sequence

    @property
    def value(self) -> int:
        return self._sequence


#: Shared per-sink sequence registry. Independent EventEmitter instances
#: attached to the same sink draw from one monotonic source, so two
#: emitters on one sink can never both emit the same sequence number.
_SINK_SOURCES: dict[int, SharedSequenceSource] = {}


def shared_sequence_source(sink: Callable[[TrainingEvent], object]) -> SharedSequenceSource:
    """Return the monotonic counter bound to this exact sink identity."""
    key = id(sink)
    source = _SINK_SOURCES.get(key)
    if source is None:
        source = SharedSequenceSource()
        _SINK_SOURCES[key] = source
    return source


class EventEmitter:
    """Emit monotonic typed events through the injected sink; never prints.

    The sequence source is shared per sink: two emitters on one sink
    cannot both emit sequence 1.
    """

    def __init__(self, sink: Callable[[TrainingEvent], None]):
        self._sink = sink
        self._source = shared_sequence_source(sink)

    @property
    def sequence(self) -> int:
        return self._source.value

    def emit(self, kind: str, payload: Mapping[str, object]) -> TrainingEvent:
        if kind not in TRAINING_EVENT_NAMES:
            raise ValueError(f"unknown training event kind: {kind!r}")
        event = TrainingEvent(
            name=kind, sequence=self._source.next(), payload=payload,
        )
        self._sink(event)
        return event
