"""Complete, serializable training state.

The cursor is the position in the data stream, and it is recorded to the
microbatch boundary so an interrupted run resumes on exactly the same example
with exactly the same accumulation state. The rest of the state is what an
exact resume needs: every RNG that could advance, and every lifecycle cursor
that a restart must not silently reset.

Nothing here touches the GPU, torch, or the filesystem. State is a value: it is
built by a caller that has read real tensors, serialised by the checkpoint
writer, and restored by the resume path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["CheckpointState", "TrainingCursor"]

#: The state files a complete checkpoint must carry, in publication order.
STATE_ROLES: tuple[str, ...] = (
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "global_rng",
    "sampler",
    "dataloader_generator",
    "worker_rng",
    "early_stop",
    "evaluation",
    "lifecycle",
)


@dataclass(frozen=True)
class TrainingCursor:
    """Position in the data stream, recorded at a microbatch boundary."""

    examples_seen: int = 0
    microbatch_position: int = 0
    accumulation_boundary: int = 0
    epoch: int = 0

    def __post_init__(self) -> None:
        for name in ("examples_seen", "microbatch_position", "accumulation_boundary", "epoch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(name + " must be a non-negative integer")
        if self.microbatch_position > self.accumulation_boundary:
            raise ValueError("microbatch_position exceeds accumulation_boundary")

    @property
    def at_boundary(self) -> bool:
        """True when no partial accumulation is in flight."""
        return self.microbatch_position == self.accumulation_boundary

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TrainingCursor":
        if not isinstance(value, dict):
            raise ValueError("a cursor must round-trip through a mapping")
        missing = [
            key
            for key in ("examples_seen", "microbatch_position", "accumulation_boundary", "epoch")
            if key not in value
        ]
        if missing:
            raise ValueError("cursor is missing fields: " + ", ".join(missing))
        unexpected = set(value) - {
            "examples_seen", "microbatch_position", "accumulation_boundary", "epoch",
        }
        if unexpected:
            raise ValueError("cursor has unexpected fields: " + ", ".join(sorted(unexpected)))
        return cls(**{key: value[key] for key in value})


@dataclass(frozen=True)
class CheckpointState:
    """Every piece of state an exact resume needs, named by its file role."""

    cursor: TrainingCursor
    model: str
    optimizer: str
    scheduler: str
    global_rng: str
    sampler: str
    dataloader_generator: str
    early_stop: str
    evaluation: str
    lifecycle: str
    scaler: str | None = None
    worker_rng: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, TrainingCursor):
            raise ValueError("cursor must be a TrainingCursor")
        for name in STATE_ROLES:
            if name == "worker_rng":
                continue
            value = self.file_for(name)
            if value is None:
                raise ValueError("checkpoint state is missing required role: " + name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("state role must be a non-empty path: " + name)
        if not all(isinstance(item, str) and item.strip() for item in self.worker_rng):
            raise ValueError("worker_rng entries must be non-empty paths")

    def file_for(self, role: str):
        """The path for one scalar state role.

        worker_rng is a tuple of per-worker paths rather than a single path, so it
        is not addressable here; use files() to get every recorded file.
        """
        if role not in STATE_ROLES:
            raise ValueError("unknown checkpoint state role: " + str(role))
        if role == "worker_rng":
            raise ValueError("worker_rng is a tuple; read the whole mapping instead")
        return getattr(self, role)

    def files(self) -> dict[str, str]:
        """Role -> path for every required file, plus each worker RNG in order."""
        mapping = {role: getattr(self, role) for role in STATE_ROLES if role != "worker_rng"}
        for index, path in enumerate(self.worker_rng):
            mapping["worker_rng." + str(index)] = path
        return mapping

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "CheckpointState":
        if not isinstance(value, dict):
            raise ValueError("checkpoint state must round-trip through a mapping")
        payload = dict(value)
        cursor = payload.pop("cursor", None)
        return cls(cursor=TrainingCursor.from_dict(cursor), **payload)
