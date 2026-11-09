"""Atomic writes: temp sibling, flush+fsync, hard-link publish or os.replace."""
import os
import tempfile
from pathlib import Path


def fsync_directory(path: Path) -> bool:
    """Make a directory entry durable; report False when unsupported."""
    if not hasattr(os, "O_DIRECTORY"):
        return False
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
        return True
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, replace_existing: bool = False) -> bool:
    """Write payload atomically.

    Default mode publishes an immutable file: fails on any existing target,
    uses an exclusive same-filesystem hard link so a concurrent creator is
    never overwritten. replace_existing=True publishes an explicitly mutable
    file via os.replace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replace_mutable_bytes(path: Path, payload: bytes) -> bool:
    return atomic_write_bytes(path, payload, replace_existing=True)


def atomic_write_json(path: Path, value: object, *, replace_existing: bool = False) -> bool:
    from slm_train.contracts.hash import canonical_json_bytes
    return atomic_write_bytes(path, canonical_json_bytes(value), replace_existing=replace_existing)
