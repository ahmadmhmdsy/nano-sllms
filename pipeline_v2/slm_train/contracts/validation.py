"""Shared contract field validators fail closed on unsafe paths."""
from pathlib import PurePosixPath, PureWindowsPath


def require_repository_relative(value: str) -> str:
    """Reject empty, absolute, UNC, and traversing paths; return the value."""
    if not value:
        raise ValueError("repository-relative path is required")
    posix = PurePosixPath(value)
    win = PureWindowsPath(value)
    if value.startswith("\\\\") or win.is_absolute() or posix.is_absolute() or win.drive:
        raise ValueError(f"absolute path is not allowed: {value}")
    if ".." in posix.parts or ".." in win.parts:
        raise ValueError(f"path traversal is not allowed: {value}")
    return value
