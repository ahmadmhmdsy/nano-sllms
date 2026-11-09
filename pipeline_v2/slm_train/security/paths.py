"""Path containment, resolved through every symlink before the answer is given.

A string prefix check is not containment. "runs/a" starts with "runs", so a naive
prefix test would accept "runs/../../etc" and would accept a symlink pointing
outside. Every function here resolves the real path of every component first,
then compares resolved paths, so a traversal and a symlink escape are both
refused rather than described.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PathSecurityError", "is_contained", "require_contained", "require_regular_file"]


class PathSecurityError(ValueError):
    """A path left its root, or resolved to something other than a plain file."""


def _resolved(path: Path) -> Path:
    """The real path, with every symlink followed, without requiring existence."""
    return Path(os.path.realpath(str(Path(path))))


def is_contained(root: Path, candidate: Path) -> bool:
    """True only when the RESOLVED candidate is strictly inside the RESOLVED root."""
    real_root = _resolved(root)
    real_candidate = _resolved(candidate)
    try:
        relative = real_candidate.relative_to(real_root)
    except ValueError:
        return False
    # A root is not inside itself: an allowlist entry must name a real child.
    return bool(relative.parts)


def require_contained(root: Path, candidate: Path, what: str = "path") -> Path:
    """Return the resolved candidate, or refuse with the reason it was refused."""
    real_root = _resolved(root)
    real_candidate = _resolved(candidate)
    if real_candidate == real_root:
        raise PathSecurityError(what + " may not be the root itself: " + str(candidate))
    if not is_contained(root, candidate):
        raise PathSecurityError(
            what + " escapes its root after resolving symlinks: "
            + str(candidate) + " -> " + str(real_candidate)
        )
    return real_candidate


def require_regular_file(root: Path, candidate: Path, what: str = "file") -> Path:
    """A contained path that is a real file, not a directory, link, or device."""
    resolved = require_contained(root, candidate, what)
    if not resolved.is_file():
        raise PathSecurityError(what + " is not a regular file: " + str(candidate))
    return resolved
