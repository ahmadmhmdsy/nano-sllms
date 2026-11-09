"""Export: the allowlist is the authorization, and it is passed straight through.

The failure this module exists to prevent is an export that copies a run root
because a caller asked for "the run" rather than naming the artifacts. So there
is no default allowlist, no glob, and no directory walk: a caller supplies exact
run-relative paths, each one is checked for containment, and only the checked
list reaches the copy operation.

The checked list IS the list handed to the copy. Re-deriving it afterwards
would reintroduce exactly the drift this guards against.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from slm_train.security.paths import PathSecurityError, require_regular_file

__all__ = ["ExportError", "ExportRequest", "export_allowlist", "resolve_allowlist"]


class ExportError(RuntimeError):
    """An export was refused, or it did not copy exactly what it was allowed to."""


class ExportRequest:
    """What to export and where to. There is no other way to build one."""

    __slots__ = ("run_root", "destination", "allowlist")

    def __init__(self, run_root, destination, allowlist):
        if not isinstance(allowlist, (list, tuple)) or not allowlist:
            raise ExportError("an export needs an explicit, non-empty allowlist")
        self.run_root = Path(run_root)
        self.destination = Path(destination)
        self.allowlist = tuple(str(item) for item in allowlist)

    def __repr__(self) -> str:
        return "ExportRequest(allowlist=" + repr(list(self.allowlist)) + ")"


def resolve_allowlist(request):
    """Every allowlist entry, checked for containment. An empty result is a refusal."""
    resolved = []
    for name in request.allowlist:
        candidate = Path(request.run_root) / name
        try:
            resolved.append((name, require_regular_file(request.run_root, candidate, name)))
        except PathSecurityError as error:
            raise ExportError("export refused: " + str(error)) from error
    if not resolved:
        raise ExportError("export refused: the allowlist resolved to nothing")
    return resolved


def export_allowlist(request):
    """Copy exactly the checked allowlist, and report exactly what was copied.

    The list returned here is the same list that was checked. Nothing is copied
    that was not checked, and nothing checked is silently skipped.
    """
    checked = resolve_allowlist(request)
    copied = []
    for name, source in checked:
        destination = Path(request.destination) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as error:
            raise ExportError("export failed for " + name + ": " + str(error)) from error
        if not destination.is_file():
            raise ExportError("export did not produce " + name)
        copied.append(name)
    if sorted(copied) != sorted(name for name, _ in checked):
        raise ExportError("export copied a different set than it was allowed to")
    return copied
