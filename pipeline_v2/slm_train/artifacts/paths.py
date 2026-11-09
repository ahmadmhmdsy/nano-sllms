"""V2 path policy: repository-relative, no traversal, no symlink components."""
from pathlib import Path, PurePosixPath, PureWindowsPath

from slm_train.identity.run_id import SAFE_RUN_COMPONENT


class V2Layout:
    prepared = "data/v2/prepared"
    tokens = "data/v2/tokens"
    attempts = "runs/v2/_attempts"
    final_runs = "runs/v2"


V2_LAYOUT = V2Layout()


def _safe_component(value: str, label: str) -> str:
    """Reject empty, dot, traversal, separator, drive, and unsafe components."""
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a plain path component: {value!r}")
    if PureWindowsPath(value).drive or value.startswith("\\\\"):
        raise ValueError(f"{label} must be a plain path component: {value!r}")
    if not SAFE_RUN_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} is not a safe run-id component: {value!r}")
    return value


def run_evidence_path(namespace: str, run_id: str) -> str:
    _safe_component(namespace, "namespace")
    _safe_component(run_id, "run_id")
    return f"{V2Layout.final_runs}/{namespace}/{run_id}"


class PathPolicy:
    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative: str) -> Path:
        if not relative:
            raise ValueError("empty path is not a valid V2 path")
        if relative.startswith("\\\\"):
            raise ValueError("UNC path is not allowed")
        posix = PurePosixPath(relative)
        win = PureWindowsPath(relative)
        if posix.is_absolute() or win.is_absolute():
            raise ValueError(f"absolute path is not allowed: {relative}")
        if ".." in posix.parts or ".." in win.parts:
            raise ValueError(f"path traversal is not allowed: {relative}")
        drive = win.drive
        if drive:
            raise ValueError(f"absolute path is not allowed: {relative}")
        current = self._root
        for part in PureWindowsPath(relative.replace("/", "\\")).parts:
            current = current / part
        # reject any symlink component before checking containment
        walked = self._root
        for part in PureWindowsPath(relative.replace("/", "\\")).parts:
            walked = walked / part
            if walked.is_symlink():
                raise ValueError(f"symlink component in path is not allowed: {relative}")
        resolved = (self._root / PureWindowsPath(relative.replace("/", "\\"))).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"path escapes policy root: {relative}")
        return resolved
