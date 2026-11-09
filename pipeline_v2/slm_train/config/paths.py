from pathlib import Path


class PathPolicy:
    """Containment policy for persisted, repository-relative paths."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _reject_symlink_components(self, raw: Path) -> None:
        for parent in (raw, *raw.parents):
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError("symlink paths are forbidden")

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("persisted paths must be repository-relative")
        if ".." in candidate.parts:
            raise ValueError("path traversal is forbidden")
        if not candidate.parts:
            raise ValueError("empty path is forbidden")
        raw = self.root / candidate
        self._reject_symlink_components(raw)
        resolved = raw.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes declared root") from exc
        return resolved

    def relative(self, path: Path) -> str:
        raw = Path(path)
        if not raw.is_absolute():
            raw = self.root / raw
        self._reject_symlink_components(raw)
        resolved = raw.resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("path escapes declared root") from exc
