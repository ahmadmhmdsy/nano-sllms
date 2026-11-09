"""Config inheritance resolution with cycle, escape, and merge guards."""

from copy import deepcopy
from pathlib import Path

import yaml

from .errors import ConfigError
from .loader import UniqueKeyLoader


def _read(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if key not in result:
            result[key] = deepcopy(value)
            continue
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        elif isinstance(result[key], list) and isinstance(value, list):
            if result[key] != value:
                raise ConfigError(f"ambiguous list merge for {key}")
        else:
            result[key] = deepcopy(value)
    return result


def resolve_inheritance(path: Path, *, root: Path | None = None) -> dict:
    boundary = (root or Path.cwd()).resolve()
    active: set[Path] = set()
    cache: dict[Path, dict] = {}

    def visit(current: Path) -> dict:
        resolved = current.resolve()
        try:
            resolved.relative_to(boundary)
        except ValueError as exc:
            raise ConfigError("config path escapes declared root") from exc
        if resolved in active:
            raise ConfigError(f"inheritance cycle detected at {resolved}")
        if resolved in cache:
            return deepcopy(cache[resolved])
        active.add(resolved)
        raw = _read(resolved)
        parents = raw.get("extends", [])
        if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
            raise ConfigError("extends must be a list of relative paths")
        merged: dict = {}
        for parent in parents:
            parent_path = Path(parent)
            if parent_path.is_absolute():
                raise ConfigError("extends entries must be relative to the declared root")
            # A ".." component is not itself the security property: a template in
            # configs/experiments must be able to name ../base/r1.yaml. What must
            # never happen is leaving the root, and that is decided by the
            # resolved-path check in visit(), which is stronger than a string test
            # because it also sees through symlinks.
            merged = _merge(merged, visit(resolved.parent / parent))
        local = {key: value for key, value in raw.items() if key != "extends"}
        merged = _merge(merged, local)
        merged["extends"] = tuple(parents)
        active.remove(resolved)
        cache[resolved] = deepcopy(merged)
        return merged

    return visit(path)
