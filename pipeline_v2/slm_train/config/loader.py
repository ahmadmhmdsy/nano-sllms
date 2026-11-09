"""YAML loading with duplicate-key detection and strict validation."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from .errors import ConfigError
from .models import ConfigDocument


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: UniqueKeyLoader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _normalize_sequences(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_sequences(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_normalize_sequences(item) for item in value)
    return value


def _validation_message(exc: ValidationError) -> str:
    """Translate Pydantic's generic extra-key wording to the public contract."""
    return str(exc).replace("Extra inputs are not permitted", "unknown key is not permitted")


def load_config(path: Path, *, root: Path | None = None) -> ConfigDocument:
    try:
        from .inheritance import resolve_inheritance

        resolved_path = Path(path).resolve()
        boundary = Path(root).resolve() if root is not None else None
        if boundary is None:
            cwd = Path.cwd().resolve()
            boundary = cwd if resolved_path.is_relative_to(cwd) else resolved_path.parent.parent
        raw = resolve_inheritance(Path(path), root=boundary)
        return ConfigDocument.model_validate(_normalize_sequences(raw))
    except ValidationError as exc:
        raise ConfigError(_validation_message(exc)) from exc
    except (OSError, yaml.YAMLError, ConfigError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
