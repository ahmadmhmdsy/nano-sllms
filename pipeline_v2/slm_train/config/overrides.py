"""Explicit, allowlisted CLI override application."""

from copy import deepcopy
from typing import Any

from .models import ConfigDocument


SAFE_OVERRIDES = frozenset({"run.seed", "runtime.device", "runtime.precision"})


def apply_safe_overrides(config: ConfigDocument, values: dict[str, Any]) -> ConfigDocument:
    unknown = set(values) - SAFE_OVERRIDES
    if unknown:
        raise ValueError(f"unsafe config overrides: {sorted(unknown)}")
    updated = deepcopy(config.model_dump())
    for path, value in values.items():
        section, field = path.split(".", 1)
        if section not in updated or field not in updated[section]:
            raise ValueError(f"unknown override target: {path}")
        updated[section][field] = value
    return ConfigDocument.model_validate(updated)
