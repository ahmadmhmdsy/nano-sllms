"""Model loading that cannot execute code.

Two rules, and neither has an exception:

1. Only safetensors is accepted for weights. A ".pt" or ".bin" is refused by
   extension, before anything is opened.
2. Any tensor-bearing file that is not safetensors is loaded weights-only, with
   no unpickler that can call a constructor.

A config that names its own class to instantiate is a remote code execution
primitive, so a config-derived type name is refused rather than resolved.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "ALLOWED_WEIGHT_SUFFIXES",
    "UNSAFE_TYPE_KEYS",
    "ModelSecurityError",
    "require_safe_weight_suffix",
    "require_no_dynamic_type",
]

ALLOWED_WEIGHT_SUFFIXES = (".safetensors",)

#: Keys whose value is a type name or an import path. A config carrying one of
#: these is asking the loader to construct something on its behalf.
UNSAFE_TYPE_KEYS = (
    "auto_map",
    "architectures",
    "custom_pipelines",
    "module",
    "quantization_config",
)


class ModelSecurityError(ValueError):
    """A model artifact or config would have required executing untrusted code."""


def require_safe_weight_suffix(path: Path) -> Path:
    """Refuse any weight file that is not safetensors, before it is opened."""
    candidate = Path(path)
    if candidate.suffix.lower() not in ALLOWED_WEIGHT_SUFFIXES:
        listed = ", ".join(ALLOWED_WEIGHT_SUFFIXES)
        raise ModelSecurityError(
            "weights must be " + listed + ", refusing: " + str(candidate)
        )
    return candidate


def require_no_dynamic_type(config, where: str = "config") -> dict:
    """Refuse a config that names a type, a module, or an auto-mapping.

    The scan is DEEP. A top-level-only check is trivially bypassed by nesting
    one level down, and a nested auto_map is exactly how a config asks a loader
    to import and call something on its behalf.
    """
    _scan(config, where, "config")
    if not isinstance(config, dict):
        raise ModelSecurityError(where + " must be a mapping, not " + repr(config))
    return config


def _scan(node, where: str, trail: str) -> None:
    """Walk the whole document, refusing an unsafe key at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in UNSAFE_TYPE_KEYS:
                location = where + " carries " + trail + "." + str(key)
                raise ModelSecurityError(location + ", which can select a type to execute")
            _scan(value, where, trail + "." + str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _scan(value, where, trail + "[" + str(index) + "]")
