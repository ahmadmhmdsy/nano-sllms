"""Device selection for an explicitly requested GPU run."""
from __future__ import annotations


class DeviceUnavailable(RuntimeError):
    """The requested device cannot be used. Never silently fall back to CPU."""


def select_device(requested: str) -> str:
    """Normalize and validate a requested device string.

    This performs no hardware access: it only rejects a request the pipeline
    will not honour. A caller that names "auto" still gets a BLOCKED result from
    the probe rather than a quiet CPU downgrade.
    """
    if not isinstance(requested, str) or not requested.strip():
        raise DeviceUnavailable("a device must be named explicitly")
    text = requested.strip()
    if text == "cpu":
        raise DeviceUnavailable("a GPU run may not request the cpu device")
    if text == "auto":
        raise DeviceUnavailable("device auto is not resolvable without a GPU probe")
    if not text.startswith("cuda"):
        raise DeviceUnavailable("only cuda devices are supported: " + repr(requested))
    suffix = text[4:]
    if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
        raise DeviceUnavailable("malformed cuda device: " + repr(requested))
    if suffix and int(suffix[1:]) < 0:
        raise DeviceUnavailable("cuda device index must not be negative")
    return text


def enforce_allocator_ceiling(device: str, fraction: float) -> float:
    """Cap this process at a fraction of the device before any tensor is made.

    A thin delegate on purpose: the implementation lives in resources.gpu_probe
    because that is the only module permitted to touch a device, and this one
    must stay free of any device API of its own.
    """
    from slm_train.resources.gpu_probe import apply_allocator_ceiling

    return apply_allocator_ceiling(device, fraction)
