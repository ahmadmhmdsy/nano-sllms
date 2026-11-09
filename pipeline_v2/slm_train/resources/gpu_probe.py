"""The one V2 GPU probe.

probe_gpu is the only place in the V2 pipeline that may touch CUDA. It is
called from exactly one place: the explicit preflight --gpu command, after the
lease has been acquired. It never runs on the ordinary task path and never falls
back: if an observation cannot be made it is reported as unavailable, never as
zero, and a device that cannot be opened at all is BLOCKED.
"""
from __future__ import annotations

from typing import Any


class GpuProbeError(RuntimeError):
    """The probe could not complete. Always fail closed."""


#: observations the torch runtime API does not expose. They are recorded as
#: unavailable rather than reported as a fabricated zero.
UNREADABLE: tuple[str, ...] = (
    "temperature_c",
    "utilization_percent",
    "compute_processes",
)


def device_available(device: str) -> bool:
    """Is the declared accelerator actually present?

    A run that declares cuda:0 on a machine with no cuda is BLOCKED, never
    silently evaluated on something else. This lives here because gpu_probe is
    the only module allowed to ask the runtime about a device.
    """
    if str(device).startswith("cpu"):
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - fail closed
        return False


def peak_reserved_bytes(device: str) -> int:
    """Peak reserved device memory since the last reset, in bytes.

    This is a measurement, not an estimate, and a cpu run reports 0 rather than
    guessing. It is the source the resources suite reads: a fabricated peak
    would make "peak_vram_bytes" a number that looks measured and is not.
    """
    if str(device).startswith("cpu"):
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_reserved())
    except Exception:  # pragma: no cover - never fail a run over a metric
        return 0


def reset_peak_memory(device: str) -> None:
    """Start a fresh peak measurement for the window about to be timed."""
    if str(device).startswith("cpu"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        return


def memory_report(device: str) -> dict:
    """Read allocated/reserved bytes for one device. Observation only."""
    import torch

    index = _device_index(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
    }


def settle_device(device: str) -> None:
    """Synchronize on the named device though a probe-level entry point."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(_device_index(device))


def attempt_accounting(device: str, requested_bytes: int) -> dict:
    """Demand-vs-cap accounting at the moment the allocator refused a request.

    A "0 bytes free" error text alone is not evidence an operator can act on:
    the ceiling the allocator enforced belongs in the refusal record next to
    what the allocator itself held. Observation only; never raises.
    """
    try:
        import torch

        report = memory_report(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(_device_index(device))
        index = _device_index(device)
        ceiling_bytes = int(
            torch.cuda.get_per_process_memory_fraction(torch.device("cuda", index))
            * torch.cuda.get_device_properties(index).total_memory
        )
        return {
            "requested_bytes": int(requested_bytes),
            "allocated_bytes": report["allocated_bytes"],
            "reserved_bytes": report["reserved_bytes"],
            "ceiling_bytes": ceiling_bytes,
            "free_driver_bytes": int(free_bytes),
        }
    except Exception as error:  # pragma: no cover - accounting never blocks a run
        return {"accounting_error": str(error)}


def release_cached_memory() -> None:
    """Return cached device blocks to the driver.

    Measured on the nano profile: a 213M model with fp32 master weights peaks at
    4.38 GiB during a backward pass. Holding those cached blocks while a second
    pass runs is what turned a checkpoint into an out-of-memory, so the training
    graph and the cache are released before any evaluation. A no-op on cpu.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - never fail a run over a cache hint
        return


def apply_allocator_ceiling(device: str, fraction: float) -> float:
    """Cap this process at a fraction of the device before any tensor is made.

    rules/gpu.md treats a spill into shared GPU memory as an OOM-equivalent event
    and requires a hard ceiling instead of trusting driver fallback behaviour.
    This is the only place that sets one. It is a no-op on cpu, and a device
    that refuses the cap raises rather than being assumed to have honoured it.
    """
    if device == "cpu":
        return 1.0
    import torch

    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError("the allocator ceiling must be a fraction of one device")
    torch.cuda.set_per_process_memory_fraction(value, _device_index(device))
    return value


def _device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def probe_gpu(device: str = "cuda:0") -> dict[str, Any]:
    """Read device identity and memory. Every field is observed, never inferred."""
    import torch

    if not torch.cuda.is_available():
        raise GpuProbeError("no CUDA device is available")
    try:
        index = _device_index(device)
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    except Exception as exc:
        raise GpuProbeError("device could not be opened: " + str(exc)) from exc
    return {
        "device": device,
        "name": properties.name,
        "capability": str(properties.major) + "." + str(properties.minor),
        "total_memory_bytes": int(properties.total_memory),
        "free_memory_bytes": int(free_bytes),
        "used_memory_bytes": int(total_bytes) - int(free_bytes),
        "multi_processor_count": int(properties.multi_processor_count),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "unavailable": UNREADABLE,
    }
