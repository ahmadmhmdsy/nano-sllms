"""Explicit preflight report.

preflight is the report a human reads before deciding to open a GPU window. In
its CPU form it never touches CUDA. In its --gpu form it is the ONLY caller of
probe_gpu, and it records the exact report path so the decision is evidence
rather than a remembered number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from slm_train.artifacts.atomic import atomic_write_json

REPORT_NAME = "preflight.json"


def report_path(run_root: Path) -> Path:
    return Path(run_root) / REPORT_NAME


def run_cpu_preflight(root: Path, doctor_result: Optional[dict] = None) -> dict:
    """The CPU preflight. Identical to a doctor run, with an explicit reason."""
    from slm_train.resources.doctor import run_doctor

    result = doctor_result if doctor_result is not None else run_doctor(root=Path(root))
    return {
        "mode": "cpu",
        "status": result["status"],
        "reason": "CPU preflight never touches the GPU",
        "gpu_touched": False,
        "checks": result["checks"],
    }


def run_gpu_preflight(root: Path, device: str) -> dict:
    """The GPU preflight. A probe failure is BLOCKED, never a CPU downgrade."""
    from slm_train.resources.device import DeviceUnavailable, select_device
    from slm_train.resources.gpu_probe import GpuProbeError, probe_gpu

    base: dict[str, Any] = {
        "mode": "gpu",
        "gpu_touched": False,
        "device": device,
        "observation": {},
        "unavailable": (),
        "estimate": {},
        "decision": "BLOCKED",
        "reason": "",
    }
    try:
        selected = select_device(device)
    except DeviceUnavailable as exc:
        base["reason"] = str(exc)
        return base
    base["device"] = selected
    try:
        observed: dict[str, Any] = probe_gpu(selected)
    except GpuProbeError as exc:
        base["reason"] = "probe failed: " + str(exc)
        return base
    base["gpu_touched"] = True
    base["observation"] = observed
    base["unavailable"] = tuple(observed.get("unavailable", ()))
    free_bytes = int(observed.get("free_memory_bytes", 0))
    base["estimate"] = {"free_memory_bytes": free_bytes}
    if free_bytes <= 0:
        base["reason"] = "device reports no free memory"
        return base
    base["decision"] = "PASS"
    base["reason"] = "device is visible and reports free memory"
    return base


def write_preflight(root: Path, report: dict) -> Path:
    """Persist the report and return the exact path it was written to."""
    target = report_path(root)
    atomic_write_json(target, report, replace_existing=True)
    return target
