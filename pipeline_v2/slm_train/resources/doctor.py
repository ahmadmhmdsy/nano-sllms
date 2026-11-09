"""CPU-only environment doctor.

The doctor answers "is this checkout ready to run" without touching the GPU. It
never imports torch, never imports torch.cuda, never shells out to nvidia-smi,
and never acquires a lease. This module deliberately does not import, alias, or
re-export the GPU probe: a test asserts that the only importer of probe_gpu in
the whole V2 package is preflight.py.
"""
from __future__ import annotations

import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

V2_ROOTS = ("data/v2", "runs/v2", "artifacts/v2", "logs")
REQUIRED_MODULES = ("pydantic", "torch", "transformers", "numpy", "filelock")


def _check_venv() -> dict[str, str]:
    venv = os.environ.get("VIRTUAL_ENV", "")
    prefix = getattr(sys, "prefix", "")
    if not venv and not prefix:
        return {"label": "FAIL", "detail": "no virtual environment is active"}
    return {"label": "PASS", "detail": "venv=" + (venv or prefix)}


def _check_package_metadata() -> dict[str, str]:
    try:
        version = metadata.version("numpy")
    except Exception as exc:
        return {"label": "FAIL", "detail": "package metadata unreadable: " + str(exc)}
    return {"label": "PASS", "detail": "numpy=" + version}


def _check_imports() -> dict[str, str]:
    import importlib.util

    missing = [
        name for name in REQUIRED_MODULES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return {
            "label": "BLOCKED",
            "detail": "missing modules: " + ", ".join(sorted(missing)),
        }
    return {"label": "PASS", "detail": "all required modules are importable"}


def _check_writable_roots(root: Path) -> dict[str, str]:
    if not root.exists():
        return {
            "label": "BLOCKED",
            "detail": "V2 root does not exist: " + str(root),
        }
    for name in V2_ROOTS:
        target = root / name
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {
                "label": "FAIL",
                "detail": "cannot create " + name + ": " + str(exc),
            }
    return {"label": "PASS", "detail": "V2 roots are writable"}


def _check_disk(root: Path) -> dict[str, str]:
    try:
        usage = shutil.disk_usage(str(root if root.exists() else Path(".")))
    except OSError as exc:
        return {"label": "BLOCKED", "detail": "disk usage unreadable: " + str(exc)}
    free_gib = usage.free / (1024 ** 3)
    if free_gib < 1.0:
        return {
            "label": "BLOCKED",
            "detail": "only " + format(free_gib, ".2f") + " GiB free",
        }
    return {
        "label": "PASS",
        "detail": format(free_gib, ".2f") + " GiB free",
    }


def _check_config_paths(root: Path) -> dict[str, str]:
    if not (root / "pipeline_v2").exists():
        return {
            "label": "BLOCKED",
            "detail": "pipeline_v2 is not present under " + str(root),
        }
    return {"label": "PASS", "detail": "pipeline_v2 is present"}


def _worst(labels: list[str]) -> str:
    for candidate in ("FAIL", "BLOCKED", "NEEDS_USER_DECISION", "SKIPPED"):
        if candidate in labels:
            return candidate
    return "PASS"


def run_doctor(root: Path) -> dict:
    """Run every CPU-only check and report one labeled status. Never probes the GPU."""
    root = Path(root)
    checks = {
        "venv": _check_venv(),
        "package_metadata": _check_package_metadata(),
        "imports": _check_imports(),
        "writable_roots": _check_writable_roots(root),
        "disk": _check_disk(root),
        "config_paths": _check_config_paths(root),
    }
    return {
        "status": _worst([check["label"] for check in checks.values()]),
        "gpu_touched": False,
        "checks": checks,
    }
