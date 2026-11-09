"""Task 8 unit tests, part 3: device selection and the preflight report."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from slm_train.resources.device import DeviceUnavailable, select_device
from slm_train.resources.preflight import (
    run_cpu_preflight,
    run_gpu_preflight,
    write_preflight,
)

V2_PACKAGE = Path(__file__).resolve().parents[2] / "slm_train"


@pytest.mark.parametrize("good", ["cuda", "cuda:0", "cuda:3", " cuda:1 "])
def test_device_selection_accepts_an_explicit_cuda_request(good) -> None:
    assert select_device(good) == good.strip()


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "cpu", "auto", "mps", "cuda:", "cuda:x", "cuda:-1", "gpu:0", None, 0],
)
def test_device_selection_refuses_anything_but_an_explicit_cuda_request(bad) -> None:
    with pytest.raises(DeviceUnavailable):
        select_device(bad)


def test_a_refused_device_never_degrades_to_cpu() -> None:
    with pytest.raises(DeviceUnavailable) as caught:
        select_device("auto")
    assert "cpu" not in str(caught.value).lower() or "auto" in str(caught.value)



def test_cpu_preflight_never_touches_the_gpu(tmp_path, monkeypatch) -> None:
    from slm_train.resources import gpu_probe

    def forbidden(*args, **kwargs):
        raise AssertionError("the CPU preflight must not probe the GPU")

    monkeypatch.setattr(gpu_probe, "probe_gpu", forbidden)
    report = run_cpu_preflight(root=tmp_path)
    assert report["mode"] == "cpu"
    assert report["gpu_touched"] is False
    assert report["status"] in ("PASS", "FAIL", "BLOCKED", "NEEDS_USER_DECISION")


def test_gpu_preflight_blocks_when_the_probe_fails(tmp_path, monkeypatch) -> None:
    from slm_train.resources import gpu_probe

    def refused(device="cuda:0"):
        raise gpu_probe.GpuProbeError("no CUDA device is available")

    monkeypatch.setattr(gpu_probe, "probe_gpu", refused)
    report = run_gpu_preflight(root=tmp_path, device="cuda:0")
    assert report["decision"] == "BLOCKED"
    assert report["reason"].startswith("probe failed:")
    assert "cpu" not in report, "a failed probe must never fall back to the cpu"


def test_gpu_preflight_blocks_on_an_unreadable_device(tmp_path) -> None:
    report = run_gpu_preflight(root=tmp_path, device="auto")
    assert report["decision"] == "BLOCKED"
    assert report["gpu_touched"] is False


def test_gpu_preflight_records_the_observation_and_the_report_path(tmp_path, monkeypatch) -> None:
    from slm_train.resources import gpu_probe

    observed = {
        "device": "cuda:0",
        "name": "probe-card",
        "free_memory_bytes": 8 * 1024 ** 3,
        "total_memory_bytes": 16 * 1024 ** 3,
        "unavailable": gpu_probe.UNREADABLE,
    }
    monkeypatch.setattr(gpu_probe, "probe_gpu", lambda device="cuda:0": dict(observed))
    report = run_gpu_preflight(root=tmp_path, device="cuda:0")
    assert report["decision"] == "PASS"
    assert report["gpu_touched"] is True
    assert report["observation"]["name"] == "probe-card"
    assert report["estimate"]["free_memory_bytes"] == 8 * 1024 ** 3
    assert set(report["unavailable"]) == set(gpu_probe.UNREADABLE)
    path = write_preflight(root=tmp_path, report=report)
    assert path.exists()
    assert json.loads(path.read_text("utf-8"))["decision"] == "PASS"


def test_gpu_preflight_blocks_when_the_device_reports_no_free_memory(tmp_path, monkeypatch) -> None:
    from slm_train.resources import gpu_probe

    monkeypatch.setattr(
        gpu_probe,
        "probe_gpu",
        lambda device="cuda:0": {
            "device": device,
            "free_memory_bytes": 0,
            "unavailable": gpu_probe.UNREADABLE,
        },
    )
    report = run_gpu_preflight(root=tmp_path, device="cuda:0")
    assert report["decision"] == "BLOCKED"
    assert "no free memory" in report["reason"]


def test_preflight_is_the_only_caller_of_the_gpu_probe() -> None:
    """AST check: only resources/preflight.py may import probe_gpu."""
    offenders: list[str] = []
    for path in V2_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("gpu_probe"):
                continue
            names = {alias.name for alias in node.names}
            if "probe_gpu" in names and path.name != "preflight.py":
                offenders.append(path.name + ":" + str(node.lineno))
    assert offenders == [], "only preflight.py may import probe_gpu: " + str(offenders)


def test_no_module_other_than_gpu_probe_imports_torch_cuda() -> None:
    offenders: list[str] = []
    for path in V2_PACKAGE.rglob("*.py"):
        if path.name == "gpu_probe.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "") == "torch.cuda":
                offenders.append(str(path))
            if isinstance(node, ast.Attribute) and node.attr.startswith("cuda"):
                offenders.append(str(path))
    assert offenders == [], "only gpu_probe.py may touch cuda: " + str(offenders)


def test_no_module_references_nvidia_smi() -> None:
    """A docstring may mention it; no executable string may name it."""
    offenders: list[str] = []
    for path in V2_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        holders = (
            ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
        )
        for node in ast.walk(tree):
            if isinstance(node, holders):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "nvidia-smi" in node.value
                and node.value not in docstrings
            ):
                offenders.append(path.name + ":" + str(node.lineno))
    assert offenders == [], "nvidia-smi is not part of the V2 pipeline: " + str(offenders)
