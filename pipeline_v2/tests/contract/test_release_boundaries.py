"""Release 1 boundaries: what V2 must never do, checked on every run."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V2_SOURCES = [ROOT / "pipeline_v2" / "run.py", *sorted((ROOT / "pipeline_v2" / "slm_train").rglob("*.py"))]

#: Modules whose presence in a V2 source would mean V1 is reachable at runtime.
FORBIDDEN_ROOTS = frozenset({"src", "scripts", "laya"})


def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_v2_source_list_is_not_empty():
    """An empty list would make every boundary test pass for the wrong reason."""
    assert len(V2_SOURCES) > 50, len(V2_SOURCES)


def test_v2_has_no_v1_runtime_imports():
    for path in V2_SOURCES:
        modules = _imported_modules(path)
        for module in modules:
            for forbidden in FORBIDDEN_ROOTS:
                assert not module.startswith(forbidden + "."), (path, module)
                assert module != forbidden, (path, module)


def test_no_v2_source_dynamically_loads_a_module():
    """A dynamic LOAD is the boundary crossing. Reading metadata is not.

    doctor.py legitimately calls importlib.util.find_spec to report whether
    slm_train is installed, and importlib.metadata for its version. Neither
    executes a module, so neither is a V1 boundary crossing. Banning the whole
    importlib namespace would forbid those checks without forbidding anything
    real, so the ban is on the loading calls specifically.
    """
    forbidden_calls = ("importlib.import_module", "__import__", "importlib.reload")
    for path in V2_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                raise AssertionError("dynamic import in " + str(path))
            if isinstance(node.func, ast.Attribute):
                assert ast.unparse(node.func) not in forbidden_calls, (path, node.func)


def test_no_v2_source_reads_a_v1_file_at_runtime():
    """Reading a V1 artifact by path is as much a boundary crossing as importing it."""
    offenders = []
    for path in V2_SOURCES:
        text = path.read_text(encoding="utf-8")
        for marker in ("/src/", "\\src\\", "src/train", "scripts/train"):
            if marker in text:
                offenders.append((path.name, marker))
    assert not offenders, offenders


def test_acceptance_requires_explicit_gpu_window():
    text = (ROOT / "pipeline_v2/docs/acceptance.md").read_text(encoding="utf-8")
    assert "user-opened GPU window" in text
    assert "preflight --gpu" in text
    assert "zero additional flags" in text


def test_the_runbook_states_the_oom_policy_rather_than_improvising_it():
    text = (ROOT / "pipeline_v2/docs/acceptance.md").read_text(encoding="utf-8")
    assert "BLOCKED" in text
    assert "NEEDS_USER_DECISION" in text
    assert "No foreign process is ever killed" in text


def test_the_runbook_claims_only_what_the_published_runs_ran():
    """The runbook cites exact published run statuses, and keeps the still
    unexecuted promotion sequences labelled NEEDS_USER_DECISION rather than
    PASS."""
    text = (ROOT / "pipeline_v2/docs/acceptance.md").read_text(encoding="utf-8")
    assert "NEEDS_USER_DECISION" in text
    for run in ("gpu_smoke", "nano_codesechnet", "micro_full",
                "gpu_smoke-failed-fp16-step2", "smoke"):
        assert "runs/v2/core/" + run + "/status.json" in text
    assert "has NOT been executed" not in text


def test_doctor_never_probes_a_device():
    from slm_train.resources.doctor import run_doctor

    result = run_doctor(ROOT)
    assert result.get("gpu_touched") is False


def test_only_preflight_gpu_probes_a_device():
    """No other V2 module may reach for a device on its own."""
    offenders = []
    for path in sorted((ROOT / "pipeline_v2" / "slm_train").rglob("*.py")):
        if path.name in ("gpu_probe.py", "preflight.py", "device.py", "doctor.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if "torch.cuda" in text or "pynvml" in text or "nvidia-smi" in text:
            offenders.append(path.name)
    assert not offenders, offenders


def test_every_experiment_template_validates_and_stays_in_r1():
    from slm_train.cli import main

    for name in ("smoke", "pilot", "target"):
        path = ROOT / "pipeline_v2" / "configs" / "experiments" / (name + ".yaml")
        assert path.is_file(), path
        assert main(["validate", "--config", str(path)]) == 0, name


def test_plan_runs_end_to_end_and_writes_nothing(capsys):
    """plan is the command users reach for first; a crash here is a bad first run."""
    import json

    from slm_train.cli import main

    config = ROOT / "pipeline_v2" / "configs" / "experiments" / "smoke.yaml"
    assert main(["plan", "--config", str(config)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "PASS"
    assert document["writes"] is False
    assert document["acquires_device"] is False
    assert len(document["stages"]) == 13
    for stage in document["stages"]:
        assert stage["name"] and isinstance(stage["outputs"], list)


def test_every_stage_declares_a_completion_marker(capsys):
    import json

    from slm_train.cli import main

    config = ROOT / "pipeline_v2" / "configs" / "experiments" / "smoke.yaml"
    main(["plan", "--config", str(config)])
    document = json.loads(capsys.readouterr().out)
    for stage in document["stages"]:
        assert stage["completion_marker"], stage["name"]


def test_a_promotion_block_must_carry_all_six_evidence_references():
    """A half-filled promotion is worse than none: it looks accepted."""
    from pydantic import ValidationError

    from slm_train.config.models import PromotionReference

    complete = {
        "predecessor_stage": "S",
        "predecessor_run_id": "run-1",
        "checkpoint": "checkpoints/step-00000010",
        "data_manifest": "manifests/data.json",
        "evaluator_manifest": "manifests/evaluator.json",
        "gate_report": "reports/smoke-gate.json",
    }
    assert PromotionReference(**complete).predecessor_stage == "S"
    for missing in complete:
        partial = {k: v for k, v in complete.items() if k != missing}
        try:
            PromotionReference(**partial)
            raise AssertionError("a promotion without " + missing + " was accepted")
        except ValidationError:
            pass
