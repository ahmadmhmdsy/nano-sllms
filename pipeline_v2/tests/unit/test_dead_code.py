"""Milestone 3.1: HFCausalLMTrainer dead-code removal tests.

The V2 pipeline trains through the hand-rolled loop in
`slm_train/orchestration/handlers.py`. The `HFCausalLMTrainer` class in
`slm_train/training/hf_causal_lm.py` was a dead abstraction: only `prepare()`
was implemented and every other method raised `PipelineError`. Milestone 3.1
removes it while keeping the utilities that are actually used
(`resolve_output_dir`, `load_state_dict_safely`).
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "pipeline_v2"
    / "slm_train"
    / "training"
    / "hf_causal_lm.py"
)

# Symbols that were dead code and must not reappear in the module.
DEAD_SYMBOLS = ("HFCausalLMTrainer", "build_training_arguments")


def _module_source() -> str:
    assert MODULE_PATH.is_file(), f"module missing: {MODULE_PATH}"
    return MODULE_PATH.read_text(encoding="utf-8")


def _module_tree() -> ast.Module:
    return ast.parse(_module_source(), filename=str(MODULE_PATH))


def _top_level_definitions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_hf_causal_lm_trainer_not_importable():
    """The dead HFCausalLMTrainer class must not be importable."""
    module = importlib.import_module("slm_train.training.hf_causal_lm")
    assert not hasattr(module, "HFCausalLMTrainer")
    with pytest.raises(ImportError):
        from slm_train.training.hf_causal_lm import HFCausalLMTrainer  # noqa: F401


def test_module_defines_no_dead_symbols():
    """No dead class or function may be defined in the module."""
    defined = _top_level_definitions(_module_tree())
    for symbol in DEAD_SYMBOLS:
        assert symbol not in defined, f"dead symbol still defined: {symbol}"


def test_module_has_no_unused_imports():
    """Every import in the module must be used; no dead imports left behind."""
    tree = _module_tree()
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                imported[alias.asname or alias.name] = node.lineno
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for name, lineno in imported.items():
        assert name in used, f"unused import {name!r} at line {lineno}"


def test_all_matches_definitions():
    """__all__ must list exactly the module's public definitions."""
    tree = _module_tree()
    defined = _top_level_definitions(tree)
    all_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    all_names = [element.value for element in node.value.elts]
    assert all_names is not None, "module must declare __all__"
    for name in all_names:
        assert name in defined, f"__all__ lists undefined name: {name}"
    for name in defined:
        assert name in all_names, f"public definition missing from __all__: {name}"


def test_surviving_utilities_still_work():
    """Code that is actually used must survive the dead-code removal."""
    from slm_train.errors import PipelineError
    from slm_train.training.hf_causal_lm import (
        load_state_dict_safely,
        resolve_output_dir,
    )

    assert resolve_output_dir("checkpoints/r1/run-1") == "checkpoints/r1/run-1"
    with pytest.raises(PipelineError):
        resolve_output_dir(None)
    assert callable(load_state_dict_safely)


# ---------------------------------------------------------------------------
# Milestone 3.2: heartbeat dead-code removal tests.
#
# The heartbeat module (slm_train/monitoring/heartbeat.py) was never called
# by any pipeline code. The gap-analysis confirmed it: "Heartbeat monitor is
# never called" and "monitoring/heartbeat.py is currently unreachable".
# Milestone 3.2 deletes the entire module.
# ---------------------------------------------------------------------------

HEARTBEAT_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "pipeline_v2"
    / "slm_train"
    / "monitoring"
    / "heartbeat.py"
)

# Symbols that were defined in the dead heartbeat module.
HEARTBEAT_DEAD_SYMBOLS = (
    "HealthSnapshot",
    "observe_liveness",
    "utc_now",
    "write_heartbeat",
    "read_heartbeat",
    "heartbeat_decision",
)


def test_heartbeat_module_file_does_not_exist():
    """The dead heartbeat module file must be deleted entirely."""
    assert not HEARTBEAT_MODULE_PATH.is_file(), (
        f"dead heartbeat module still exists: {HEARTBEAT_MODULE_PATH}"
    )


def test_heartbeat_module_not_importable():
    """The dead heartbeat module must not be importable."""
    with pytest.raises(ImportError):
        importlib.import_module("slm_train.monitoring.heartbeat")


def test_heartbeat_dead_symbols_not_importable():
    """No symbol from the dead heartbeat module may be importable."""
    for symbol in HEARTBEAT_DEAD_SYMBOLS:
        with pytest.raises(ImportError):
            exec(f"from slm_train.monitoring.heartbeat import {symbol}")


def test_no_pipeline_code_references_heartbeat():
    """No pipeline source file may reference the dead heartbeat module."""
    pipeline_root = (
        Path(__file__).resolve().parents[3]
        / "pipeline_v2"
        / "slm_train"
    )
    for py_file in pipeline_root.rglob("*.py"):
        if py_file == HEARTBEAT_MODULE_PATH:
            continue
        source = py_file.read_text(encoding="utf-8")
        assert "monitoring.heartbeat" not in source, (
            f"pipeline file still references heartbeat: {py_file}"
        )
        assert "from slm_train.monitoring import heartbeat" not in source, (
            f"pipeline file still imports heartbeat: {py_file}"
        )
