"""E-005 monitor unit tests: pure logic, GPU-free, no tense-vocab.

Reads train.monitor / train.early_stop as a plain mapping (the same shape the
parsed config document exposes) and exercises the monitor contract:
cadence, multi-metric records, best tracking, underfit/overfit signals,
early stop with min-steps gate, and the validity-rising veto.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.evaluation.probes import probe_result, syntax_issues
from slm_train.training.monitor import TrainMonitor, TrainMonitorConfig


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "monitor_report.jsonl", tmp_path / "monitor_summary.json"


def test_cadence_fraction():
    cfg = TrainMonitorConfig({"monitor": {"enabled": True, "eval_fraction": 0.1}})
    assert cfg.cadence_steps(600) == 60
    assert cfg.cadence_steps(1800) == 180
    assert cfg.minimum_steps_for_stop(600) == 300


def test_records_multi_metric_and_best_tracking(paths):
    cfg = TrainMonitorConfig({"monitor": {"enabled": True}})
    monitor = TrainMonitor(cfg, 600, paths[0], paths[1], clock=lambda: 0.0)
    monitor.record(60, 2.0, [2.0])
    monitor.record(120, 1.8, [1.8])
    rec = monitor.record(180, 1.9, [1.9])
    assert rec["val_best"] == 1.8
    assert rec["probes_since_val_best"] == 1
    assert paths[0].read_text(encoding="utf-8").strip().count("\n") == 2


def test_early_stop_after_patience_and_min_steps(paths):
    cfg = TrainMonitorConfig({
        "monitor": {"enabled": True},
        "early_stop": {"enabled": True, "patience_evals": 2, "min_steps_fraction": 0.5},
    })
    monitor = TrainMonitor(cfg, 600, paths[0], paths[1], clock=lambda: 0.0)
    for step, val in [(60, 2.0), (120, 1.8), (180, 1.7), (240, 1.6), (300, 1.55), (360, 1.56), (420, 1.57), (480, 1.58)]:
        monitor.record(step, val, [val])
    assert monitor.stop_reason is not None
    assert monitor.best_val_step == 300
    assert monitor.summarize()["early_stop"]


def test_min_steps_gate_protects_a_short_budget(paths):
    cfg = TrainMonitorConfig({
        "monitor": {"enabled": True},
        "early_stop": {"enabled": True, "patience_evals": 1, "min_steps_fraction": 0.5},
    })
    monitor = TrainMonitor(cfg, 300, paths[0], paths[1], clock=lambda: 0.0)  # under the round 400 floor
    for step in (30, 60, 90, 120):
        monitor.record(step, 2.0, [2.0])
    assert monitor.stop_reason is None


def test_validity_rising_vetoes_stop(paths):
    cfg = TrainMonitorConfig({
        "monitor": {"enabled": True, "probe_tasks_file": "tasks.jsonl", "probe_tasks": 4},
        "early_stop": {"enabled": True, "patience_evals": 1, "min_steps_fraction": 0.1},
    })
    monitor = TrainMonitor(cfg, 600, paths[0], paths[1], task_probe=lambda: {"valid": 0.8}, clock=lambda: 0.0)
    monitor.record(100, 1.5)
    assert monitor.stop_reason is None  # validity still rising
    monitor.record(200, 1.5)
    assert monitor.stop_reason is not None  # validity stalled + val stalled


def test_overfit_signal_flags_val_regression(paths):
    cfg = TrainMonitorConfig({"monitor": {"enabled": True, "overfit_patience": 2}})
    monitor = TrainMonitor(cfg, 600, paths[0], paths[1], clock=lambda: 0.0)
    monitor.record(60, 1.0)
    rec = monitor.record(120, 1.2)
    assert rec["overfit_signal"] is False  # one probe past best - under patience
    rec = monitor.record(180, 1.1)
    assert rec["overfit_signal"] is True  # >= patience probes past the 5% band
    summary = monitor.summarize()
    assert summary["overfit_signal_any"] is True


def test_probe_failure_is_observed_not_fatal(paths):
    cfg = TrainMonitorConfig({"monitor": {"enabled": True, "probe_tasks_file": "tasks.jsonl", "probe_tasks": 4}})
    monitor = TrainMonitor(cfg, 600, paths[0], paths[1], task_probe=lambda: (_ for _ in ()).throw(RuntimeError("boom")), clock=lambda: 0.0)
    rec = monitor.record(60, 1.5)
    assert "boom" in rec["probe_error"]
    assert monitor.stop_reason is None


def test_syntax_probe_contract():
    good = "UI1 T ui.chain\nTASK paint plates for dash radius delta\nROOT {\"arch\": \"dash\"}\nSTAX {\"view\": \"v1\"}\nEL 0"
    assert syntax_issues(good) == []
    assert syntax_issues("TASK broken")[0] == "no-UI1-header"
    assert "STAX-not-json" in syntax_issues(
        good.replace("STAX {\"view\": \"v1\"}", "STAX not json")
    )


def test_probe_result_aggregates():
    rows = [
        {"arch": "dashboard_metrics", "chain": "UI1 T ui.chain\nTASK rusted tile counts rank across cohort spark\nROOT {}\nSTAX {}\nEL 0", "task": "rusted tile counts rank across cohort spark"},
        {"arch": "todo_list", "chain": "TASK broken", "task": "broken"},
    ]
    out = probe_result(rows)
    assert out["valid"] == 0.5 and out["n"] == 2
    assert "eos_rate" in out and "arch_match" in out
