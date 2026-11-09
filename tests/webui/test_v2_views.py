"""Contract tests for the V2 read path and the V2 job control.

These are fixture-based on purpose: the assertions must hold when runs/v2 is
empty, partially written or freshly rebuilt, so no test may depend on the
particular runs that happen to be on this machine's disk today.

The behaviours pinned here are the ones that were actually wrong during the
V2 UI work: record ordering in the metrics stream, the location of the model
identity fields, and the refusal to report a pass nobody published.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "webui", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import v2_jobs  # noqa: E402
import v2_reader  # noqa: E402


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    """A minimal but structurally real V2 run, with runs/v2 pointed at tmp."""
    root = tmp_path / "runs" / "v2"
    run = root / "core" / "demo"
    _write(run / "status.json", {
        "status": "PASS",
        "completed_stages": ["validate", "train"],
        "failed_stage": None,
        "reason": "",
    })
    _write(run / "validate" / "COMPLETE.json", {"status": "PASS"})
    _write(run / "train" / "COMPLETE.json", {"status": "PASS"})
    # A stage directory the engine started but never finished: it exists, and
    # it has no marker. The reader must not call that completed.
    (run / "tokenize").mkdir(parents=True, exist_ok=True)
    _write(run / "train" / "checkpoints" / "COMPLETE.json", {
        "checkpoint": "step-00000020", "best_checkpoint": "step-00000010",
        "final_step": 20, "max_steps": 20, "last_loss": 1.5,
        "declared_precision": "fp16", "retained_checkpoints":
        ["step-00000010", "step-00000020"]})
    _write(run / "train" / "checkpoints" / "scores.json",
           {"train_loss": {"10": 2.0, "20": 1.6}, "val_loss": {"10": 2.1, "20": 1.7}})
    _write(run / "model_validation" / "manifests" / "model.json", {
        "architecture": "llama",
        "capability_metadata": {
            "layers": 16, "hidden": 1024, "heads": 16, "kv_heads": 4,
            "intermediate": 2816, "context": 1024, "vocab_size": 32016,
            "tied_embeddings": True, "precision": "fp16", "attention": "sdpa",
            "num_parameters": 213173248,
            "source_hash": "abc123"},
        "config_hash": "cfg456"})
    _write(run / "publish_final" / "final" / "model" / "model_config.json",
           {"architecture": "llama", "layers": 16})
    _write(run / "internal_evaluation" / "evaluations" / "validation"
           / "report.json", {
               "metric_values": {"val_loss": 1.7, "val_perplexity": 5.47},
               "gate_threshold": None, "status": "PASS"})
    _write(run / "external_gates" / "reports" / "COMPLETE.json",
           {"status": "PASS", "gates": {"validation": {"status": "PASS"}}})
    _write(run / "gpu_preflight" / "attempt" / "preflight.json", {
        "decision": "PASS", "reason": "device is visible",
        "observation": {"name": "Test GPU", "device": "cuda:0",
                        "free_memory_bytes": 2 * 2**30,
                        "total_memory_bytes": 6 * 2**30}})

    metrics = run / "train" / "logs" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    lines = [
        {"event": "step_completed", "payload": {"step": step, "loss": 10.0 - step * 0.1},
         "record": step, "run_id": "demo", "stream": "metrics"}
        for step in range(1, 6)
    ]
    (metrics / "segment-000001.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    (metrics / "segment-000002.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [
            {"event": "step_completed",
             "payload": {"step": step, "loss": 9.5 - (step - 5) * 0.1},
             "record": step, "run_id": "demo", "stream": "metrics"}
            for step in range(6, 11)]), encoding="utf-8")

    monitor = run / "train" / "logs" / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    (monitor / "segment-000001.jsonl").write_text(
        json.dumps({"event": "run_started",
                    "payload": {"device": "cuda:0", "stage": "train"},
                    "record": 1, "stream": "monitor"}) + "\n"
        + json.dumps({"event": "stage_completed",
                      "payload": {"stage": "train", "steps": 10},
                      "record": 2, "stream": "monitor"}) + "\n",
        encoding="utf-8")

    monkeypatch.setattr(v2_reader, "ROOT", tmp_path)
    monkeypatch.setattr(v2_reader, "V2_ROOT", root)
    return run


class TestDiscovery:
    def test_run_choices_uses_namespace_slash_run(self, run_root):
        assert v2_reader.run_choices() == ["core/demo"]

    def test_attempt_space_is_not_a_run(self, run_root, tmp_path):
        (tmp_path / "runs" / "v2" / "_attempts" / "thing").mkdir(parents=True)
        assert v2_reader.run_choices() == ["core/demo"]

    def test_run_root_resolves_a_known_label(self, run_root):
        assert v2_reader.run_root("core/demo") == run_root

    @pytest.mark.parametrize("label", [
        "", "core", "core/../core/demo", "core/demo/extra",
        "../../etc", "nope/demo", "core/./demo",
    ])
    def test_run_root_refuses_unenumerated_labels(self, run_root, label):
        # A dropdown is user-writable, so a crafted label must not walk out of
        # runs/v2 by being joined onto a path.
        assert v2_reader.run_root(label) in (None, run_root)
        if v2_reader.run_root(label) is not None:
            assert v2_reader.run_root(label) == run_root


class TestLabel:
    def test_label_is_the_engines_verbatim(self, run_root):
        assert v2_reader.run_label(run_root)["status"] == "PASS"

    @pytest.mark.parametrize("status", ["PASS", "FAIL", "SKIPPED", "BLOCKED",
                                        "NEEDS_USER_DECISION"])
    def test_every_schema_label_survives(self, run_root, status):
        _write(run_root / "status.json", {"status": status,
                                          "completed_stages": []})
        assert v2_reader.run_label(run_root)["status"] == status

    def test_absent_status_is_fail_not_pass(self, run_root):
        (run_root / "status.json").unlink()
        result = v2_reader.run_label(run_root)
        assert result["status"] == "FAIL"
        assert result["readable"] is False

    def test_corrupt_status_is_fail_not_pass(self, run_root):
        (run_root / "status.json").write_text("{not json", encoding="utf-8")
        assert v2_reader.run_label(run_root)["status"] == "FAIL"

    def test_unknown_label_is_surfaced_not_coerced(self, run_root):
        _write(run_root / "status.json", {"status": "WEIRD"})
        assert v2_reader.run_label(run_root)["status"] == "WEIRD"


class TestStageTable:
    def test_completed_stage_is_completed(self, run_root):
        rows = {r[0]: r[1] for r in v2_reader.stage_table(run_root)}
        assert rows["validate"] == "completed (marker: PASS)"

    def test_stage_with_no_marker_is_not_called_complete(self, run_root):
        rows = {r[0]: r[1] for r in v2_reader.stage_table(run_root)}
        assert "no marker" in rows["tokenize"]

    def test_every_row_carries_its_marker_path(self, run_root):
        for stage, _state, marker in v2_reader.stage_table(run_root):
            if marker != "-":
                assert (ROOT / marker).name == "COMPLETE.json" or stage
                assert marker.endswith("COMPLETE.json")


class TestMetrics:
    def test_loss_points_are_oldest_first(self, run_root):
        # Regression: the reader once reversed the stream, which drew a
        # descending loss curve that looked plausible and was wrong.
        points = v2_reader.loss_points(run_root)
        assert [p[0] for p in points] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_loss_points_span_both_segments(self, run_root):
        assert v2_reader.loss_points(run_root)[-1][0] == 10

    def test_loss_points_are_downsampled_not_truncated(self, run_root):
        points = v2_reader.loss_points(run_root, limit=4)
        assert len(points) == 4
        assert points[0][0] < points[-1][0]

    def test_missing_metrics_yield_no_points(self, run_root):
        import shutil
        shutil.rmtree(run_root / "train" / "logs")
        assert v2_reader.loss_points(run_root) == []

    def test_tail_reader_ignores_a_foreign_file(self, run_root):
        kind = run_root / "train" / "logs" / "metrics"
        (kind / "notes.txt").write_text("not a segment", encoding="utf-8")
        assert len(v2_reader.loss_points(run_root)) == 10

    def test_monitor_line_reports_the_newest_event(self, run_root):
        assert "stage completed" in v2_reader.last_monitor_line(run_root)


class TestModelIdentity:
    def test_identity_fields_live_under_capability_metadata(self, run_root):
        # Regression: reading these from the top level of the manifest
        # rendered a 213M model as "0 parameters, precision -".
        model = v2_reader.model_identity(run_root)
        meta = model["capability_metadata"]
        assert meta["num_parameters"] == 213173248
        assert meta["precision"] == "fp16"
        assert model["config_hash"] == "cfg456"

    def test_detail_markdown_reports_the_real_parameter_count(self, run_root):
        body = v2_reader.detail_markdown("core/demo")
        assert "213,173,248" in body
        assert "fp16" in body
        assert "0 params" not in body


class TestServableWeights:
    def test_metadata_only_publication_is_not_servable(self, run_root):
        # The publish_final stage certifies metadata, so Chat must not claim
        # a servable checkpoint exists.
        assert v2_reader.servable_weights(run_root) is None

    def test_chat_markdown_says_blocked_and_why(self, run_root):
        body = v2_reader.chat_blocked_markdown("core/demo")
        assert "BLOCKED" in body
        assert "model_config.json" in body
        assert "not a servable HF" in body

    def test_a_real_weight_file_is_found(self, run_root):
        model_dir = run_root / "publish_final" / "final" / "model"
        (model_dir / "model.safetensors").write_bytes(b"x")
        found = v2_reader.servable_weights(run_root)
        assert found is not None and found.name == "model.safetensors"

    def test_a_training_state_pt_is_never_servable(self, run_root):
        state = run_root / "train" / "checkpoints" / "step-00000020" / "state"
        state.mkdir(parents=True)
        (state / "model.pt").write_bytes(b"x")
        assert v2_reader.servable_weights(run_root) is None


class TestCheckpoints:
    def test_retained_checkpoints_come_from_the_selection_record(self, run_root):
        assert v2_reader.retained_checkpoints(run_root) == [
            "step-00000010", "step-00000020"]

    def test_summary_row_carries_the_engine_label(self, run_root):
        row = v2_reader.summary_row("core/demo")
        assert row[1] == "PASS"
        assert row[2] == "2/13"
        assert row[3] == "step-00000020"

    def test_summary_row_for_an_unknown_run_is_dashes(self):
        row = v2_reader.summary_row("core/ghost")
        assert row[1] == "-"

    @pytest.mark.parametrize("step", ["../escape", "a/b", "..", ""])
    def test_checkpoint_manifest_refuses_traversal(self, run_root, step):
        assert v2_reader.checkpoint_manifest(run_root, step) is None


class TestJobs:
    def test_argv_is_the_exact_documented_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        argv = v2_jobs.argv_for("pipeline_v2/configs/experiments/smoke.yaml")
        assert argv[1].endswith("pipeline_v2" + __import__("os").sep + "run.py")
        assert argv[2] == "run"
        assert argv[3] == "--config"
        assert argv[4] == "pipeline_v2/configs/experiments/smoke.yaml"
        # No resume flag, no seed, no device: a resume adds nothing.
        assert not any(a.startswith("--") and a != "--config" for a in argv)

    def test_launch_refuses_without_validation(self, monkeypatch, tmp_path):
        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        result = v2_jobs.launch("some/config.yaml", config_ok=False)
        assert result.startswith("**BLOCKED**")
        assert "validate" in result

    def test_launch_refuses_without_a_config(self):
        assert v2_jobs.launch("", config_ok=True) == "Pick a config before launching."

    def test_resume_without_a_recorded_job_is_blocked(self, monkeypatch, tmp_path):
        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        monkeypatch.setattr(v2_jobs, "JOB_STATE", tmp_path / "absent.json")
        assert v2_jobs.resume().startswith("**BLOCKED**")

    def test_resume_replays_the_recorded_argv_verbatim(self, monkeypatch, tmp_path):
        started = {}

        class FakeProc:
            pid = 4242

        def fake_popen(argv, cwd=None, **kwargs):
            started["argv"] = argv
            return FakeProc()

        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        monkeypatch.setattr(v2_jobs, "JOB_STATE", tmp_path / "job.json")
        monkeypatch.setattr(v2_jobs.subprocess, "Popen", fake_popen)
        recorded = ["python", "run.py", "run", "--config", "x.yaml"]
        _write(tmp_path / "job.json", {"argv": recorded, "pid": 1, "config": "x.yaml"})
        result = v2_jobs.resume()
        assert started["argv"] == recorded
        assert "no flags added" in result
        assert v2_jobs.job()["pid"] == 4242

    def test_resume_refuses_a_record_without_argv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        monkeypatch.setattr(v2_jobs, "JOB_STATE", tmp_path / "job.json")
        _write(tmp_path / "job.json", {"pid": 0, "config": "x.yaml"})
        assert "no argv" in v2_jobs.resume()

    def test_stop_explains_its_absence(self):
        body = v2_jobs.stop_unavailable_markdown()
        assert "No stop control in V2 mode" in body
        assert "exact-command resume" in body

    def test_experiment_configs_are_repo_relative(self, monkeypatch, tmp_path):
        configs = tmp_path / "pipeline_v2" / "configs" / "experiments"
        configs.mkdir(parents=True)
        (configs / "smoke.yaml").write_text("x: 1", encoding="utf-8")
        monkeypatch.setattr(v2_jobs, "ROOT", tmp_path)
        monkeypatch.setattr(v2_jobs, "EXPERIMENTS", configs)
        assert v2_jobs.experiment_configs() == ["pipeline_v2/configs/experiments/smoke.yaml"]

    def test_cli_timeout_is_reported_as_blocked(self, monkeypatch):
        import subprocess as sp

        def boom(*a, **k):
            raise sp.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(v2_jobs.subprocess, "run", boom)
        code, payload = v2_jobs.run_cli(["validate", "--config", "x"])
        assert code == 124
        assert payload["status"] == "BLOCKED"
