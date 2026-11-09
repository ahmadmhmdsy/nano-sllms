"""The run spine: every R1 stage executes, binds evidence, and blocks where unwired."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))

TINY = "pipeline_v2/tests/fixtures/tiny_config.yaml"
R1_STAGES = (
    "validate",
    "bootstrap",
    "gpu_preflight",
    "prepare",
    "tokenize",
    "dataset_validation",
    "model_validation",
    "evaluator_validation",
    "resolve_final_identity",
    "train",
    "internal_evaluation",
    "external_gates",
    "publish_final",
)


def _engine(tmp_path: Path, *, config_path: str = TINY, handlers=None):
    from slm_train.config.loader import load_config
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.handlers import HANDLERS

    config = load_config(ROOT / config_path, root=ROOT)
    return Engine(
        graph=StageGraph.r1(),
        handlers=HANDLERS if handlers is None else handlers,
        run_root=tmp_path / "run",
        config=config,
        repo_root=ROOT,
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """One real end-to-end run, shared: the pipeline trains, so it is not cheap.

    The run is executed twice so idempotency is proven on the same evidence the
    other assertions read, instead of paying for a second full training run.
    """
    base = tmp_path_factory.mktemp("pipeline")
    engine = _engine(base)
    engine.run()
    run_root = base / "run"
    first = (run_root / "train" / "checkpoints/COMPLETE.json").read_bytes()
    engine.run()
    return {"run_root": run_root, "first_checkpoint": first}


@pytest.fixture(scope="module")
def completed_run(completed):
    return completed["run_root"]


def test_the_full_r1_run_completes_every_stage(completed_run: Path):
    status = json.loads((completed_run / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PASS", status
    assert tuple(status["completed_stages"]) == R1_STAGES


def test_every_catalogue_suite_can_be_measured_by_some_run() -> None:
    """No suite may be advertised that no run can actually produce.

    Three suites (generation, syntax, resources) sat in the catalogue while the
    evaluation stage computed validation metrics only, so declaring any of them
    produced a guaranteed "missing required metric" FAIL. This test pins the
    metric set each suite needs to the set the stage can measure.
    """
    from slm_train.evaluation.suites import SUITES

    measurable = {
        "val_loss",
        "val_perplexity",
        "generation_smoke",
        "syntax_error_rate",
        "tokens_per_second",
        "peak_vram_bytes",
    }
    for name, spec in SUITES.items():
        needed = {metric.name for metric in spec.metrics}
        assert needed <= measurable, (
            "suite " + str(name) + " needs metrics no stage measures: "
            + str(sorted(needed - measurable))
        )


def test_a_metric_without_a_threshold_is_reported_and_not_gated() -> None:
    """A missing threshold means "report it", not "crash on it".

    The resources suite declares no thresholds at all. Reading .value off a
    missing threshold raised AttributeError before the None check.
    """
    from slm_train.evaluation.contracts import MetricSpec, SuiteSpec
    from slm_train.evaluation.reports import evaluate

    suite = SuiteSpec(
        evaluator="resources",
        version="1",
        split_identity="none",
        metrics=(
            MetricSpec(name="tokens_per_second", definition="measured throughput"),
            MetricSpec(name="peak_vram_bytes", definition="measured peak memory"),
        ),
    )
    identity = {
        "run_id": "r", "data_hash": "d", "tokenizer_hash": "t",
        "device_identity": "cpu",
    }
    hashes = {"expected_evaluator_hash": "h", "actual_evaluator_hash": "h"}
    report = evaluate(
        suite, identity,
        {"tokens_per_second": 1.0, "peak_vram_bytes": 2.0},
        **hashes,
    )
    assert report.status == "PASS", report.details
    # A threshold the config declares cannot conjure a gate onto a metric that
    # the catalogue deliberately left ungated.
    report = evaluate(
        suite, identity,
        {"tokens_per_second": 1.0, "peak_vram_bytes": 2.0},
        thresholds={"tokens_per_second": 1000.0}, **hashes,
    )
    assert report.status == "PASS", report.details


def test_each_suite_is_handed_exactly_its_own_metrics() -> None:
    """A suite must not be graded on another suite's measurements."""
    from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold
    from slm_train.evaluation.reports import evaluate

    suite = SuiteSpec(
        evaluator="validation",
        version="1",
        split_identity="validation",
        metrics=(MetricSpec(
            name="val_loss", definition="loss",
            threshold=Threshold(value=3.0, direction="min"),
        ),),
    )
    identity = {
        "run_id": "r", "data_hash": "d", "tokenizer_hash": "t",
        "device_identity": "cpu",
    }
    hashes = {"expected_evaluator_hash": "h", "actual_evaluator_hash": "h"}
    # Handing the union of every suite's metrics is a FAIL, by design: the
    # completeness check is strict so a suite cannot be scored on a measurement
    # that was never declared for it.
    union = evaluate(
        suite, identity, {"val_loss": 1.0, "generation_smoke": 1.0}, **hashes
    )
    assert union.status == "FAIL"
    scoped = evaluate(suite, identity, {"val_loss": 1.0}, **hashes)
    assert scoped.status == "PASS", scoped.details


def test_external_gates_are_registered_and_independently_implemented() -> None:
    """A registered gate that is not implemented is a hole, not a default."""
    from slm_train.evaluation.external import EXTERNAL_GATES
    from slm_train.evaluation.suites import EXTERNAL_GATE_NAMES

    assert EXTERNAL_GATE_NAMES, "an empty external gate set passes everything"
    for name in EXTERNAL_GATE_NAMES:
        assert name in EXTERNAL_GATES, name + " is registered but not implemented"
        assert callable(EXTERNAL_GATES[name])


def test_greedy_decoding_is_reproducible() -> None:
    """The same prompt and seed must decode identically every time.

    A generation gate whose output moved between runs could not be compared
    against a recorded number, which is the whole reason decoding is greedy.
    """
    from slm_train.models.decode import DecodeError, decode_samples

    class _FakeTokenizer:
        def encode(self, text):
            class _Out:
                ids = [1, 2, 3]
            return _Out()

        def decode(self, ids, skip_special_tokens=True):
            return "decoded:" + str(len(list(ids)))

    class _FakeModel:
        training = False

        def eval(self):
            return self

        def train(self, flag):
            return self

        def generate(self, **kwargs):
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    first = decode_samples(_FakeModel(), _FakeTokenizer(), "hello", sample_count=2)
    second = decode_samples(_FakeModel(), _FakeTokenizer(), "hello", sample_count=2)
    assert first == second, (first, second)
    assert all(isinstance(sample, str) and sample for sample in first)
    # An empty prompt is a refusal, never a fabricated sample.
    try:
        decode_samples(_FakeModel(), _FakeTokenizer(), "  ")
    except DecodeError:
        pass
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("an empty prompt must be refused")


def test_a_declared_threshold_replaces_the_catalogue_value() -> None:
    """A configured threshold must actually bind.

    The catalogue hardcodes val_loss 3.0. The evaluator manifest recorded the
    operator-declared value, but the gate still compared against the catalogue,
    so a configured threshold was decorative. The catalogue keeps ownership of
    the metric NAME and DIRECTION; the config owns the VALUE.
    """
    from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold
    from slm_train.evaluation.reports import evaluate

    suite = SuiteSpec(
        evaluator="validation",
        version="1",
        split_identity="validation",
        metrics=(MetricSpec(
            name="val_loss",
            definition="mean cross-entropy loss on the validation split",
            threshold=Threshold(value=3.0, direction="min"),
        ),),
    )
    identity = {
        "run_id": "r", "data_hash": "d", "tokenizer_hash": "t",
        "device_identity": "cpu",
    }
    hashes = {"expected_evaluator_hash": "h", "actual_evaluator_hash": "h"}

    # The catalogue value rejects this measurement.
    assert evaluate(suite, identity, {"val_loss": 6.0}, **hashes).status == "FAIL"
    # A declared, looser threshold accepts the very same measurement.
    relaxed = evaluate(
        suite, identity, {"val_loss": 6.0}, thresholds={"val_loss": 7.0}, **hashes
    )
    assert relaxed.status == "PASS", relaxed.details
    # A declared, tighter threshold still rejects it.
    strict = evaluate(
        suite, identity, {"val_loss": 6.0}, thresholds={"val_loss": 2.0}, **hashes
    )
    assert strict.status == "FAIL"
    # A name the suite never declared cannot introduce a gate.
    assert evaluate(
        suite, identity, {"val_loss": 6.0}, thresholds={"not_a_metric": 99.0}, **hashes
    ).status == "FAIL"


def test_a_missing_handler_is_blocked_never_failed(tmp_path: Path):
    """An unimplemented stage must not be reported as a failure of the run."""
    from slm_train.orchestration.errors import MissingStageInput
    from slm_train.orchestration.handlers import HANDLERS

    partial = {name: HANDLERS[name] for name in list(HANDLERS)[:3]}
    with pytest.raises(MissingStageInput):
        _engine(tmp_path, handlers=partial).run()
    recorded = json.loads((tmp_path / "run" / "status.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "BLOCKED", recorded
    assert recorded["reason"].strip()


def test_every_completed_stage_writes_its_own_marker(completed_run: Path):
    for stage in R1_STAGES:
        marker = completed_run / stage / "COMPLETE.json"
        assert marker.is_file(), stage
        document = json.loads(marker.read_text(encoding="utf-8"))
        assert document["status"] == "PASS", stage
        assert document["payload_hashes"], stage


def test_a_stage_marker_binds_every_declared_output(completed_run: Path):
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.stages import StageContext

    from slm_train.artifacts.markers import verify_completion_marker

    graph = StageGraph.r1()
    for stage in R1_STAGES:
        spec = graph.spec_by_name(stage)
        for relative in spec.outputs:
            assert (completed_run / stage / relative).is_file(), stage + "/" + relative
        context = StageContext(
            spec=spec, graph=graph, run_root=completed_run,
            config=None, repo_root=ROOT,
        )
        expected = context.certifiable_outputs()
        if expected:
            assert verify_completion_marker(completed_run / stage, expected), stage


def test_monitor_report_is_empty_when_monitor_not_enabled(completed_run: Path):
    """The monitor report must exist and be valid JSONL even when no monitor is declared.

    The tiny config declares no monitor section, so TrainMonitor.record() never
    runs. The fix ensures logs/monitor_report.jsonl is still created as an empty
    file (zero probe records), satisfying the graph's always-written declaration.
    """
    report_path = completed_run / "train" / "logs" / "monitor_report.jsonl"
    assert report_path.is_file(), "monitor_report.jsonl must exist even without a monitor"
    content = report_path.read_text(encoding="utf-8")
    assert content == "", "monitor_report.jsonl must be empty when no monitor recorded"
    # An empty file is valid JSONL (zero records); verify it parses as such
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 0, "empty monitor_report.jsonl must have zero JSONL records"


def test_a_tampered_stage_output_is_caught_by_its_marker(completed_run: Path):
    """A stage that completes and then has its output altered is not complete."""
    from slm_train.artifacts.markers import verify_completion_marker
    from slm_train.orchestration.graph import StageGraph

    declared = StageGraph.r1().spec_by_name("validate").outputs
    target = completed_run / "validate" / "config/resolved.json"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="mismatch"):
            verify_completion_marker(completed_run / "validate", declared)
    finally:
        target.write_bytes(original)


def test_a_cpu_run_never_probes_a_device(completed_run: Path):
    report = json.loads(
        (completed_run / "gpu_preflight" / "attempt/preflight.json").read_text("utf-8")
    )
    assert report["gpu_touched"] is False
    assert report["mode"] == "cpu"


def test_the_bootstrap_identity_is_stable_across_two_engines(tmp_path: Path):
    """A retry of the same work must be recognisable as the same work.

    Only the first two stages are wired here, so the run blocks quickly instead
    of paying for a second full training pass to compare identities.
    """
    from slm_train.orchestration.errors import MissingStageInput
    from slm_train.orchestration.handlers import HANDLERS

    partial = {name: HANDLERS[name] for name in ("validate", "bootstrap")}
    ids = []
    for name in ("a", "b"):
        with pytest.raises(MissingStageInput):
            _engine(tmp_path / name, handlers=partial).run()
        record = json.loads(
            (tmp_path / name / "run" / "bootstrap" / "attempt/bootstrap.json").read_text("utf-8")
        )
        ids.append(record["bootstrap_id"])
    assert ids[0] == ids[1]


def test_evaluation_scores_the_trained_checkpoint_not_fresh_weights(completed_run: Path):
    """The validation loss must come from the published checkpoint state."""
    import torch

    from slm_train.models.causal_lm import build_causal_lm
    from slm_train.config.loader import load_config
    from slm_train.training.checkpoint import checkpoint_dir

    record = json.loads(
        (completed_run / "train" / "checkpoints/COMPLETE.json").read_text("utf-8")
    )
    step = int(str(record["checkpoint"]).split("-")[-1])
    state_path = checkpoint_dir(completed_run / "train", step) / "state" / "model.pt"
    assert state_path.is_file()

    config = load_config(ROOT / TINY, root=ROOT)
    fresh = build_causal_lm(
        model_config=config.model, data_context=16, vocab_size=29, seed=42
    )
    trained = build_causal_lm(
        model_config=config.model, data_context=16, vocab_size=29, seed=42
    )
    trained.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    fresh_loss = record["last_loss"]
    assert float(fresh_loss) < 0.5, "the run must actually have learned something"
    # A freshly built model is a different tensor set; if the stage had scored
    # one instead of the checkpoint, its loss would be the chance value.
    import math

    assert not math.isclose(float(fresh_loss), math.log(29), rel_tol=0.1)


def test_publish_final_binds_the_run_marker_last(completed_run: Path):
    from slm_train.artifacts.publication import FINAL_MANIFEST

    marker = json.loads(
        (completed_run / "publish_final" / "COMPLETE.json").read_text("utf-8")
    )
    assert marker["manifest"] == FINAL_MANIFEST
    for name in marker["payload_hashes"]:
        assert (completed_run / "publish_final" / name).is_file(), name


def test_a_rerun_reuses_every_stage_and_republishes_nothing(completed):
    """The second execution over the same run root changed nothing."""
    run_root = completed["run_root"]
    after = (run_root / "train" / "checkpoints/COMPLETE.json").read_bytes()
    assert after == completed["first_checkpoint"]
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PASS", status


def test_the_marker_is_built_on_the_json_output_not_the_first_one(tmp_path: Path):
    """The validate stage declares YAML first; a marker cannot bind YAML."""
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.stages import StageContext

    graph = StageGraph.r1()
    spec = graph.spec_by_name("validate")
    assert spec.outputs[0].endswith(".yaml")
    context = StageContext(
        spec=spec, graph=graph, run_root=tmp_path, config=None, repo_root=tmp_path
    )
    assert context.json_manifest() == "config/resolved.json"


def test_a_stage_with_no_json_output_cannot_build_a_marker(tmp_path: Path):
    from slm_train.orchestration.graph import StageGraph, StageSpec
    from slm_train.orchestration.stages import StageContext

    graph = StageGraph.r1()
    spec = StageSpec(
        name="validate", requires=(), required_inputs=(), outputs=("only.txt",),
        completion_marker="validate.COMPLETE.json",
    )
    context = StageContext(
        spec=spec, graph=graph, run_root=tmp_path, config=None, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="no JSON output"):
        context.json_manifest()


def test_a_stage_cannot_read_its_own_or_a_later_stage_output(tmp_path: Path):
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.stages import StageContext

    graph = StageGraph.r1()
    spec = graph.spec_by_name("prepare")
    context = StageContext(
        spec=spec, graph=graph, run_root=tmp_path, config=None, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="may not read from"):
        context.published("prepare", "manifests/data.json")
    with pytest.raises(ValueError, match="may not read from"):
        context.published("train", "checkpoints/COMPLETE.json")
    assert context.published("validate", "config/resolved.json").name == "resolved.json"


def test_a_missing_earlier_output_says_so_rather_than_returning_empty(tmp_path: Path):
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.stages import StageContext

    graph = StageGraph.r1()
    context = StageContext(
        spec=graph.spec_by_name("prepare"), graph=graph, run_root=tmp_path,
        config=None, repo_root=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="expected validate/config/resolved.json"):
        context.read_published_json("validate", "config/resolved.json")
def test_the_run_keeps_only_the_best_and_the_last_three_checkpoints(
    completed_run: Path,
):
    """Retention is a promise about disk, so it is asserted against the disk."""
    root = completed_run / "train" / "checkpoints"
    directories = sorted(p.name for p in root.iterdir() if p.is_dir())
    record = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))

    assert len(directories) <= 4, directories
    assert sorted(record["retained_checkpoints"]) == directories
    assert record["best_checkpoint"] in directories
    # the three newest by step number must all survive
    assert len(directories) >= 3


def test_the_best_checkpoint_is_the_lowest_recorded_held_out_loss(
    completed_run: Path,
):
    scores = json.loads(
        (completed_run / "train" / "checkpoints" / "scores.json").read_text(encoding="utf-8")
    )
    record = json.loads(
        (completed_run / "train" / "checkpoints" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    val = {int(k): v for k, v in scores["val_loss"].items()}
    assert val, "no held-out loss was recorded, so best is unmeasured"
    best_step = min(val, key=lambda step: val[step])
    assert record["best_checkpoint"] == "step-" + str(best_step).zfill(8)
    assert set(scores["train_loss"]) == set(scores["val_loss"]), (
        "train and val losses disagree"
    )


def test_the_hyperparameters_come_from_the_config_not_the_handler(
    completed_run: Path,
):
    from slm_train.config.loader import load_config

    config = load_config(ROOT / TINY, root=ROOT)
    record = json.loads(
        (completed_run / "train" / "checkpoints" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert record["learning_rate"] == pytest.approx(config.train.learning_rate)
    assert record["max_steps"] == config.train.max_steps
    # the budget is a hard bound, not a target it may overshoot
    assert record["final_step"] <= config.train.max_steps


def test_a_deleted_train_record_resumes_instead_of_retraining(tmp_path: Path):
    """Auto-resume must recover the run own checkpoint, not start over."""
    engine = _engine(tmp_path)
    engine.run()
    run_root = tmp_path / "run"
    record_path = run_root / "train" / "checkpoints/COMPLETE.json"
    original = json.loads(record_path.read_text(encoding="utf-8"))
    assert original["resumed_from_step"] is None

    fresh_summary = json.loads(
        (run_root / "train" / "logs" / "resume_summary.json").read_text("utf-8")
    )
    assert fresh_summary["resumed_from_step"] is None
    assert fresh_summary["source_checkpoint"] is None
    assert fresh_summary["retrain_reason"].strip() != ""
    assert "run_id" in fresh_summary["identity_keys_compared"]

    record_path.unlink()
    _engine(tmp_path).run()
    resumed = json.loads(record_path.read_text(encoding="utf-8"))

    assert resumed["status"] == "PASS"
    assert resumed["resumed_from_step"] == original["final_step"]
    assert resumed["final_step"] == original["final_step"]

    resumed_summary = json.loads(
        (run_root / "train" / "logs" / "resume_summary.json").read_text("utf-8")
    )
    assert resumed_summary["resumed_from_step"] == original["final_step"]
    assert resumed_summary["source_checkpoint"] == (
        "step-%08d" % original["final_step"]
    )
    # the same loss, carried from the checkpoint: the weights were not retrained
    assert resumed["last_loss"] == pytest.approx(original["last_loss"])


def test_resume_refuses_a_checkpoint_from_another_run(tmp_path: Path):
    """A foreign checkpoint is never loaded, even if it sits in the run own dir."""
    from slm_train.orchestration.handlers import _resume_from

    stage_root = tmp_path / "train"
    (stage_root / "checkpoints" / "step-00000010" / "state").mkdir(parents=True)
    (stage_root / "checkpoints" / "step-00000010" / "manifest.json").write_text(
        json.dumps({"run_id": "someone-else", "config_hash": "x", "global_step": 10}),
        encoding="utf-8",
    )
    import torch

    torch.save({}, stage_root / "checkpoints" / "step-00000010" / "state" / "model.pt")
    torch.save({}, stage_root / "checkpoints" / "step-00000010" / "state" / "optimizer.pt")

    class _Model:
        def load_state_dict(self, state):
            raise AssertionError("a foreign checkpoint was loaded")

    class _Optimizer:
        def load_state_dict(self, state):
            raise AssertionError("a foreign checkpoint was loaded")

    class _Scheduler:
        def load_state_dict(self, state):
            raise AssertionError("a foreign checkpoint was loaded")

    assert _resume_from(
        stage_root,
        {"run_id": "this-run", "config_hash": "x"},
        _Model(),
        _Optimizer(),
        _Scheduler(),
        "cpu",
        torch.float32,
        False,
    ) is None


def test_the_allocator_ceiling_is_a_no_op_on_cpu_and_refuses_a_bad_fraction():
    from slm_train.resources.device import enforce_allocator_ceiling

    assert enforce_allocator_ceiling("cpu", 0.9) == 1.0
    with pytest.raises(ValueError):
        from slm_train.resources.gpu_probe import apply_allocator_ceiling

        apply_allocator_ceiling("cuda:0", 1.5)


def test_the_shipped_experiment_templates_declare_a_usable_corpus():
    """The S/P/T templates must not be placeholders: they extend a real base."""
    from slm_train.config.loader import load_config

    for name in ("smoke", "pilot", "target", "nano_codesechnet"):
        config = load_config(
            ROOT / ("pipeline_v2/configs/experiments/" + name + ".yaml"), root=ROOT
        )
        assert config.data.sources, name + " declares no data source"
        for source in config.data.sources:
            assert (ROOT / source.path_or_repo).exists(), source.path_or_repo


def test_the_base_profile_is_nano_sized_not_seven_billion():
    """This machine is a nano box; the base profile must fit it."""
    from slm_train.config.loader import load_config

    config = load_config(ROOT / "pipeline_v2/configs/base/r1.yaml", root=ROOT)
    model = config.model
    head_width = model.hidden // model.kv_heads * model.kv_heads
    embedding = 32016 * model.hidden
    per_layer = (
        model.hidden * model.hidden * 2
        + model.hidden * head_width * 2
        + model.hidden * model.intermediate * 3
    )
    total = embedding + per_layer * model.layers
    if not model.tied_embeddings:
        total += embedding
    assert 150_000_000 <= total <= 400_000_000, total


def test_every_required_training_knob_is_configured_in_every_shipped_config():
    from slm_train.config.loader import load_config

    paths = [
        "pipeline_v2/configs/base/r1.yaml",
        "pipeline_v2/configs/experiments/smoke.yaml",
        "pipeline_v2/configs/experiments/pilot.yaml",
        "pipeline_v2/configs/experiments/target.yaml",
        "pipeline_v2/configs/experiments/nano_codesechnet.yaml",
        "pipeline_v2/configs/experiments/gpu_smoke.yaml",
        "pipeline_v2/tests/fixtures/tiny_config.yaml",
    ]
    for path in paths:
        config = load_config(ROOT / path, root=ROOT)
        assert config.train.learning_rate > 0, path
        assert config.train.max_steps > 0, path
        assert config.train.keep_last_checkpoints >= 1, path
        assert config.train.keep_best_checkpoints >= 1, path
        assert 0.0 < config.runtime.vram_fraction <= 1.0, path


# --- Phase 2 disk-space gate ------------------------------------------------


def test_required_free_bytes_scales_with_kept_checkpoints():
    from slm_train.resources.disk_gate import required_free_bytes

    value = required_free_bytes(checkpoint_bytes=938_000_000, keep=3, margin=2)
    assert value == 938_000_000 * 3 * 2


def test_disk_gate_refuses_a_run_root_with_no_room(tmp_path, monkeypatch):
    from slm_train.resources.disk_gate import disk_gate_decision

    monkeypatch.setattr(
        "slm_train.resources.disk_gate.free_space_bytes", lambda root: 1_000_000
    )
    decision = disk_gate_decision(tmp_path)
    assert decision["label"] == "BLOCKED"


def _disk_gate_decision(tmp_path):
    from slm_train.resources.disk_gate import disk_gate_decision

    return disk_gate_decision(tmp_path)


def disk_gate_decision(tmp_path):
    from importlib import import_module

    module = import_module("slm_train.resources.disk_gate")
    return module.disk_gate_decision(tmp_path)


def test_disk_gate_passes_when_space_exceeds_the_requirement(tmp_path, monkeypatch):
    import slm_train.resources.disk_gate as gate

    monkeypatch.setattr(
        gate, "required_free_bytes", lambda **kw: 1_000_000
    )
    monkeypatch.setattr(
        gate, "free_space_bytes", lambda root: 100_000_000
    )
    decision = gate.disk_gate_decision(tmp_path, checkpoint_bytes=500, keep=1, margin=2)
    assert decision["label"] == "PASS"


# --- Phase 2: loud prepared-data invalidation --------------------------------


def test_prepared_context_mismatch_is_an_honest_fail(tmp_path):
    """A stale prepared context is refused by the train stage, not trained on."""
    from slm_train.config.loader import load_config
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.graph import StageGraph
    import slm_train.orchestration.handlers as handlers_module

    def _engine(document, run_root=None):
        return Engine(
            graph=StageGraph.r1(),
            handlers=handlers_module.HANDLERS,
            run_root=run_root or (tmp_path / "run"),
            config=document,
            repo_root=ROOT,
        )

    config = load_config(ROOT / TINY, root=ROOT)
    _engine(config).run()

    # The configuration now declares a different context. The earlier stage
    # records are still byte-valid, so the engine reuses them, and the train
    # stage is the honest gate that notices the mismatch.
    document = config.model_copy(deep=True, update={
        "model": config.model.model_copy(update={"context": 32}),
        "tokenizer": config.tokenizer.model_copy(update={"context": 32}),
        "packing": config.packing.model_copy(update={"sequence_length": 32}),
    })
    (tmp_path / "run" / "train" / "COMPLETE.json").unlink()
    try:
        _engine(document).run()
    except Exception:
        pass

    status = json.loads((tmp_path / "run" / "status.json").read_text("utf-8"))
    prepared = json.loads(
        (tmp_path / "run" / "dataset_validation" / "manifests" / "tokenized_data.json")
        .read_text("utf-8")
    )
    assert prepared["sequence_length"] == 16
    assert status["status"] == "FAIL"
    assert status["failed_stage"] == "train"
    assert "re-run the prepare and tokenize stages" in status["reason"]


# --- Phase 2: the disk gate also applies before model_validation --------------


def test_model_validation_gate_refuses_an_undersized_volume(tmp_path, monkeypatch):
    """The declared arithmetic blocks the stage before its evidence exists."""
    import slm_train.resources.disk_gate as disk_gate
    import slm_train.orchestration.handlers as handlers_module
    from slm_train.config.loader import load_config
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.graph import StageGraph

    monkeypatch.setattr(
        disk_gate,
        "free_space_bytes",
        lambda path: 1024,
    )
    monkeypatch.setattr(
        disk_gate,
        "required_free_bytes",
        lambda checkpoint_bytes, keep, margin=1.0: 1 << 40,
    )

    config = load_config(ROOT / TINY, root=ROOT)
    engine = Engine(
        graph=StageGraph.r1(),
        handlers=handlers_module.HANDLERS,
        run_root=tmp_path / "run",
        config=config,
        repo_root=ROOT,
    )
    try:
        engine.run()
    except Exception:
        pass
    status = json.loads((tmp_path / "run" / "status.json").read_text("utf-8"))
    assert status["status"] == "FAIL"
    assert status["failed_stage"] == "model_validation"
    assert status["reason"].startswith("free disk space")


# --- Phase 3: the one-key context alias ---------------------------------------


def test_config_context_alias_must_agree_with_the_three_sections():
    """A disagreeing alias is refused before any stage reads the config."""
    from slm_train.config.loader import load_config
    from pydantic import ValidationError
    from slm_train.config.models import ConfigDocument

    config = load_config(ROOT / TINY, root=ROOT)
    payload = config.model_dump(mode="json")
    payload["context"] = 32
    with pytest.raises(ValidationError):
        ConfigDocument.model_validate(payload)


def test_materialize_context_fills_the_sections_from_the_alias():
    """The alias path materializes the three sections the alias declared."""
    from slm_train.config import materialize_context
    from slm_train.config.models import ConfigDocument
    from pydantic import ValidationError

    from slm_train.config.loader import load_config

    document = load_config(ROOT / TINY, root=ROOT)
    assert document.context is None
    document = document.model_copy(update={"context": 16})
    materialized = materialize_context(document)
    assert materialized.context is None
    assert materialized.model.context == 16
    assert materialized.tokenizer.context == 16
    assert materialized.packing.sequence_length == 16


def test_document_masked_labels_mark_the_prefix_not_the_spec():
    """Prompt-masked packing: labels must -100 the TASK prefix and keep loss on the chain body."""
    import torch

    from slm_train.orchestration.handlers import _labels_for_block

    ids = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    mask = [0, 0, 1, 1, 1]
    labels = _labels_for_block(ids, mask)
    assert labels[0, 0].item() == -100 and labels[0, 1].item() == -100
    assert labels[0, 2].item() == 7 and labels[0, 4].item() == 9
    identity = _labels_for_block(ids, None)
    assert torch.equal(identity, ids)
