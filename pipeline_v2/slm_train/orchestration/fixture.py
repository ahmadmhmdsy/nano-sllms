"""A one-step CPU fixture that exercises the REAL production path.

This is not a mock of the pipeline. It uses the same contracts, the same atomic
store, the same log index, the same checkpoint publication, the same evaluator,
the same gates, and the same final publication as a real run. The ONLY thing
replaced is the training loop: one deterministic step stands in for the loop
that would otherwise need a device and hours of compute.

That is the point. A fixture that stubbed the publication code would prove
nothing about publication. This one proves that the whole chain actually
produces a bound COMPLETE.json, on CPU, with no network, in under a second.
"""
from __future__ import annotations

from pathlib import Path

from slm_train.artifacts.publication import publish_final, write_run_complete
from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.contracts.hash import sha256_bytes
from slm_train.contracts.run import RunIdentityManifest
from slm_train.evaluation.reports import evaluate, write_report
from slm_train.evaluation.suites import suite_for
from slm_train.monitoring.bus import EventBus
from slm_train.orchestration.stages import RunResult
from slm_train.training.checkpoint import publish_checkpoint
from slm_train.training.state import CheckpointState, TrainingCursor

__all__ = ["FIXTURE_STAGES", "run_cpu_fixture"]

FIXTURE_STAGES = ("prepare", "train", "evaluate", "publish")

RUN_ID = "fixture-run"
LOG_INDEX = "logs/log-manifest.json"
RUN_IDENTITY = "manifests/run_identity.json"


def _digest(label: str) -> str:
    """A stable, obviously-synthetic digest. Not a measurement of anything."""
    return sha256_bytes(("fixture:" + label).encode("utf-8"))


def _state() -> CheckpointState:
    return CheckpointState(
        cursor=TrainingCursor(1, 0, 1, 1),
        model="model.pt",
        optimizer="optimizer.pt",
        scheduler="scheduler.pt",
        scaler="scaler.pt",
        global_rng="global_rng.pt",
        sampler="sampler.pt",
        dataloader_generator="dataloader_generator.pt",
        worker_rng=("worker_rng_0.pt",),
        early_stop="early_stop.json",
        evaluation="evaluation.json",
        lifecycle="lifecycle.json",
    )


def _run_identity() -> RunIdentityManifest:
    return RunIdentityManifest(
        schema_version=1,
        run_id=RUN_ID,
        experiment_id="fixture",
        config_path="pipeline_v2/tests/fixtures/tiny_config.yaml",
        config_hash=_digest("config"),
        data_hash=_digest("data"),
        tokenizer_hash=_digest("tokenizer"),
        tokenized_hash=_digest("tokenized"),
        model_hash=_digest("model"),
        evaluator_hash=_digest("evaluator"),
        code_hash=_digest("code"),
        device_identity={"kind": "cpu", "index": "0"},
        working_directory=".",
        evidence_root=RUN_ID,
    )


def run_cpu_fixture(run_root) -> RunResult:
    """Prepare, take one step, evaluate, publish, and bind COMPLETE.json."""
    from slm_train.artifacts.manifest import write_manifest

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    completed = []

    # 1. identity
    identity = _run_identity()
    write_manifest(root, RUN_IDENTITY, identity.model_dump(mode="json"))
    completed.append("prepare")

    # 2. events -> immutable log segments and the log index
    bus = EventBus(root, RUN_ID)
    bus.emit("run_started", {"stage": "fixture"})
    bus.emit("step_completed", {"step": 1, "loss": 2.5, "tokens": 8})
    bus.emit("checkpoint_completed", {"checkpoint": 1})
    bus.flush()
    completed.append("train")

    # 3. one deterministic step, published through the real checkpoint code
    state = _state()
    payloads = {role: ("fixture-" + role).encode("utf-8") for role in state.files()}
    checkpoint_manifest = CheckpointManifest(
        schema_version=1,
        run_id=RUN_ID,
        config_hash=_digest("config"),
        tokenizer_hash=_digest("tokenizer"),
        model_hash=_digest("model"),
        code_hash=_digest("code"),
        global_step=1,
        epoch=1.0,
        files=(),
        complete=True,
    )
    publish_checkpoint(root, 1, state, payloads, checkpoint_manifest)

    # 4. evaluate through the real gates and write the real report
    suite = suite_for("validation")
    report = evaluate(
        suite,
        {
            "run_id": RUN_ID,
            "data_hash": _digest("data"),
            "tokenizer_hash": _digest("tokenizer"),
            "device_identity": "cpu",
            "checkpoint_identity": "step-00000001",
            "seed": 42,
        },
        {"val_loss": 2.5, "val_perplexity": 12.182493960703473},
        expected_evaluator_hash=_digest("evaluator"),
        actual_evaluator_hash=_digest("evaluator"),
    )
    if report.status != "PASS":
        return RunResult(
            status="FAIL",
            completed_stages=tuple(completed),
            failed_stage="evaluate",
            reason=str(report.details.get("reason", "evaluation did not pass")),
        )
    report_relative = write_report(root, "eval-1", report)
    completed.append("evaluate")

    # 5. final publication, then the marker that binds everything.
    # The log index is the real one the event bus published; it is bound as
    # evidence, never rewritten.
    if not (root / LOG_INDEX).is_file():
        raise RuntimeError("the event bus must publish the log index: " + LOG_INDEX)
    bound = {RUN_IDENTITY: (root / RUN_IDENTITY).read_bytes(), LOG_INDEX: (root / LOG_INDEX).read_bytes()}
    required = {
        name: sha256_bytes(payload)
        for name, payload in bound.items()
    }
    required[report_relative] = sha256_bytes((root / report_relative).read_bytes())
    publish_final(
        root,
        run_id=RUN_ID,
        selected_checkpoint="step-00000001",
        model_payloads={"model.safetensors": b"fixture-weights"},
        tokenizer_payloads={"tokenizer.json": b"fixture-tokens"},
        required_hashes=required,
    )
    to_bind = dict(bound)
    to_bind[report_relative] = (root / report_relative).read_bytes()
    write_run_complete(root, to_bind)
    completed.append("publish")
    return RunResult(status="PASS", completed_stages=tuple(completed))
