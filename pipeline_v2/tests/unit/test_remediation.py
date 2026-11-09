"""Tests for the P0 correctness items of the remediation plan.

Every test here was written RED against the pre-fix behaviour: the config
declared a thing and the run did not do it. These tests pin the CORRECT
behaviour: the loop honours gradient_accumulation, the scheduler is real, the
optimizer is the declared one, normalization certified is normalization
applied, and a migrated config is a valid V2 override document.
"""
from __future__ import annotations

import pytest


class _TrainStub:
    def __init__(self, scheduler="cosine", warmup_steps=0, weight_decay=0.0,
                 optimizer="adamw"):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer = optimizer


# --- optimizer / scheduler factories --------------------------------------


def _build(scheduler, warmup=0, max_steps=10, lr=1.0):
    return _real_build(scheduler, warmup, max_steps, lr)


def _real_build(scheduler, warmup, max_steps, lr):
    import torch
    from slm_train.training.schedule import build_optimizer, build_scheduler

    parameter = torch.nn.Parameter(torch.zeros(1))
    config = _TrainStub(scheduler=scheduler, warmup_steps=warmup)
    optimizer = build_optimizer(config, [parameter], lr)
    schedule = build_scheduler(config, optimizer, max_steps)
    return schedule, optimizer


def test_the_scheduler_ramps_through_warmup_then_decays():
    schedule, optimizer = _build("cosine", warmup=2, max_steps=10, lr=1.0)
    lrs = []
    for _ in range(12):
        optimizer.step()
        lrs.append(schedule.get_last_lr()[0])
        schedule.step()
    assert lrs[0] == pytest.approx(1.0 / 2)
    assert lrs[1] == pytest.approx(1.0)
    after = lrs[3:]
    for earlier, later in zip(after, after[1:]):
        assert later <= earlier + 1e-9
    assert after[-1] < 0.01


def test_a_constant_scheduler_stays_at_the_base_learning_rate():
    schedule, optimizer = _build("constant", warmup=0, max_steps=5, lr=0.25)
    for _ in range(5):
        optimizer.step()
        assert schedule.get_last_lr()[0] == pytest.approx(0.25)
        schedule.step()


def test_the_optimizer_honours_weight_decay():
    import torch
    from slm_train.training.schedule import build_optimizer

    optimizer = build_optimizer(
        _TrainStub("cosine", 0, 0.1, "adamw"),
        [torch.nn.Parameter(torch.zeros(1))], 1e-2)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert type(optimizer) is torch.optim.AdamW


def test_an_undeclared_optimizer_or_scheduler_is_refused():
    from slm_train.training.schedule import build_optimizer, build_scheduler

    with pytest.raises(ValueError, match="unsupported train.optimizer"):
        build_optimizer(_TrainStub(optimizer="sgd"), [], 1e-3)
    with pytest.raises(ValueError, match="unsupported train.scheduler"):
        build_scheduler(_TrainStub(scheduler="crazy"), None, 10)


def test_the_config_schema_knows_warmup_and_weight_decay():
    from slm_train.config.models import TrainingConfig

    fields = TrainingConfig.model_fields
    assert "warmup_steps" in fields
    assert "weight_decay" in fields


# --- normalization ----------------------------------------------------------


def test_nfkc_is_applied_and_certified_as_applied():
    import unicodedata
    from slm_train.data.dedupe import apply_normalization

    text, applied = apply_normalization("\ufb01ne", "unicode_nfkc")
    assert applied == "unicode_nfkc"
    assert text == " ".join(unicodedata.normalize("NFKC", "\ufb01ne").split())


def test_nfkc_collapse_makes_nfkc_equivalents_identical():
    from slm_train.data.dedupe import apply_normalization, text_key

    first, _ = apply_normalization("\ufb01ne", "unicode_nfkc")
    second, _ = apply_normalization("fine", "unicode_nfkc")
    assert text_key(first) == text_key(second)


def test_an_undeclared_normalization_is_refused_before_any_row_is_read():
    from slm_train.data.dedupe import _probe_normalization

    with pytest.raises(ValueError, match="unsupported data.normalization"):
        _probe_normalization("stripe_case")


# --- filters -----------------------------------------------------------------


def test_declared_filters_reach_the_real_min_and_max_characters():
    from slm_train.data.prepare import _validate_options

    resolved = _validate_options({
        "dataset_id": "d", "generation_id": "g", "split_salt": 0,
        "validation_fraction": 0.1,
        "filters": ["min_chars:200", "max_chars:5000"],
    })
    assert resolved["min_chars"] == 200
    assert resolved["max_chars"] == 5000


def test_an_unknown_filter_name_is_refused_in_options():
    from slm_train.data.prepare import _validate_options

    with pytest.raises(ValueError, match="unknown data filter"):
        _validate_options({
            "dataset_id": "d", "generation_id": "g", "split_salt": 0,
            "validation_fraction": 0.1,
            "filters": ["brain_chars:3"],
        })


# --- held-out roles ----------------------------------------------------------


def test_test_and_external_gate_roles_force_the_held_out_split():
    from slm_train.data.split import role_forces_held_out

    assert role_forces_held_out("test") is True
    assert role_forces_held_out("external-gate") is True
    assert role_forces_held_out("train") is False


# --- config schema guards ----------------------------------------------------


def test_warmup_larger_than_max_steps_is_refused_by_the_schema():
    from pydantic import ValidationError
    from slm_train.config.models import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig(
            optimizer="adamw", scheduler="cosine", learning_rate=1e-3,
            max_steps=10, batch_size=1, gradient_accumulation=1,
            max_grad_norm=1.0, checkpoint_every_steps=1, evaluation_cadence=1,
            checkpoint_policy="last", keep_last_checkpoints=1,
            keep_best_checkpoints=0, warmup_steps=10,
        )


def test_the_config_refuses_an_unknown_optimizer_or_scheduler():
    from pydantic import ValidationError
    from slm_train.config.models import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig(
            optimizer="sgd", scheduler="cosine", learning_rate=1e-3,
            max_steps=10, batch_size=1, gradient_accumulation=1,
            max_grad_norm=1.0, checkpoint_every_steps=1, evaluation_cadence=1,
            checkpoint_policy="best", keep_last_checkpoints=1,
            keep_best_checkpoints=0,
        )
    with pytest.raises(ValidationError):
        TrainingConfig(
            optimizer="adamw", scheduler="polynomial", learning_rate=1e-3,
            max_steps=10, batch_size=1, gradient_accumulation=1,
            max_grad_norm=1.0, checkpoint_every_steps=1, evaluation_cadence=1,
            checkpoint_policy="best", keep_last_checkpoints=1,
            keep_best_checkpoints=0,
        )


# --- migrate-v1 --------------------------------------------------------------


def _migrate(tmp_path, text):
    from slm_train.adapters.legacy_v1 import migrate_v1_config

    source = tmp_path / "v1.yaml"
    source.write_text(text, encoding="utf-8")
    return migrate_v1_config(source, tmp_path / "v2.yaml")


_V1_SAMPLE = (
    "name: pilot\n"
    "train:\n"
    "  lr: 0.0004\n"
    "  batch: 2\n"
    "  accum: 8\n"
    "  scheduler: cosine\n"
    "  optim: adamw_torch\n"
    "  warmup_steps: 10\n"
    "  weight_decay: 0.1\n"
    "  max_steps: 1000\n"
    "  max_grad_norm: 1.0\n"
    "  seed: 42\n"
    "model:\n"
    "  layers: 2\n"
    "  hidden: 64\n"
    "  heads: 4\n"
    "  kv_heads: 2\n"
    "  ffn: 128\n"
    "  ctx: 64\n"
    "data:\n"
    "  min_chars: 200\n"
    "  val_fraction: 0.2\n"
    "  dedupe: true\n"
    "some_unknown_v1_knob: 3\n"
)


def test_a_nested_v1_train_block_maps_to_v2_fields_without_v1_code():
    import yaml
    from slm_train.adapters.legacy_v1 import migrate_v1_config

    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = pathlib.Path(directory) / "v1.yaml"
        source.write_text(_V1_SAMPLE, encoding="utf-8")
        result = migrate_v1_config(source, pathlib.Path(directory) / "v2.yaml")
        written = yaml.safe_load(result.target.read_text(encoding="utf-8"))
    assert written["schema_version"] == 2
    assert written["train"]["learning_rate"] == pytest.approx(0.0004)
    assert written["train"]["gradient_accumulation"] == 8
    assert written["train"]["optimizer"] == "adamw"
    assert written["train"]["warmup_steps"] == 10
    assert written["run"]["seed"] == 42
    assert any("train.lr" in note for note in result.migrated_fields)


def test_a_migrated_document_only_emits_schema_known_keys():
    import pathlib
    import tempfile

    import yaml
    from slm_train.adapters.legacy_v1 import migrate_v1_config
    from slm_train.config.models import ConfigDocument

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        source = root / "v1.yaml"
        source.write_text(_V1_SAMPLE, encoding="utf-8")
        result = migrate_v1_config(source, root / "v2.yaml")
        written = yaml.safe_load(result.target.read_text(encoding="utf-8"))
    known_top = set(ConfigDocument.model_fields) | {
        "extends", "release", "capabilities", "schema_version"}
    for key in written:
        assert key in known_top, "unknown key: " + str(key)
    assert "provenance" not in written
    assert any("some_unknown_v1_knob" in note for note in result.notes)


def test_a_test_role_source_declaration_is_refused_by_the_schema():
    # Milestone 1.3: held-out roles are refused at the config schema level,
    # not merely certified in the preparation lane. A config that declares
    # role: test would leak held-out data into training.
    from pydantic import ValidationError

    from slm_train.config.models import FreshnessPolicy, SourceDeclaration

    with pytest.raises(ValidationError):
        SourceDeclaration(
            source_id="held", kind="local", path_or_repo="x/y", revision=None,
            file_set=("f.jsonl",), role="test", license="MIT",
            expected_rows=0, expected_tokens=0,
            freshness_policy=FreshnessPolicy(max_age="none", supersession="none"),
        )


# --- Phase 2: policy selection, lease, multi-run status ----------


def test_a_declared_best_policy_selects_the_lowest_val_loss_step():
    import pathlib
    import tempfile

    from slm_train.training.selection import (
        select_policy_checkpoint, validate_policy,
    )

    with tempfile.TemporaryDirectory() as directory:
        stage_root = pathlib.Path(directory) / "train"
        checkpoints = stage_root / "checkpoints"
        for step in (10, 20, 30):
            (checkpoints / ("step-" + str(step).zfill(8))).mkdir(parents=True)
            (checkpoints / ("step-" + str(step).zfill(8)) / "manifest.json").write_text("{}", encoding="utf-8")
        atomic = checkpoints / "scores.json"
        atomic.write_text(
            '{"val_loss": {"10": 2.5, "20": 1.1, "30": 1.4}}', encoding="utf-8")
        assert select_policy_checkpoint(stage_root, "best", 30) == 20
        assert select_policy_checkpoint(stage_root, "last", 30) == 30
        assert validate_policy("best") == "best"
        with pytest.raises(ValueError, match="unsupported train.checkpoint_policy"):
            validate_policy("highest")


def test_the_lease_refuses_recovery_when_the_owner_start_time_violates():
    from slm_train.resources.lease import _owner_start_matches, process_start_time_of

    assert process_start_time_of(-1) is None
    # an unparseable recorded time cannot be proven to match: not False
    assert _owner_start_matches(1, "garbage") is None
