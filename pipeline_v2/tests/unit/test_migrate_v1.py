"""Tests for the migrate-v1 command: it must produce valid V2 configs.

A migrated config is a translation, not an adoption. It must be loadable and
pass schema validation, not just be a YAML file that looks plausible. These
tests pin that contract: the migration parses the V1 file with a real YAML
parser, maps only fields with a real V2 meaning, and emits a document that
the V2 loader accepts.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from slm_train.adapters.legacy_v1 import migrate_v1_config
from slm_train.config.loader import load_config

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BASE_CONFIG = _ROOT / "pipeline_v2" / "configs" / "base" / "r1.yaml"

_V1_SAMPLE = (
    "name: pilot\n"
    "seed: 42\n"
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
    "  tie_embeddings: true\n"
    "data:\n"
    "  min_chars: 200\n"
    "  val_fraction: 0.2\n"
    "  dedupe: true\n"
    "some_unknown_v1_knob: 3\n"
)


def _setup_config_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal config tree so extends: ../base/r1.yaml resolves."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "r1.yaml").write_text(
        _BASE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


def test_migrate_v1_produces_a_valid_config(tmp_path):
    """The migrated config must load and pass schema validation."""
    exp_dir = _setup_config_tree(tmp_path)
    source = exp_dir / "v1.yaml"
    source.write_text(_V1_SAMPLE, encoding="utf-8")
    target = exp_dir / "v2.yaml"

    result = migrate_v1_config(source, target)

    # The output must be loadable and pass schema validation
    config = load_config(target, root=tmp_path)
    assert config.schema_version == 2
    assert config.run.name == "pilot"
    assert config.run.seed == 42


def test_migrate_v1_maps_nested_v1_fields_to_v2(tmp_path):
    """Nested V1 blocks (train.lr, model.layers) must map to real V2 fields."""
    exp_dir = _setup_config_tree(tmp_path)
    source = exp_dir / "v1.yaml"
    source.write_text(_V1_SAMPLE, encoding="utf-8")
    target = exp_dir / "v2.yaml"

    migrate_v1_config(source, target)
    config = load_config(target, root=tmp_path)

    assert config.train.learning_rate == pytest.approx(0.0004)
    assert config.train.batch_size == 2
    assert config.train.gradient_accumulation == 8
    assert config.train.max_steps == 1000
    assert config.train.max_grad_norm == pytest.approx(1.0)
    assert config.train.scheduler == "cosine"
    assert config.train.optimizer == "adamw"
    assert config.train.warmup_steps == 10
    assert config.train.weight_decay == pytest.approx(0.1)
    assert config.model.layers == 2
    assert config.model.hidden == 64
    assert config.model.heads == 4
    assert config.model.kv_heads == 2
    assert config.model.intermediate == 128
    assert config.model.context == 64
    assert config.model.tied_embeddings is True
    assert config.data.validation_fraction == pytest.approx(0.2)


def test_migrate_v1_reports_unmigrated_fields(tmp_path):
    """Fields with no V2 equivalent must be reported, not silently dropped."""
    exp_dir = _setup_config_tree(tmp_path)
    source = exp_dir / "v1.yaml"
    source.write_text(_V1_SAMPLE, encoding="utf-8")
    target = exp_dir / "v2.yaml"

    result = migrate_v1_config(source, target)

    assert any("some_unknown_v1_knob" in note for note in result.notes)
    assert any("train.lr" in field for field in result.migrated_fields)


def test_migrate_v1_preserves_the_source(tmp_path):
    """The V1 input must be byte-identical after migration."""
    exp_dir = _setup_config_tree(tmp_path)
    source = exp_dir / "v1.yaml"
    source.write_text(_V1_SAMPLE, encoding="utf-8")
    target = exp_dir / "v2.yaml"

    before = source.read_bytes()
    result = migrate_v1_config(source, target)
    after = source.read_bytes()

    assert before == after
    assert result.source_sha256_before == result.source_sha256_after


def test_migrate_v1_produces_schema_known_keys_only(tmp_path):
    """The emitted document must only contain keys the V2 schema knows."""
    from slm_train.config.models import ConfigDocument

    exp_dir = _setup_config_tree(tmp_path)
    source = exp_dir / "v1.yaml"
    source.write_text(_V1_SAMPLE, encoding="utf-8")
    target = exp_dir / "v2.yaml"

    migrate_v1_config(source, target)
    written = yaml.safe_load(target.read_text(encoding="utf-8"))

    known_top = set(ConfigDocument.model_fields) | {
        "extends", "release", "capabilities", "schema_version"}
    for key in written:
        assert key in known_top, "unknown key: " + str(key)
