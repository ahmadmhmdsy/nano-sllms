"""Milestone 2.2: runtime.worker_count is removed from the config schema.

`worker_count` was declared on RuntimeConfig but never read by any stage
(remediation plan Task 3.3, Option A). A declared-but-unconsumed field is a
lie in a governed-config pipeline, so the schema must not declare it and a
config that still carries it must be refused.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_train.config.loader import ConfigError, load_config
from slm_train.config.models import RuntimeConfig

ROOT = Path(__file__).resolve().parents[3]


def test_worker_count_is_not_a_runtime_config_field() -> None:
    assert "worker_count" not in RuntimeConfig.model_fields


def test_runtime_config_refuses_worker_count() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(
            device="cuda:0",
            precision="fp16",
            launch_profile="nano-200m",
            vram_fraction=0.75,
            worker_count=0,
        )


def test_config_with_runtime_worker_count_is_rejected(tmp_path: Path) -> None:
    """A full config document declaring runtime.worker_count fails to load."""
    base = (ROOT / "pipeline_v2/configs/base/r1.yaml").read_text(encoding="utf-8")
    text = base.replace(
        "runtime:\n  device: cuda:0",
        "runtime:\n  device: cuda:0\n  worker_count: 0",
    )
    assert "worker_count: 0" in text
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((ConfigError, ValidationError)):
        load_config(path)


def test_generated_json_schema_has_no_worker_count() -> None:
    schema = RuntimeConfig.model_json_schema()
    assert "worker_count" not in schema.get("properties", {})
