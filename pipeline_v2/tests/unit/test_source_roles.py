"""Milestone 1.3: held-out source roles are refused at the config schema.

A source declared with role "test" or "external-gate" is held-out evaluation
data. Allowing those roles in the config schema would leak held-out data into
training, so the schema must refuse them outright (remediation plan Task 2.4,
Option A). Only "train" and "validation" are legal source roles.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_train.config.models import FreshnessPolicy, SourceDeclaration

ROOT = Path(__file__).resolve().parents[3]


def _make_source(role: str) -> SourceDeclaration:
    return SourceDeclaration(
        source_id="held",
        kind="local",
        path_or_repo="x/y",
        revision=None,
        file_set=("f.jsonl",),
        role=role,
        license="MIT",
        expected_rows=0,
        expected_tokens=0,
        freshness_policy=FreshnessPolicy(max_age="none", supersession="none"),
    )


def test_test_role_is_refused() -> None:
    with pytest.raises(ValidationError):
        _make_source("test")


def test_external_gate_role_is_refused() -> None:
    with pytest.raises(ValidationError):
        _make_source("external-gate")


def test_train_role_is_accepted() -> None:
    assert _make_source("train").role == "train"


def test_validation_role_is_accepted() -> None:
    assert _make_source("validation").role == "validation"


def test_held_out_roles_are_refused_through_the_config_loader(tmp_path: Path) -> None:
    """A full config document declaring a held-out role fails to load."""
    from slm_train.config.loader import ConfigError, load_config

    base = (ROOT / "pipeline_v2/configs/base/r1.yaml").read_text(encoding="utf-8")
    # The base profile declares role: train; swap in a held-out role.
    text = base.replace("role: train", "role: test")
    assert "role: test" in text
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((ConfigError, ValidationError)):
        load_config(path)
