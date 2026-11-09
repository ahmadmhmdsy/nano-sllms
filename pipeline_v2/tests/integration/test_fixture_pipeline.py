"""The one-step CPU fixture, end to end, through the real production code.

If this test passes, the whole chain works on CPU: identity, event segments, the
log index, a real checkpoint with its completion marker, a real evaluator and
gate, a real final publication, and a COMPLETE.json that binds all of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))


def test_tiny_fixture_prepares_trains_evaluates_and_publishes(tmp_path: Path) -> None:
    from slm_train.orchestration.fixture import run_cpu_fixture

    result = run_cpu_fixture(tmp_path)
    assert result.status == "PASS", result.reason
    assert (tmp_path / "manifests" / "run_identity.json").is_file()
    assert (tmp_path / "manifests" / "final_publication.json").is_file()
    assert (tmp_path / "COMPLETE.json").is_file()
    complete = json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["status"] == "PASS"
    assert "logs/log-manifest.json" in complete["payload_hashes"]


def test_the_fixture_publishes_a_complete_checkpoint(tmp_path: Path) -> None:
    from slm_train.orchestration.fixture import run_cpu_fixture
    from slm_train.training.selection import select_checkpoint

    run_cpu_fixture(tmp_path)
    selected = select_checkpoint(tmp_path)
    assert selected.number == 1
    assert selected.reason == "verified"


def test_the_fixture_writes_no_absolute_path_into_any_published_artifact(tmp_path: Path) -> None:
    from slm_train.orchestration.fixture import run_cpu_fixture

    run_cpu_fixture(tmp_path)
    for path in sorted(tmp_path.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text, path


def test_the_fixture_needs_no_device_and_no_network(tmp_path: Path) -> None:
    """The fixture must be importable and runnable on a machine with no GPU."""
    import slm_train.orchestration.fixture as fixture

    source = Path(fixture.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "cuda", "nvidia", "requests", "huggingface_hub"):
        assert forbidden not in source, forbidden


def test_the_fixture_evaluation_report_is_reopenable_and_passing(tmp_path: Path) -> None:
    from slm_train.orchestration.fixture import run_cpu_fixture

    run_cpu_fixture(tmp_path)
    report = json.loads(
        (tmp_path / "evaluations" / "eval-1" / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert set(report["metric_values"]) == {"val_loss", "val_perplexity"}
    assert report["expected_evaluator_manifest_hash"]


def test_the_fixture_log_stream_verifies(tmp_path: Path) -> None:
    from slm_train.artifacts.log_index import verify_stream
    from slm_train.orchestration.fixture import run_cpu_fixture

    run_cpu_fixture(tmp_path)
    assert verify_stream(tmp_path, "metrics").ok


def test_a_second_fixture_run_refuses_rather_than_overwriting(tmp_path: Path) -> None:
    """Publication is create-if-absent, so a rerun must be a refusal."""
    from slm_train.orchestration.fixture import run_cpu_fixture

    run_cpu_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        run_cpu_fixture(tmp_path)


@pytest.mark.parametrize("relative", [
    "manifests/run_identity.json",
    "manifests/final_publication.json",
    "final/model/model.safetensors",
    "final/tokenizer/tokenizer.json",
    "evaluations/eval-1/report.json",
    "logs/log-manifest.json",
])
def test_every_expected_artifact_is_actually_published(tmp_path: Path, relative: str) -> None:
    from slm_train.orchestration.fixture import run_cpu_fixture

    run_cpu_fixture(tmp_path)
    assert (tmp_path / relative).is_file(), relative
