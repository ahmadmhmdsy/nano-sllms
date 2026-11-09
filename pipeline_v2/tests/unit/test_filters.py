"""Tests for data filters: the manifest certifies what was applied.

The declared data.filters must be the APPLIED filters (A6): a manifest
certifying min_chars:100 over data that was never filtered is the exact
class of false certification these tests exist to make impossible. The
prepare lane applies the filters; the stage manifest must record the
applied values, not merely the declared intent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.data.prepare import prepare_generation
from slm_train.data.sources.local import LocalSourceReader

DECLARATION = {
    "declared_source_id": "declared-source",
    "expected_row_budget": 10,
    "expected_token_budget": 100,
}

BASE_OPTIONS = {
    "dataset_id": "fixture",
    "generation_id": "g001",
    "split_salt": 17,
    "validation_fraction": 0.34,
    "dedupe": False,
    "min_chars": 1,
    "regeneration_command": ("test-command",),
}

ROOT = Path(__file__).resolve().parents[3]
TINY = "pipeline_v2/tests/fixtures/tiny_config.yaml"


def _reader_for(tmp_path: Path, rows: "list[str]") -> LocalSourceReader:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "".join(json.dumps({"text": text}, sort_keys=True) + "\n" for text in rows),
        encoding="utf-8",
    )
    return LocalSourceReader.from_file_set([source], **DECLARATION, role="train")


def _prepared_texts(published: Path) -> "list[str]":
    payload = published / "prepared.jsonl"
    return [
        json.loads(line)["text"]
        for line in payload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest(published: Path, name: str = "manifest.json") -> dict:
    return json.loads((published / name).read_text(encoding="utf-8"))


# -- min_chars is applied ------------------------------------------------------


def test_min_chars_filter_rejects_short_documents(tmp_path: Path) -> None:
    """A declared min_chars must actually reject the short documents."""
    reader = _reader_for(tmp_path, ["ab", "abcdef", "abcdefghij"])
    result = prepare_generation(reader, dict(BASE_OPTIONS, min_chars=5), tmp_path / "one")

    assert result.counts["rows_seen"] == 3
    assert result.counts["filter_rejected"] == 1
    assert result.counts["rows_after_filter"] == 2
    # only the two documents that met the threshold were published
    assert _prepared_texts(result.published_path) == ["abcdef", "abcdefghij"]


def test_min_chars_filter_keeps_document_at_the_boundary(tmp_path: Path) -> None:
    """A document exactly min_chars long is kept: the filter is "<", not "<="."""
    reader = _reader_for(tmp_path, ["abcde", "abcd"])
    result = prepare_generation(reader, dict(BASE_OPTIONS, min_chars=5), tmp_path / "one")

    assert result.counts["filter_rejected"] == 1
    assert result.counts["rows_kept"] == 1
    assert _prepared_texts(result.published_path) == ["abcde"]


# -- max_chars is applied ------------------------------------------------------


def test_max_chars_filter_truncates_long_documents(tmp_path: Path) -> None:
    """A declared max_chars must actually truncate the long documents."""
    reader = _reader_for(tmp_path, ["abcdefghij", "ab"])
    result = prepare_generation(
        reader, dict(BASE_OPTIONS, min_chars=1, max_chars=4), tmp_path / "one")

    assert result.counts["truncated"] == 1
    assert _prepared_texts(result.published_path) == ["abcd", "ab"]


def test_max_chars_filter_leaves_short_documents_untouched(tmp_path: Path) -> None:
    """Documents within max_chars pass through byte-identical."""
    reader = _reader_for(tmp_path, ["ab", "abcd"])
    result = prepare_generation(
        reader, dict(BASE_OPTIONS, min_chars=1, max_chars=10), tmp_path / "one")

    assert result.counts["truncated"] == 0
    assert _prepared_texts(result.published_path) == ["ab", "abcd"]


# -- the prepare manifest records the applied values ----------------------------


def test_prepare_manifest_records_applied_filter_values(tmp_path: Path) -> None:
    """Both published manifests record exactly the filters that were applied."""
    reader = _reader_for(tmp_path, ["abcdefghij"])
    result = prepare_generation(
        reader, dict(BASE_OPTIONS, min_chars=3, max_chars=6), tmp_path / "one")

    assert _manifest(result.published_path)["filters"] == {"min_chars": 3, "max_chars": 6}
    raw = _manifest(result.published_path, "raw-manifest.json")
    assert raw["filters"]["min_chars"] == 3
    assert raw["filters"]["max_chars"] == 6


def test_prepare_manifest_records_default_filter_values(tmp_path: Path) -> None:
    """No declared filter: the applied defaults (0 / none) are certified."""
    reader = _reader_for(tmp_path, ["ab"])
    result = prepare_generation(reader, dict(BASE_OPTIONS, min_chars=0), tmp_path / "one")

    assert _manifest(result.published_path)["filters"] == {"min_chars": 0, "max_chars": None}


def test_filters_are_part_of_the_generation_identity(tmp_path: Path) -> None:
    """A changed filter policy yields a new generation, not a refresh."""
    reader = _reader_for(tmp_path, ["abcdefghij"])
    filtered = prepare_generation(
        reader, dict(BASE_OPTIONS, min_chars=1, max_chars=4), tmp_path / "a")
    unfiltered = prepare_generation(
        reader, dict(BASE_OPTIONS, min_chars=1), tmp_path / "b")

    assert filtered.semantic_hash != unfiltered.semantic_hash
    assert _prepared_texts(filtered.published_path) == ["abcd"]
    assert _prepared_texts(unfiltered.published_path) == ["abcdefghij"]


# -- the stage manifest records the applied values, not just the declared ones ---


def _stage_engine(tmp_path: Path, config):
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.handlers import HANDLERS

    # only the stages up to and including prepare are wired; the engine then
    # blocks at tokenize, which is exactly when the prepare evidence exists.
    partial = {
        name: HANDLERS[name]
        for name in ("validate", "bootstrap", "gpu_preflight", "prepare")
    }
    return Engine(
        graph=StageGraph.r1(),
        handlers=partial,
        run_root=tmp_path / "run",
        config=config,
        repo_root=ROOT,
    )


def _config_with_filters(filters: "tuple[str, ...]"):
    from slm_train.config.loader import load_config

    config = load_config(ROOT / TINY, root=ROOT)
    return config.model_copy(deep=True, update={
        "data": config.data.model_copy(update={"filters": filters}),
    })


def test_stage_prepare_manifest_records_applied_filter_values(tmp_path: Path) -> None:
    """The stage manifest certifies the applied filter values.

    Before the fix the stage manifest recorded only {"declared": [...]}: a
    record of intent, stored beside a real filter, so a config declaring
    min_chars:100 certified filtering that the manifest never proved.
    """
    from slm_train.orchestration.errors import MissingStageInput

    config = _config_with_filters(("min_chars:100", "max_chars:500"))
    engine = _stage_engine(tmp_path, config)
    with pytest.raises(MissingStageInput):
        engine.run()

    data_manifest = json.loads(
        (tmp_path / "run" / "prepare" / "manifests" / "data.json").read_text(encoding="utf-8")
    )
    assert data_manifest["filters"]["min_chars"] == 100
    assert data_manifest["filters"]["max_chars"] == 500
    # the declared strings stay recorded for traceability
    assert data_manifest["filters"]["declared"] == ["min_chars:100", "max_chars:500"]


def test_stage_prepare_manifest_records_default_filter_values(tmp_path: Path) -> None:
    """A config declaring no filters certifies the applied defaults."""
    from slm_train.orchestration.errors import MissingStageInput

    config = _config_with_filters(())
    engine = _stage_engine(tmp_path, config)
    with pytest.raises(MissingStageInput):
        engine.run()

    data_manifest = json.loads(
        (tmp_path / "run" / "prepare" / "manifests" / "data.json").read_text(encoding="utf-8")
    )
    assert data_manifest["filters"]["min_chars"] == 0
    assert data_manifest["filters"]["max_chars"] is None


def test_changed_filter_policy_yields_a_new_generation(tmp_path: Path) -> None:
    """A changed filter policy must republish, not reuse the old generation.

    The generation id is a function of the recorded preparation semantics;
    filters are part of those semantics, so a changed filter yields a NEW
    generation rather than an implicit refresh of the unfiltered one.
    """
    from slm_train.orchestration.errors import MissingStageInput

    first = _stage_engine(tmp_path / "first", _config_with_filters(()))
    with pytest.raises(MissingStageInput):
        first.run()
    first_record = json.loads(
        (tmp_path / "first" / "run" / "prepare" / "prepared_generation" / "COMPLETE.json")
        .read_text(encoding="utf-8")
    )

    second = _stage_engine(tmp_path / "second", _config_with_filters(("min_chars:100",)))
    with pytest.raises(MissingStageInput):
        second.run()
    second_record = json.loads(
        (tmp_path / "second" / "run" / "prepare" / "prepared_generation" / "COMPLETE.json")
        .read_text(encoding="utf-8")
    )

    assert first_record["published_path"] != second_record["published_path"]
