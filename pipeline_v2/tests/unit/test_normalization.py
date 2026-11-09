"""Tests for data normalization: the manifest certifies what was applied.

The declared data.normalization must be the APPLIED normalization (A7): a
manifest certifying unicode_nfkc over data that was only whitespace-collapsed
is the exact class of false certification these tests exist to make impossible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_train.data.dedupe import apply_normalization
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


def _reader_for(tmp_path: Path, name: str, text: str) -> LocalSourceReader:
    source = tmp_path / name
    source.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    return LocalSourceReader.from_file_set([source], **DECLARATION, role="train")


def _prepared_texts(published: Path) -> list[str]:
    payload = published / "prepared.jsonl"
    return [
        json.loads(line)["text"]
        for line in payload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest(published: Path, name: str = "manifest.json") -> dict:
    return json.loads((published / name).read_text(encoding="utf-8"))


# -- declared modes are applied ------------------------------------------------


def test_nfkc_normalization_is_applied_when_declared(tmp_path: Path) -> None:
    """A declared unicode_nfkc must actually NFKC-normalize the published text."""
    raw = "\uff26\uff35\uff2c\uff2c  \uff37\uff29\uff24\uff34\uff28 \ufb01le \u2460"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    options = dict(BASE_OPTIONS, normalization="unicode_nfkc")
    result = prepare_generation(reader, options, tmp_path / "prepared")

    # fullwidth letters, the fi ligature and the circled digit all decompose
    assert _prepared_texts(result.published_path) == ["FULL WIDTH file 1"]


def test_ws_collapse_normalization_is_applied_when_declared(tmp_path: Path) -> None:
    """A declared ws-collapse must collapse every whitespace run to one space."""
    raw = "hello \t\n  world  "
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    options = dict(BASE_OPTIONS, normalization="ws-collapse")
    result = prepare_generation(reader, options, tmp_path / "prepared")

    assert _prepared_texts(result.published_path) == ["hello world"]


def test_declared_nfkc_is_not_silently_ws_collapsed(tmp_path: Path) -> None:
    """The exact false certification: certifying NFKC over ws-collapsed data.

    The source carries no whitespace at all, so whitespace collapse alone would
    leave it byte-identical. If the pipeline degraded unicode_nfkc to
    ws-collapse, the fullwidth text would survive and the manifest would
    still certify unicode_nfkc.
    """
    raw = "\uff26\uff35\uff2c\uff2c\uff37\uff29\uff24\uff34\uff28"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    options = dict(BASE_OPTIONS, normalization="unicode_nfkc")
    result = prepare_generation(reader, options, tmp_path / "prepared")

    assert _prepared_texts(result.published_path) == ["FULLWIDTH"]
    assert _manifest(result.published_path)["normalization"] == "unicode_nfkc"


# -- the manifest certifies the applied mode -----------------------------------


def test_manifest_certifies_the_applied_normalization(tmp_path: Path) -> None:
    """Both published manifests record exactly the mode that was applied."""
    raw = "\uff26\uff35\uff2c\uff2c  \uff37\uff29\uff24\uff34\uff28"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    options = dict(BASE_OPTIONS, normalization="unicode_nfkc")
    result = prepare_generation(reader, options, tmp_path / "prepared")

    assert _manifest(result.published_path)["normalization"] == "unicode_nfkc"
    assert _manifest(result.published_path, "raw-manifest.json")["normalization"] == "unicode_nfkc"


def test_ws_collapse_manifest_certifies_ws_collapse(tmp_path: Path) -> None:
    """A ws-collapse run certifies ws-collapse, not the config's default."""
    raw = "hello   world"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    options = dict(BASE_OPTIONS, normalization="ws-collapse")
    result = prepare_generation(reader, options, tmp_path / "prepared")

    assert _manifest(result.published_path)["normalization"] == "ws-collapse"
    assert _manifest(result.published_path, "raw-manifest.json")["normalization"] == "ws-collapse"


def test_undeclared_normalization_defaults_to_ws_collapse(tmp_path: Path) -> None:
    """No declared mode: ws-collapse is applied and certified as such."""
    raw = "hello   world"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    result = prepare_generation(reader, dict(BASE_OPTIONS), tmp_path / "prepared")

    assert _prepared_texts(result.published_path) == ["hello world"]
    assert _manifest(result.published_path)["normalization"] == "ws-collapse"


def test_normalization_is_part_of_the_generation_identity(tmp_path: Path) -> None:
    """A changed normalization policy yields a new generation, not a refresh."""
    raw = "\uff26\uff35\uff2c\uff2c  \uff37\uff29\uff24\uff34\uff28"
    reader = _reader_for(tmp_path, "rows.jsonl", raw)
    nfkc = prepare_generation(
        reader, dict(BASE_OPTIONS, normalization="unicode_nfkc"), tmp_path / "a")
    ws = prepare_generation(
        reader, dict(BASE_OPTIONS, normalization="ws-collapse"), tmp_path / "b")

    assert nfkc.semantic_hash != ws.semantic_hash
    assert _prepared_texts(nfkc.published_path) == ["FULL WIDTH"]
    assert _prepared_texts(ws.published_path) == ["\uff26\uff35\uff2c\uff2c \uff37\uff29\uff24\uff34\uff28"]


def test_unsupported_normalization_mode_is_refused(tmp_path: Path) -> None:
    """An undeclared mode fails closed before any row is read."""
    reader = _reader_for(tmp_path, "rows.jsonl", "text")
    with pytest.raises(ValueError, match="unsupported data.normalization"):
        prepare_generation(
            reader, dict(BASE_OPTIONS, normalization="bogus"), tmp_path / "prepared")


# -- the normalization helper contract ------------------------------------------


def test_apply_normalization_returns_the_applied_mode() -> None:
    text, applied = apply_normalization("\uff26\uff35\uff2c\uff2c", "unicode_nfkc")
    assert (text, applied) == ("FULL", "unicode_nfkc")
    text, applied = apply_normalization("a  b", "ws-collapse")
    assert (text, applied) == ("a b", "ws-collapse")
    # a non-string value normalizes to empty text but still certifies the mode
    text, applied = apply_normalization(None, "unicode_nfkc")
    assert (text, applied) == ("", "unicode_nfkc")
    with pytest.raises(ValueError, match="unsupported"):
        apply_normalization("x", "bogus")
