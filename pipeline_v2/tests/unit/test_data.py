"""Task 4: pinned sources, preparation, dedupe, salted split, publication."""
import json
from pathlib import Path

import pytest

from slm_train.data.prepare import prepare_generation
from slm_train.data.sources.base import OptionalSourceDecision
from slm_train.data.sources.huggingface import HFSourceReader
from slm_train.data.sources.local import LocalSourceReader


def _write_rows(path: Path, rows: "list[dict]") -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


DECLARATION = {
    "declared_source_id": "declared-source",
    "expected_row_budget": 10,
    "expected_token_budget": 100,
}


def declaration(**overrides) -> "dict[str, object]":
    return {**DECLARATION, **overrides}


SAMPLE_OPTIONS = {
    "dataset_id": "fixture",
    "generation_id": "g001",
    "split_salt": 17,
    "validation_fraction": 0.34,
    "dedupe": True,
    "min_chars": 1,
    "regeneration_command": ("test-command",),
}


# -- pinned identity ---------------------------------------------------------


def test_local_source_requires_exact_file_set(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        LocalSourceReader.from_path(source)


def test_from_file_set_requires_explicit_role(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        LocalSourceReader.from_file_set([source], **DECLARATION, role="")  # type: ignore[arg-type]


def test_from_file_set_rejects_unknown_role(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        LocalSourceReader.from_file_set([source], **DECLARATION, role="implied-because-named-training")  # type: ignore[arg-type]


def test_from_file_set_merkle_covers_only_listed_files(tmp_path: Path) -> None:
    listed = tmp_path / "listed.jsonl"
    listed.write_text('{"text":"one"}\n', encoding="utf-8")
    unlisted = tmp_path / "unlisted.jsonl"
    unlisted.write_text('{"text":"two"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([listed], **DECLARATION, role="train")
    identity = reader.identity()
    assert set(identity["file_digests"]) == {"listed.jsonl"}
    assert 'unlisted.jsonl' not in str(identity)


def test_from_path_rejects_unpinned_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalSourceReader.from_path(tmp_path)


def test_from_path_with_recorded_manifest_verifies_files(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    pinned = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    manifest = pinned.identity()
    reader = LocalSourceReader.from_path(source, manifest=manifest, **DECLARATION, role="train")
    assert reader.revalidate()["label"] == "PASS"


def test_tampered_source_file_fails_revalidation(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    source.write_text('{"text":"tampered"}\n', encoding="utf-8")
    with pytest.raises(Exception, match="changed after pinning"):
        reader.revalidate()


def test_hf_reader_requires_nonempty_revision_and_file_list() -> None:
    # argument validation only: no network, no role inference
    with pytest.raises(ValueError, match="revision"):
        HFSourceReader.from_repository(
            "org/dataset", revision="", files=["train.jsonl"], **DECLARATION, role="train")
    with pytest.raises(ValueError, match="files"):
        HFSourceReader.from_repository(
            "org/dataset", revision="abc", files=[], **DECLARATION, role="train")


def test_optional_source_requires_explicit_absent_decision(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="decision"):
        LocalSourceReader.from_file_set([source], **DECLARATION, role="train", required=False)


def test_declared_optional_source_with_recorded_decision_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set(
        [source],
        **DECLARATION,
        role="train",
        required=False,
        absent_source=OptionalSourceDecision(absent_content="expected", reason="explicit profile decision"),
    )
    assert reader.required is False


def test_absent_content_requires_needs_user_decision_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        OptionalSourceDecision(absent_content="expected", reason="")


# -- freshness ---------------------------------------------------------------


def test_expired_freshness_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set(
        [source], **DECLARATION, role="train", freshness_expires_after="2000-01-01T00:00:00Z")
    with pytest.raises(Exception, match="BLOCKED"):
        reader.revalidate()
    with pytest.raises(Exception, match="BLOCKED"):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path / "one")


def test_freshness_report_is_not_part_of_semantic_hash(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    first = reader.revalidate()
    second = reader.revalidate()
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    # verification time is mutable evidence and may differ between checks
    assert first["label"] == second["label"] == "PASS"


# -- preparation -------------------------------------------------------------


def test_preparation_is_deterministic_and_counts_rejections(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"id":"a","text":"alpha"}\n{"id":"b","text":"beta"}\n{"id":"c","text":"alpha"}\n',
        encoding="utf-8",
    )
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    options = dict(SAMPLE_OPTIONS)
    first = prepare_generation(reader, options, tmp_path / "one")
    second = prepare_generation(reader, options, tmp_path / "two")
    assert first.semantic_hash == second.semantic_hash
    assert first.counts["duplicates_dropped"] == 1
    assert first.counts["train"] + first.counts["validation"] == 2


def test_provenance_records_role_license_and_command(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train", license_note="CC0")
    result = prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path / "one")
    assert result.provenance["source_roles"] == {"train": 1}
    assert result.provenance["license_note"] == "CC0"


def test_empty_required_source_set_fails(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"x"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    source.unlink()
    with pytest.raises(Exception, match="FAIL"):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path / "one")


def test_empty_required_set_fails(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="FAIL"):
        LocalSourceReader.from_file_set([], **DECLARATION, role="train")


def test_changed_source_content_requires_new_generation(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader_a = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    root_a = tmp_path / "a"
    result_a = prepare_generation(reader_a, dict(SAMPLE_OPTIONS), root_a)
    source_b = tmp_path / "rows_b.jsonl"
    source_b.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
    reader_b = LocalSourceReader.from_file_set([source_b], **DECLARATION, role="train")
    result_b = prepare_generation(reader_b, dict(SAMPLE_OPTIONS), tmp_path / "b")
    assert result_a.semantic_hash != result_b.semantic_hash


def test_dedupe_before_split_keeps_first_occurrence(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source, [{"text": "same"} for _ in range(4)])
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    options = dict(SAMPLE_OPTIONS)
    options["validation_fraction"] = 0.0
    result = prepare_generation(reader, options, tmp_path / "one")
    assert result.counts["duplicates_dropped"] == 3
    assert result.counts["train"] == 1


def test_split_is_deterministic_and_salt_aware(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source, [{"text": f"row-{i}-{i}{i}-tail"} for i in range(60)])
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    base = prepare_generation(
        reader, dict(SAMPLE_OPTIONS, validation_fraction=0.2), tmp_path / "base")
    other_salt = prepare_generation(
        reader, dict(SAMPLE_OPTIONS, validation_fraction=0.2, split_salt=9999),
        tmp_path / "salt")
    assert base.counts["train"] + base.counts["validation"] == 60
    # two independent preparation runs with identical options are identical
    replay = prepare_generation(
        reader, dict(SAMPLE_OPTIONS, validation_fraction=0.2), tmp_path / "replay")
    assert replay.split_assignments == base.split_assignments
    assert replay.semantic_hash == base.semantic_hash
    # a different salt changes the assignment
    assert base.split_assignments != other_salt.split_assignments


def test_overlapping_split_is_fail() -> None:
    from slm_train.data.split import overlap_between, require_no_overlap
    assert overlap_between(["d1", "d2"], ["d2"]) == ["d2"]
    with pytest.raises(Exception, match="overlap"):
        require_no_overlap(["d1"], ["d1"])


# -- publication -------------------------------------------------------------


def test_publication_is_atomic_with_marker_and_readback(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"id":"a","text":"alpha"}\n{"id":"b","text":"beta"}\n{"id":"c","text":"alpha"}\n',
        encoding="utf-8",
    )
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    result = prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    from slm_train.artifacts.markers import verify_completion_marker
    published = result.published_path
    assert published.is_dir()
    assert (published / "COMPLETE.json").is_file()
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert verify_completion_marker(published, ["prepared.jsonl"]) is True


def test_republishing_the_same_generation_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    prepared = tmp_path / "data/v2/prepared/fixture/g001"
    prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    before = {p.name: p.read_bytes() for p in sorted(prepared.iterdir()) if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    after = {p.name: p.read_bytes() for p in sorted(prepared.iterdir()) if p.is_file()}
    assert before == after
    assert set(before) >= {"prepared.jsonl", "raw-manifest.json", "manifest.json", "COMPLETE.json"}


def test_republish_different_content_under_same_generation_fails_closed(tmp_path: Path) -> None:
    source_a = tmp_path / "rows_a.jsonl"
    source_a.write_text('{"text":"one"}\n', encoding="utf-8")
    reader_a = LocalSourceReader.from_file_set([source_a], **DECLARATION, role="train")
    options = dict(SAMPLE_OPTIONS)
    prepare_generation(reader_a, options, tmp_path)
    prepared = tmp_path / "data/v2/prepared/fixture/g001"
    before = {p.name: p.read_bytes() for p in sorted(prepared.iterdir()) if p.is_file()}
    source_b = tmp_path / "rows_b.jsonl"
    source_b.write_text('{"text":"entirely different"}\n', encoding="utf-8")
    reader_b = LocalSourceReader.from_file_set([source_b], **DECLARATION, role="train")
    with pytest.raises(FileExistsError):
        prepare_generation(reader_b, options, tmp_path)
    after = {p.name: p.read_bytes() for p in sorted(prepared.iterdir()) if p.is_file()}
    assert before == after
    strays = [p.name for p in (tmp_path / "data/v2/prepared/fixture").iterdir()
              if p.name.startswith(".tmp-")]
    assert strays == []


def test_no_temp_sibling_is_left_behind_after_publication(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    prepared_dir = tmp_path / "data/v2/prepared/fixture"
    strays = [p.name for p in prepared_dir.iterdir()
              if p.name.startswith(".") or p.name.endswith(".tmp")]
    assert strays == []



# -- review fix: construction contract (C1) ----------------------------------


def test_from_path_rejects_invalid_role(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    pin = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    manifest = pin.identity()
    with pytest.raises(ValueError, match="role"):
        LocalSourceReader.from_path(source, manifest=manifest, **DECLARATION, role="zzz")  # type: ignore[arg-type]


def test_from_path_rejects_missing_role(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    pin = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    with pytest.raises(ValueError, match="role"):
        LocalSourceReader.from_path(source, manifest=pin.identity(), **DECLARATION, role=None)


def test_from_path_rejects_implicit_optional(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    pin = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    with pytest.raises(ValueError, match="decision"):
        LocalSourceReader.from_path(
            source, manifest=pin.identity(), **DECLARATION, role="train", required=False)


def test_from_file_set_direct_construction_rejects_bad_role_and_optional(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        LocalSourceReader(source, (source,), "zzz", **DECLARATION)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decision"):
        LocalSourceReader(source, (source,), "train", required=False, **DECLARATION)


def test_from_repository_rejects_invalid_role_and_implicit_optional() -> None:
    with pytest.raises(ValueError, match="role"):
        HFSourceReader.from_repository(
            "org/dataset", revision="abc", files=["f.jsonl"], **DECLARATION, role="zzz")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decision"):
        HFSourceReader.from_repository(
            "org/dataset", revision="abc", files=["f.jsonl"], **DECLARATION, role="train", required=False)


def test_from_repository_direct_construction_enforces_gates() -> None:
    with pytest.raises(ValueError, match="role"):
        HFSourceReader("id", "rev", ("f.jsonl",), "zzz", **DECLARATION)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decision"):
        HFSourceReader("id", "rev", ("f.jsonl",), "train", required=False, **DECLARATION)


# -- review fix: freshness-first BLOCKED (M2) --------------------------------


def test_changed_then_expired_source_is_blocked_not_fail(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set(
        [source], **DECLARATION, role="train", freshness_expires_after="2000-01-01T00:00:00Z")
    source.write_text('{"text":"tampered"}\n', encoding="utf-8")
    from slm_train.data.sources.base import FreshnessError
    with pytest.raises(FreshnessError, match="BLOCKED"):
        reader.revalidate()
    # freshness policy wins over the content mismatch: no "changed after pinning"
    try:
        reader.revalidate()
    except FreshnessError:
        pass
    except Exception as error:  # pragma: no cover
        raise AssertionError(f"expected BLOCKED, got {error!r}") from error


# -- review fix: completion marker revalidation (M1) -------------------------


def test_revalidate_reopens_published_completion_marker(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    result = prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    report = reader.revalidate(published_generation=result.published_path)
    assert report["label"] == "PASS"
    # sabotage the published payload: revalidation must reopen and catch it
    (result.published_path / "prepared.jsonl").write_text("corrupt", encoding="utf-8")
    with pytest.raises(Exception, match="mismatch"):
        reader.revalidate(published_generation=result.published_path)


# -- review fix: optional-absent consumption path (I1) -----------------------


def test_optional_expected_absent_source_publishes_recorded_absent(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set(
        [source],
        **DECLARATION,
        role="validation",
        required=False,
        absent_source=OptionalSourceDecision(
            absent_content="expected", reason="profile declares this source may be absent"),
    )
    source.unlink()
    result = prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    report = json.loads(
        (result.published_path / "raw-manifest.json").read_text(encoding="utf-8"))
    assert report["validation_labels"][reader.source_id()] == "SKIPPED"
    assert report["optional_absent_source"]["absent_content"] == "expected"
    assert report["optional_absent_source"]["reason"] == "profile declares this source may be absent"
    assert report["counts"]["rows_seen"] == 0
    assert result.counts["rows_kept"] == 0


def test_optional_needs_user_decision_absent_is_not_automatic(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set(
        [source],
        **DECLARATION,
        role="validation",
        required=False,
        absent_source=OptionalSourceDecision(
            absent_content="needs_user_decision", reason="profile defers the decision"),
    )
    source.unlink()
    with pytest.raises(Exception, match="NEEDS_USER_DECISION"):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)


def test_required_missing_file_is_fail(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    source.unlink()
    with pytest.raises(Exception, match="FAIL"):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)


# -- review fix: counters identity (M5) --------------------------------------


def test_counter_identity_separates_rows_kept_from_dedupe(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"text":"alpha"}\n{"text":"beta"}\n{"text":"alpha"}\n',
        encoding="utf-8",
    )
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    result = prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path / "one")
    counts = result.counts
    assert counts["rows_seen"] == counts["rows_after_filter"] + counts["filter_rejected"]
    assert counts["rows_after_filter"] == counts["rows_kept"] + counts["duplicates_dropped"]
    assert counts["rows_after_filter"] == 3
    assert counts["duplicates_dropped"] == 1
    assert counts["rows_kept"] == 2
    raw = json.loads(
        (result.published_path / "raw-manifest.json").read_text(encoding="utf-8"))
    for key in ("rows_seen", "rows_after_filter", "rows_kept", "duplicates_dropped",
                "filter_rejected", "truncated", "overlap_count"):
        assert raw["counts"][key] == counts[key]


# -- review fix: HF revalidation fail-closed (I2) ----------------------------


def _pinned_hf_reader(tmp_path: Path, with_evidence: bool, evidence_hashes=None):
    from slm_train.identity.source import hf_source_identity
    manifest = hf_source_identity("org/dataset", "pinned-rev-0000", {"f.jsonl": "0" * 64})
    kwargs: dict = {}
    if with_evidence:
        kwargs["verified_revision"] = "pinned-rev-0000"
        kwargs["verified_file_hashes"] = (
            evidence_hashes if evidence_hashes is not None else {"f.jsonl": "0" * 64})
    return HFSourceReader.from_repository(
        "org/dataset",
        revision="pinned-rev-0000",
        files=["f.jsonl"],
        manifest=manifest,
        role="train",
        **DECLARATION,
        **kwargs,
    )


def test_hf_unpinned_revalidate_raises() -> None:
    reader = HFSourceReader.from_repository(
        "org/dataset", revision="abc", files=["f.jsonl"], **DECLARATION, role="train")
    with pytest.raises(Exception, match="unpinned"):
        reader.revalidate()


def test_hf_revalidate_never_reports_pass_without_evidence() -> None:
    reader = _pinned_hf_reader(tmp_path=None, with_evidence=False)
    report = reader.revalidate()
    assert report["label"] in ("SKIPPED",)
    assert report["label"] != "PASS"
    assert report["freshness"] == "SKIPPED"


def test_hf_revalidate_pass_only_with_matching_evidence(tmp_path=None) -> None:
    reader = _pinned_hf_reader(None, with_evidence=True)
    report = reader.revalidate()
    assert report["label"] == "PASS"
    assert report["freshness"] == "PASS"
    assert report["verified_revision"] == "pinned-rev-0000"


def test_hf_revalidate_rejects_mismatched_recorded_hash() -> None:
    from slm_train.data.sources.base import SourceValidationError
    reader = _pinned_hf_reader(None, with_evidence=True, evidence_hashes={"f.jsonl": "f" * 64})
    with pytest.raises((SourceValidationError, ValueError), match="disagrees"):
        reader.revalidate()


# -- review fix: crash-before-publish cleanup (T2) ---------------------------


def test_crash_before_publish_leaves_no_partial_generation(tmp_path: Path, monkeypatch) -> None:
    import slm_train.data.prepare as prepare_module
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    from slm_train.artifacts.manifest import write_manifest as real_write_manifest

    def crashing_write(root, relative_path, value):
        if relative_path == "raw-manifest.json":
            raise OSError("simulated crash between manifest writes")
        return real_write_manifest(root, relative_path, value)

    monkeypatch.setattr(prepare_module, "write_manifest", crashing_write)
    with pytest.raises(OSError, match="simulated crash"):
        prepare_generation(reader, dict(SAMPLE_OPTIONS), tmp_path)
    final = tmp_path / "data/v2/prepared/fixture/g001"
    assert not final.exists()
    strays = [p for p in (tmp_path / "data/v2/prepared/fixture").iterdir()
              if p.name.startswith(".tmp-")]
    assert strays == []


# -- review fix: computed overlap (M4) ---------------------------------------


def test_overlap_is_computed_and_republished(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source, [{"text": f"row-{i}-{i}{i}-tail"} for i in range(20)])
    reader = LocalSourceReader.from_file_set([source], **DECLARATION, role="train")
    result = prepare_generation(
        reader, dict(SAMPLE_OPTIONS, validation_fraction=0.0), tmp_path / "one")
    assert result.counts["overlap_count"] == 0
    manifest = json.loads(
        (result.published_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["overlap_report"] == {"overlap_count": 0, "overlapping_ids": []}



# -- review fix: offline collect_file_hashes contracts (I3) ------------------


def test_collect_file_hashes_requires_explicit_pinned_revision() -> None:
    # No I/O, no network import: the unpinned revision fails before the hub
    # module is even imported.
    import sys
    reader = HFSourceReader.from_repository(
        "org/dataset", revision="abc", files=["f.jsonl"], **DECLARATION, role="train")
    blocked = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = None  # any import attempt halts
    try:
        with pytest.raises(ValueError, match="revision"):
            HFSourceReader.collect_file_hashes("", "org/dataset", ["f.jsonl"])
        with pytest.raises(ValueError, match="revision"):
            HFSourceReader.collect_file_hashes(None, "org/dataset", ["f.jsonl"])
        with pytest.raises(ValueError, match="repository"):
            HFSourceReader.collect_file_hashes("abc", "", ["f.jsonl"])
        with pytest.raises(ValueError, match="file list"):
            HFSourceReader.collect_file_hashes("abc", "org/dataset", [])
    finally:
        if blocked is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = blocked


def test_collect_file_hashes_refuses_to_invent_hashes(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    captured = {}
    saved = sys.modules.pop("huggingface_hub", None)
    calls = {"download": []}

    def fake_list_repo_files(repository, revision=None, token=None):
        captured["list"] = (repository, revision, token)
        return ["declared.jsonl", "unlisted.jsonl"]

    def fake_hf_hub_download(repo_id, filename, revision=None, token=None):
        calls["download"].append((repo_id, filename, revision, token))
        local = tmp_path / filename
        local.write_bytes(filename.encode("utf-8"))
        return str(local)

    stub = types.ModuleType("huggingface_hub")
    stub.list_repo_files = fake_list_repo_files
    stub.hf_hub_download = fake_hf_hub_download
    sys.modules["huggingface_hub"] = stub
    try:
        from slm_train.contracts.hash import sha256_bytes
        hashes = HFSourceReader.collect_file_hashes(
            "pinned-rev", "org/dataset", ["declared.jsonl"])
        # offline behavior: only hashes actually read from the pinned files are
        # recorded, nothing is invented for an undeclared or absent file
        assert set(hashes) == {"declared.jsonl"}
        assert hashes["declared.jsonl"] == sha256_bytes(b"declared.jsonl")
        assert captured["list"][2] is False and captured["list"][1] == "pinned-rev"
        assert calls["download"][0][3] is False
        # a declared file absent from the pinned revision never returns a
        # fabricated hash and never a partial digest map
        with pytest.raises(ValueError, match="declared file is absent"):
            HFSourceReader.collect_file_hashes(
                "pinned-rev", "org/dataset", ["declared.jsonl", "missing.jsonl"])
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


# -- follow-up fix: HF evidence bound to the DECLARED file set (N1) -----------


def _declared_hf_manifest(digests):
    from slm_train.identity.source import hf_source_identity
    return hf_source_identity("org/dataset", "pinned-rev-0000", digests)


def test_hf_direct_construction_rejects_manifest_disjoint_from_declared_files():
    manifest = _declared_hf_manifest({"g.jsonl": "0" * 64})
    with pytest.raises(ValueError, match="declared"):
        HFSourceReader(
            "org/dataset", "pinned-rev-0000", ("f.jsonl",), "train",
            manifest=manifest, **DECLARATION)


def test_hf_direct_construction_rejects_empty_manifest_digest_map():
    manifest = _declared_hf_manifest({})
    with pytest.raises(ValueError, match="declared"):
        HFSourceReader(
            "org/dataset", "pinned-rev-0000", ("f.jsonl",), "train",
            manifest=manifest, **DECLARATION)


def test_hf_direct_evidence_disjoint_from_declared_set_never_passes():
    manifest = _declared_hf_manifest({"f.jsonl": "0" * 64})
    reader = HFSourceReader(
        "org/dataset", "pinned-rev-0000", ("f.jsonl",), "train",
        manifest=manifest, **DECLARATION,
        verified_revision="pinned-rev-0000",
        verified_file_hashes={"g.jsonl": "0" * 64},
    )
    from slm_train.data.sources.base import SourceValidationError
    with pytest.raises(SourceValidationError, match="declared"):
        reader.revalidate()


def test_hf_direct_empty_evidence_never_passes():
    manifest = _declared_hf_manifest({"f.jsonl": "0" * 64})
    reader = HFSourceReader(
        "org/dataset", "pinned-rev-0000", ("f.jsonl",), "train",
        manifest=manifest, **DECLARATION,
        verified_revision="pinned-rev-0000",
        verified_file_hashes={},
    )
    from slm_train.data.sources.base import SourceValidationError
    with pytest.raises(SourceValidationError, match="declared"):
        reader.revalidate()


def test_hf_direct_partial_evidence_never_passes():
    manifest = _declared_hf_manifest({"f.jsonl": "0" * 64, "h.jsonl": "1" * 64})
    reader = HFSourceReader(
        "org/dataset", "pinned-rev-0000", ("f.jsonl", "h.jsonl"), "train",
        manifest=manifest, **DECLARATION,
        verified_revision="pinned-rev-0000",
        verified_file_hashes={"f.jsonl": "0" * 64},
    )
    from slm_train.data.sources.base import SourceValidationError
    with pytest.raises(SourceValidationError, match="declared"):
        reader.revalidate()


# -- follow-up fix: declared source identity fields (N2) ----------------------


def test_declared_source_id_is_stable_across_local_repin(tmp_path: Path) -> None:
    directory_a = tmp_path / "a"
    directory_b = tmp_path / "b"
    directory_a.mkdir()
    directory_b.mkdir()
    for directory in (directory_a, directory_b):
        (directory / "rows.jsonl").write_text('{"text":"one"}\n', encoding="utf-8")
    first = LocalSourceReader.from_file_set(
        [directory_a / "rows.jsonl"],
        role="train", declared_source_id="dataset-core",
        expected_row_budget=8, expected_token_budget=80)
    second = LocalSourceReader.from_file_set(
        [directory_b / "rows.jsonl"],
        role="train", declared_source_id="dataset-core",
        expected_row_budget=8, expected_token_budget=80)
    # re-pinning identical content under a fresh manifest keeps the declared id
    assert first.source_id() == second.source_id() == "dataset-core"
    repl = LocalSourceReader.from_path(
        directory_b / "rows.jsonl", manifest=second.identity(),
        role="train", declared_source_id="dataset-core",
        expected_row_budget=8, expected_token_budget=80)
    assert repl.source_id() == "dataset-core"
    # the id is declared, not derived from the current file content
    assert repl.source_id() != f"local:{repl.identity()['root_hash'][:16]}"


def test_declared_source_id_is_stable_across_hf_repin() -> None:
    manifest = _declared_hf_manifest({"f.jsonl": "0" * 64})
    repin = _declared_hf_manifest({"f.jsonl": "0" * 64})
    first = HFSourceReader.from_repository(
        "org/dataset", revision="pinned-rev-0000", files=["f.jsonl"],
        manifest=manifest, role="train",
        declared_source_id="hf-declared", expected_row_budget=4,
        expected_token_budget=40)
    second = HFSourceReader.from_repository(
        "org/dataset", revision="pinned-rev-0000", files=["f.jsonl"],
        manifest=repin, role="train",
        declared_source_id="hf-declared", expected_row_budget=4,
        expected_token_budget=40)
    assert first.source_id() == second.source_id() == "hf-declared"


def test_changing_the_declared_budget_is_visible_in_the_declaration(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    first = LocalSourceReader.from_file_set(
        [source], role="train", declared_source_id="dataset-core",
        expected_row_budget=8, expected_token_budget=80)
    second = LocalSourceReader.from_file_set(
        [source], role="train", declared_source_id="dataset-core",
        expected_row_budget=9, expected_token_budget=90)
    provenance_first = first.provenance()
    provenance_second = second.provenance()
    assert provenance_first["expected_row_budget"] == 8
    assert provenance_first["expected_token_budget"] == 80
    assert provenance_second["expected_row_budget"] == 9
    assert provenance_second["expected_token_budget"] == 90
    assert provenance_second["declared_source_id"] == "dataset-core"
    assert provenance_first["source_id"] == "dataset-core"


def test_absent_budget_is_rejected_not_defaulted(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(TypeError):
        LocalSourceReader.from_file_set([source], role="train", declared_source_id="x")
    with pytest.raises(ValueError, match="declared"):
        LocalSourceReader(
            tmp_path, (source,), "train", declared_source_id="x",
            expected_row_budget=None, expected_token_budget=10)
    with pytest.raises(ValueError, match="declared"):
        HFSourceReader(
            "org/dataset", "abc", ("f.jsonl",), "train", declared_source_id="x",
            expected_row_budget=1, expected_token_budget=None)
    with pytest.raises(ValueError, match="declared"):
        HFSourceReader.from_repository(
            "org/dataset", revision="abc", files=["f.jsonl"], role="train",
            declared_source_id="", expected_row_budget=1, expected_token_budget=1)


# -- Task 5: tokenizer identity, packing, shards, dataset validation ---------


def _pin_local(tmp_path: Path):
    """A real pinned tokenizer identity: exact local file-set identity."""
    from slm_train.data.tokenizer import pin_local_tokenizer
    vocab = tmp_path / "tokenizer.json"
    vocab.write_text('{"vocab": true}', encoding="utf-8")
    return pin_local_tokenizer(
        [vocab], vocabulary_size=64, max_token_id=10, package_version="test-1",
        bos=0, eos=1, pad=2, special_tokens=("cls",), context_length=32)


def _pack(tokens, sequence_length, mode, tmp_path, **kwargs):
    """pack_tokens with the mandatory pinned tokenizer fixture."""
    from slm_train.data.packing import pack_tokens
    return pack_tokens(
        tokens, sequence_length, mode, tmp_path,
        tokenizer=_pin_local(tmp_path), **kwargs)


def test_packing_records_tail_loss(tmp_path: Path) -> None:
    result = _pack([1, 2, 3, 4, 5], 2, "concat", tmp_path)
    assert result.counts["usable_blocks"] == 2
    assert result.counts["dropped_tail"] == 1
    assert result.shards


def test_dataset_rejects_a_wrong_generation_manifest(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    with pytest.raises(ValueError, match="completion"):
        PackedDataset.from_manifest(tmp_path / "missing.json")


# -- review fix: mandatory pinned tokenizer (C2) -----------------------------


def test_pack_tokens_without_a_pinned_tokenizer_is_rejected(tmp_path: Path) -> None:
    from slm_train.data.packing import pack_tokens
    from slm_train.data.tokenizer import TokenizerError
    with pytest.raises(TokenizerError, match="pinned"):
        pack_tokens([1, 2], 2, "concat", tmp_path)


def test_unpinned_tokenizer_identity_rejected(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack([1, 2, 3, 4, 5], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"] = {**manifest["tokenizer"], "repository_identity": ""}
    result.manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(ValueError, match="unpinned"):
        PackedDataset.from_manifest(result.manifest_path, 2)


def test_token_above_declared_max_is_rejected(tmp_path: Path) -> None:
    from slm_train.data.packing import pack_tokens
    from slm_train.data.tokenizer import TokenizerError
    identity = _pin_local(tmp_path)   # max_token_id=10, vocabulary_size=64
    with pytest.raises(TokenizerError, match="exceeds the recorded maximum token id"):
        pack_tokens([1, 11], 2, "concat", tmp_path, tokenizer=identity)


def test_negative_token_id_is_labeled_fail_not_overflow(tmp_path: Path) -> None:
    from slm_train.data.packing import pack_tokens
    from slm_train.data.tokenizer import TokenizerError
    identity = _pin_local(tmp_path)
    with pytest.raises(TokenizerError, match="out of range"):
        pack_tokens([1, -1], 2, "concat", tmp_path, tokenizer=identity)


def test_tokenized_manifest_roundtrip(tmp_path: Path) -> None:
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    tokenized = result.tokenized_manifest()
    assert tokenized.sequence_length == 2
    assert tokenized.usable_blocks == 2
    assert tokenized.packing_mode == "concat"
    assert tokenized.shards[0].path == result.shards[0].path
    assert tokenized.shards[0].sha256 == result.shards[0].sha256


# -- review fix: primary manifest tamper is detected (I2) --------------------


def test_manifest_generation_tamper_is_detected(tmp_path: Path) -> None:
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["generation"] = "tampered-generation"
    result.manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    from slm_train.data.dataset import PackedDataset
    with pytest.raises(Exception, match="manifest.json"):
        PackedDataset.from_manifest(result.manifest_path, 2)


# -- review fix: tampered mask file is detected (I3) --------------------------


def test_tampered_mask_file_is_rejected(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack(
        [1, 2, 3, 4], 2, "document_masked", tmp_path,
        documents=[{"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]}])
    mask_path = result.published_path / result.shards[0].mask_path
    mask_path.write_bytes(mask_path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(Exception, match="mask"):
        PackedDataset.from_manifest(result.manifest_path, 2)


# -- review fix: mask required before memmaps (M5) ----------------------------


def test_mask_required_before_memmaps(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    from slm_train.data.dataset import PackedDataset
    from slm_train.contracts.tokenized import TokenizedDataManifest

    def _boom(*args, **kwargs):
        raise AssertionError("memmap constructed before mask prerequisites passed")

    monkeypatch.setattr(np, "memmap", _boom)
    shard = tmp_path / "shard.bin"
    shard.write_bytes(b"")
    tm = TokenizedDataManifest(
        schema_version=1,
        shards=({"schema_version": 1, "path": "shard.bin",
                 "size_bytes": 0, "sha256": "0" * 64},),
        mask_path="shard.mask.bin",
        sequence_length=2, packing_mode="document_masked",
        usable_blocks=0, dropped_tail_tokens=0, overflow_blocks=0,
        source_data_hash="hash",
    )
    with pytest.raises(ValueError, match="loss-mask file per shard"):
        PackedDataset(tm, 2, shard_paths=[shard], mask_paths=[None])


# -- Task 5: token validation and packing modes ------------------------------


def test_out_of_range_token_is_fail_not_truncated(tmp_path: Path) -> None:
    from slm_train.data.tokenizer import validate_token_ids, TokenizerError
    identity = _pin_local(tmp_path)
    with pytest.raises(TokenizerError, match="out of range"):
        validate_token_ids([1, 99, 2], identity)


def test_concat_preserves_source_order(tmp_path: Path) -> None:
    import numpy as np
    tokens = [7, 3, 9, 1, 8, 2, 5]
    result = _pack(tokens, 3, "concat", tmp_path)
    shard = np.memmap(result.published_path / result.shards[0].path, dtype="<u4", mode="r")
    assert list(shard) == [7, 3, 9, 1, 8, 2]  # order preserved, tail dropped
    assert result.counts["dropped_tail"] == 1
    assert result.counts["dropped_tail_tokens"] == 1  # one recorded token lost


def test_document_masked_retains_boundaries_and_masks(tmp_path: Path) -> None:
    import numpy as np
    result = _pack(
        [1, 2, 3, 4, 5, 6], 2, "document_masked", tmp_path,
        documents=[
            {"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]},
            {"tokens": [5, 6], "mask": [0, 1]},
        ])
    assert result.counts["total_tokens"] == 6
    assert result.counts["usable_blocks"] == 3
    tokens = np.memmap(result.published_path / result.shards[0].path, dtype="<u4", mode="r")
    masks = np.memmap(result.published_path / result.shards[0].mask_path, dtype="<u1", mode="r")
    assert list(masks) == [1, 1, 0, 1]  # one bit per token, boundary kept
    assert list(tokens) == [1, 2, 3, 4]  # first harness document order is preserved


def test_document_masked_contract_rejects_missing_masks(tmp_path: Path) -> None:
    from slm_train.contracts.tokenized import TokenizedDataManifest
    with pytest.raises(ValueError, match="mask_path is required"):
        TokenizedDataManifest(
            schema_version=1,
            shards=({"schema_version": 1, "path": "shard.bin",
                     "size_bytes": 0, "sha256": "0" * 64},),
            mask_path=None,
            sequence_length=2, packing_mode="document_masked",
            usable_blocks=0, dropped_tail_tokens=0, overflow_blocks=0,
            source_data_hash="hash",
        )


def test_mask_file_missing_is_an_error_not_silent_ones(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack(
        [1, 2, 3, 4], 2, "document_masked", tmp_path,
        documents=[{"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]}])
    (result.published_path / result.shards[0].mask_path).unlink()
    with pytest.raises(Exception, match="mask is missing"):
        PackedDataset.from_manifest(result.manifest_path, 2)


def test_overflow_is_recorded_not_dropped(tmp_path: Path) -> None:
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path, overflow_blocks=3)
    # usable blocks are computed from the actual token stream, not echoed back
    assert result.counts["usable_blocks"] == 2
    assert result.counts["total_tokens"] == 4
    assert result.counts["dropped_tail_tokens"] == 0
    # the caller-recorded overflow is a separate counter, never a block source
    assert 0 < result.counts["overflow_blocks"] < result.counts["total_tokens"]


def test_tail_can_be_explicitly_retained_with_pad(tmp_path: Path) -> None:
    import numpy as np
    result = _pack([1, 2, 3, 4, 5], 2, "concat", tmp_path,
                   retain_tail=True, pad_token_id=9)
    # 5 real tokens == 2 full blocks + 1 retained partial padded with 1 pad token
    assert result.counts["dropped_tail_tokens"] == 0
    assert result.counts["retained_partial_blocks"] == 1
    assert result.counts["retained_partial_padded_tokens"] == 1
    shard = np.memmap(result.published_path / result.shards[0].path, dtype="<u4", mode="r")
    assert len(shard) == 6  # 2 full blocks + one padded partial block
    assert list(shard) == [1, 2, 3, 4, 5, 9]  # the pad token is the recorded pad id


def test_republishing_different_content_fails_closed(tmp_path: Path) -> None:
    first = _pack([1, 2, 3], 2, "concat", tmp_path)
    before = {p.name: p.read_bytes() for p in sorted(first.published_path.iterdir())}
    # identical packed content hashes to the same generation: fail closed, no overwrite
    with pytest.raises(FileExistsError):
        _pack([1, 2, 3], 2, "concat", tmp_path)
    after = {p.name: p.read_bytes() for p in sorted(first.published_path.iterdir())}
    assert before == after


# -- Task 5: shard verification and dataset construction --------------------


def test_shard_sizes_and_hashes_are_read_back(tmp_path: Path) -> None:
    from slm_train.contracts.hash import sha256_file
    result = _pack(list(range(8)), 4, "concat", tmp_path)
    shard_path = result.published_path / result.shards[0].path
    assert shard_path.stat().st_size == result.shards[0].size_bytes
    assert sha256_file(shard_path) == result.shards[0].sha256


def test_tampered_shard_bytes_are_rejected(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    shard_path = result.published_path / result.shards[0].path
    shard_path.write_bytes(shard_path.read_bytes()[:-1] + b"\xff")
    with pytest.raises(Exception, match="read-back failed|harness marker mismatch"):
        PackedDataset.from_manifest(result.manifest_path, 2)


def test_wrong_tokenizer_identity_is_rejected(tmp_path: Path) -> None:
    from slm_train.data.dataset import DatasetError, PackedDataset
    from slm_train.data.packing import pack_tokens
    from slm_train.data.tokenizer import pin_local_tokenizer, pin_hf_tokenizer
    from slm_train.contracts.hash import canonical_sha256
    from slm_train.data.tokenizer import manifest_json
    hf = pin_hf_tokenizer("org/tokenizer", "rev0", {"vocab.json": "0" * 64},
                          vocabulary_size=32, max_token_id=4, package_version="v")
    assert hf.manifest is not None
    wrong_hash = canonical_sha256(
        {k: v for k, v in manifest_json(hf).items() if k != "extra_record"})
    vocab_file = tmp_path / "tok.json"
    vocab_file.write_text("{}", encoding="utf-8")
    identity = pin_local_tokenizer(
        [vocab_file], vocabulary_size=32, max_token_id=4, package_version="v")
    assert identity.manifest != hf.manifest
    published = pack_tokens([1, 2, 3, 4], 2, "concat", tmp_path, tokenizer=identity)
    with pytest.raises((ValueError, DatasetError), match="tokenizer identity"):
        PackedDataset.from_manifest(
            published.manifest_path, 2, expected_tokenizer_hash=wrong_hash)


def test_wrong_source_generation_is_rejected(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    with pytest.raises(ValueError, match="completion"):
        PackedDataset.from_manifest(
            result.manifest_path, 2, expected_generation="g-does-not-exist")


def test_negative_and_out_of_range_indices(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    dataset = PackedDataset.from_manifest(result.manifest_path, 2)
    tail = dataset[-1]
    assert list(tail["input_ids"]) == [3, 4]
    with pytest.raises(IndexError):
        dataset[dataset.blocks]
    with pytest.raises(IndexError):
        dataset[-(dataset.blocks + 1)]


def test_no_broad_glob_is_used(tmp_path: Path, monkeypatch) -> None:
    import glob as glob_module
    import os
    from slm_train.data.dataset import PackedDataset
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    def _forbidden(*args, **kwargs):
        raise AssertionError("broad directory glob must not be used")
    monkeypatch.setattr("pathlib.Path.glob", _forbidden)
    monkeypatch.setattr("pathlib.Path.iterdir", _forbidden)
    monkeypatch.setattr("pathlib.Path.rglob", _forbidden)
    monkeypatch.setattr(glob_module, "glob", _forbidden)
    monkeypatch.setattr(glob_module, "iglob", _forbidden)
    monkeypatch.setattr(os, "scandir", _forbidden)
    dataset = PackedDataset.from_manifest(result.manifest_path, 2)
    assert dataset.blocks == len(dataset) == 2


def test_document_masked_indexing_semantics(tmp_path: Path) -> None:
    from slm_train.data.dataset import PackedDataset
    result = _pack(
        [1, 2, 3, 4], 2, "document_masked", tmp_path,
        documents=[{"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]}])
    dataset = PackedDataset.from_manifest(result.manifest_path, 2)
    assert dataset.blocks == 2
    block = dataset[1]
    assert list(block["loss_mask"]) == [0, 1]


# -- Task 5 review fix: validate_packed_tokens labeled checks (C1) -----------


def _reforge(published, shard_edits=None) -> None:
    """Rewrite the marker chain after probing an edit of published bytes.

    A test helper for tampered publications: it re-records the shard hashes,
    tokenized manifest, prepared manifest hash map, and completion marker so
    a single targeted defect is the surviving labeled failure."""
    import numpy as np

    from slm_train.artifacts.markers import write_completion_marker
    from slm_train.contracts.hash import canonical_json_bytes, sha256_file
    from slm_train.data.packing import TOKENIZED_MANIFEST_NAME
    shard_edits = shard_edits or {}
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["shards"]:
        shard_path = published / entry["path"]
        edit = shard_edits.get(entry["path"])
        if edit is not None:
            np.asarray(edit, dtype="<u4").tofile(shard_path)
        entry["size_bytes"] = shard_path.stat().st_size
        entry["sha256"] = sha256_file(shard_path)
        if entry.get("mask_path"):
            mask_path = published / entry["mask_path"]
            entry["mask_size_bytes"] = mask_path.stat().st_size
            entry["mask_sha256"] = sha256_file(mask_path)
    (published / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    tokenized = json.loads((published / "tokenized-manifest.json").read_text(encoding="utf-8"))
    tokenized["shards"] = manifest["shards"]
    tokenized["usable_blocks"] = int(manifest["counts"]["usable_blocks"])
    tokenized["dropped_tail_tokens"] = int(manifest["counts"]["dropped_tail_tokens"])
    (published / "tokenized-manifest.json").write_bytes(canonical_json_bytes(tokenized))
    prepared = json.loads((published / "tokens-prepared-manifest.json").read_text(encoding="utf-8"))
    names = ["manifest.json", TOKENIZED_MANIFEST_NAME]
    for entry in manifest["shards"]:
        names.append(str(entry["path"]))
        if entry.get("mask_path") is not None:
            names.append(str(entry["mask_path"]))
    prepared["manifest"] = {name: sha256_file(published / name) for name in names}
    (published / "tokens-prepared-manifest.json").write_bytes(canonical_json_bytes(prepared))
    write_completion_marker(
        published,
        {name: (published / name).read_bytes() for name in names},
        "tokens-prepared-manifest.json",
    )


def test_validate_passes_a_valid_concat_manifest(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "PASS"
    assert report["checks"] and all(v == "PASS" for v in report["checks"].values())


def test_validate_passes_a_valid_document_masked_manifest(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack(
        [1, 2, 3, 4], 2, "document_masked", tmp_path,
        documents=[{"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]}])
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "PASS"
    assert all(v == "PASS" for v in report["checks"].values())


def test_validate_retain_tail_counts_identity_passes(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2, 3, 4, 5], 2, "concat", tmp_path,
                   retain_tail=True, pad_token_id=9)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "PASS"  # a valid retain-tail pack is never FAIL
    assert report["checks"]["counts_identity"] == "PASS"


def test_validate_dropped_tail_counts_identity_passes(tmp_path: Path) -> None:
    """A DROPPED tail is the case the fixtures never exercised.

    With count_and_drop the shards carry usable*seq tokens while total_tokens
    counts the real stream, so the dropped tail must be SUBTRACTED. A plus here
    only passes when the tail is zero, which is why every fixture passed while
    the first real corpus failed.
    """
    from slm_train.data.validate import validate_packed_tokens

    # 7 tokens at sequence length 4: two whole blocks and a 3-token tail that
    # count_and_drop discards, so the shards hold 8 and the stream held 7+3.
    result = _pack([1, 2, 3, 4, 5, 6, 7], 4, "concat", tmp_path,
                   retain_tail=False)
    counts = result.counts
    assert counts["dropped_tail_tokens"] > 0, "this test needs a nonzero dropped tail"
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["checks"]["counts_identity"] == "PASS", report["reason"]
    assert report["label"] == "PASS"


def test_validate_missing_mask_is_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack(
        [1, 2, 3, 4], 2, "document_masked", tmp_path,
        documents=[{"tokens": [1, 2, 3, 4], "mask": [1, 1, 0, 1]}])
    (result.published_path / result.shards[0].mask_path).unlink()
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["shard_readback"] == "FAIL"
    assert "mask" in report["reason"]


def test_validate_shard_size_and_hash_mismatch_is_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    shard_path = result.published_path / result.shards[0].path
    shard_path.write_bytes(shard_path.read_bytes()[:-1] + b"\xff")
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["shard_readback"] == "FAIL"
    assert "read-back failed" in report["reason"]


def test_validate_counts_identity_violation_is_fail(tmp_path: Path) -> None:
    from slm_train.contracts.hash import canonical_json_bytes
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2, 3, 4], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["usable_blocks"] = 3
    (result.published_path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    _reforge(result.published_path)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["counts_identity"] == "FAIL"
    assert report["checks"]["completion_marker"] == "PASS"


def test_validate_tokenizer_extra_field_is_labeled_fail(tmp_path: Path) -> None:

    from slm_train.data.validate import validate_packed_tokens
    from slm_train.contracts.hash import canonical_json_bytes
    result = _pack([1, 2], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["extra_key"] = "forbidden extra fields are always FAIL"
    (result.published_path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    _reforge(result.published_path)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["tokenizer_manifest"] == "FAIL"
    assert "FAIL" in report["reason"]


def test_validate_malformed_special_token_list_is_labeled_fail(tmp_path: Path) -> None:

    from slm_train.data.validate import validate_packed_tokens
    from slm_train.contracts.hash import canonical_json_bytes
    result = _pack([1, 2], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["special_tokens"] = [123]
    (result.published_path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    _reforge(result.published_path)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["tokenizer_manifest"] == "FAIL"
    assert "FAIL" in report["reason"]


def test_validate_out_of_vocabulary_token_is_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    from slm_train.contracts.hash import canonical_json_bytes
    result = _pack([1, 2], 2, "concat", tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["dropped_tail_tokens"] = 0
    manifest["counts"]["dropped_tail"] = 0
    (result.published_path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    _reforge(result.published_path, {"tokens-00000.bin": [42, 42]})
    report = validate_packed_tokens(result.published_path / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["token_vocabulary"] == "FAIL"
    assert "42" in report["reason"]


def test_validate_missing_generation_is_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    report = validate_packed_tokens(tmp_path / "absent" / "manifest.json")
    assert report["label"] == "FAIL"
    assert report["checks"]["manifest_load"] == "FAIL"


# -- Task 5 follow-up: structural corruption fails closed into a label (N1) ---


def _reforge_manifest(published, corrupt) -> None:
    """Reforge the marker chain after a *structural* manifest edit.

    Unlike _reforge this tolerates the very corruption it publishes: a shard
    entry that is not a JSON object, or one that lost a required key, is kept
    exactly as the test wrote it and is simply not re-hashed, so the only
    surviving defect is the one under test.
    """
    from slm_train.artifacts.markers import write_completion_marker
    from slm_train.contracts.hash import canonical_json_bytes, sha256_file
    from slm_train.data.packing import TOKENIZED_MANIFEST_NAME

    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    corrupt(manifest)
    (published / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    names = ["manifest.json", TOKENIZED_MANIFEST_NAME]
    for entry in manifest.get("shards", []):
        if not isinstance(entry, dict):
            continue
        for key in ("path", "mask_path"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                names.append(value)
    prepared = json.loads(
        (published / "tokens-prepared-manifest.json").read_text(encoding="utf-8"))
    prepared["manifest"] = {name: sha256_file(published / name) for name in names}
    (published / "tokens-prepared-manifest.json").write_bytes(canonical_json_bytes(prepared))
    write_completion_marker(
        published,
        {name: (published / name).read_bytes() for name in names},
        "tokens-prepared-manifest.json",
    )


def _counts_not_a_dict(manifest) -> None:
    manifest["counts"] = ["usable_blocks", "is", "not", "an", "object"]


def _count_not_an_int(manifest) -> None:
    manifest["counts"]["usable_blocks"] = "1"


def _shard_entry_not_a_dict(manifest) -> None:
    manifest["shards"] = ["tokens-00000.bin"]


def _shard_missing_required_key(manifest) -> None:
    del manifest["shards"][0]["sha256"]


_STRUCTURAL_CORRUPTIONS = (
    _counts_not_a_dict,
    _count_not_an_int,
    _shard_entry_not_a_dict,
    _shard_missing_required_key,
)


def _assert_labeled_fail(report, check: str) -> None:
    """The report is a labeled result, and only labeled checks inside it."""
    from slm_train.data.validate import LABELS
    assert report["label"] == "FAIL"
    assert report["label"] in LABELS
    assert report["checks"][check] == "FAIL"
    assert all(value in LABELS for value in report["checks"].values())


def test_validate_non_dict_counts_section_is_a_labeled_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2], 2, "concat", tmp_path)
    _reforge_manifest(result.published_path, _counts_not_a_dict)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    _assert_labeled_fail(report, "manifest_structure")
    assert "counts" in report["reason"]


def test_validate_non_int_count_value_is_a_labeled_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2], 2, "concat", tmp_path)
    _reforge_manifest(result.published_path, _count_not_an_int)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    _assert_labeled_fail(report, "manifest_structure")
    assert "not an integer" in report["reason"]


def test_validate_non_dict_shard_entry_is_a_labeled_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2], 2, "concat", tmp_path)
    _reforge_manifest(result.published_path, _shard_entry_not_a_dict)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    _assert_labeled_fail(report, "manifest_structure")
    assert "shard entry 0" in report["reason"]


def test_validate_missing_required_shard_key_is_a_labeled_fail(tmp_path: Path) -> None:
    from slm_train.data.validate import validate_packed_tokens
    result = _pack([1, 2], 2, "concat", tmp_path)
    _reforge_manifest(result.published_path, _shard_missing_required_key)
    report = validate_packed_tokens(result.published_path / "manifest.json")
    _assert_labeled_fail(report, "manifest_structure")
    assert "sha256" in report["reason"]


@pytest.mark.parametrize(
    "corrupt", _STRUCTURAL_CORRUPTIONS,
    ids=("counts_not_a_dict", "count_not_an_int", "shard_not_a_dict", "shard_missing_key"),
)
def test_malformed_manifest_never_raises_a_library_exception(corrupt, tmp_path: Path) -> None:
    """Even in strict mode a corrupt record is a labeled failure, not a crash."""
    from slm_train.data.tokenizer import TokenizerError
    from slm_train.data.validate import LABELS, validate_packed_tokens
    result = _pack([1, 2], 2, "concat", tmp_path)
    _reforge_manifest(result.published_path, corrupt)
    with pytest.raises(TokenizerError) as raised:
        validate_packed_tokens(result.published_path / "manifest.json", strict=True)
    assert raised.value.label in LABELS
    assert raised.value.label == "FAIL"


# -- Task 5 follow-up: the dead builtin pin factory is gone (N2) ---------------


def test_builtin_pin_is_not_an_exported_pin_factory() -> None:
    """Only a real pinned file-set identity may manufacture a pin (N2)."""
    import slm_train.data.tokenizer as tokenizer_module

    assert not hasattr(tokenizer_module, "builtin_pin")
    from slm_train.data.tokenizer import pin_local_tokenizer
    assert callable(pin_local_tokenizer)


def test_document_masked_retained_tail_keeps_real_loss_bits(tmp_path: Path) -> None:
    """A retained sub-block document keeps its REAL loss bits; only pad is-loss-off.

    The previous packer sliced masks to the full-block boundary, so a document
    shorter than one block published an all-pad mask (every token loss-off) and
    training hit a NaN batch. This test pins the contract.
    """
    import numpy as np
    result = _pack(
        [1, 2, 3, 4, 5], 8, "document_masked", tmp_path,
        retain_tail=True, pad_token_id=9,
        documents=[{"tokens": [1, 2, 3, 4, 5], "mask": [1, 0, 1, 1, 1]}])
    assert result.counts["retained_partial_blocks"] == 1
    tokens = np.memmap(result.published_path / result.shards[0].path, dtype="<u4", mode="r")
    masks = np.memmap(result.published_path / result.shards[0].mask_path, dtype="<u1", mode="r")
    assert list(tokens) == [1, 2, 3, 4, 5, 9, 9, 9]
    assert list(masks) == [1, 0, 1, 1, 1, 0, 0, 0]


def test_ui_task_nl_keeps_the_task_chain_newline() -> None:
    """The ui contract's task/chain newline survives normalization (loss-mask boundary)."""
    from slm_train.data.dedupe import apply_normalization

    raw = "  a dashboard\twith kpi data \nUI1 spec\n  more "
    text, applied = apply_normalization(raw, "ui_task_nl")
    assert applied == "ui_task_nl"
    assert "\n" in text
    assert text.splitlines()[0] == "a dashboard with kpi data"
    assert text.splitlines()[1:] == ["UI1 spec", "more"]  # later newlines are kept too
