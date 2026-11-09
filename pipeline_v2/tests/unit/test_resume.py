"""Task 9 unit tests, part 1: exact resume, cursor, and checkpoint selection."""
from __future__ import annotations

import json

import pytest

from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.training.checkpoint import (
    CheckpointIncomplete,
    checkpoint_dir,
    is_complete,
    publish_checkpoint,
)
from slm_train.training.resume import (
    RESUME_IDENTITY_KEYS,
    ResumeMismatch,
    assert_exact_resume,
    resume_identity_differences,
)
from slm_train.training.selection import (
    select_checkpoint,
    verify_checkpoint,
)
from slm_train.training.state import (
    CheckpointState,
    STATE_ROLES,
    TrainingCursor,
)


def identity(**changes: str) -> dict:
    value = {key: "same" for key in RESUME_IDENTITY_KEYS}
    value.update(changes)
    return value


# --- exact resume ---------------------------------------------------------


def test_resume_accepts_an_identical_identity():
    assert_exact_resume(identity(), identity())


@pytest.mark.parametrize("key", RESUME_IDENTITY_KEYS)
def test_resume_rejects_a_change_to_any_single_identity_key(key):
    with pytest.raises(ResumeMismatch, match=key):
        assert_exact_resume(identity(), identity(**{key: "different"}))


def test_resume_rejects_changed_evaluator():
    with pytest.raises(ResumeMismatch, match="evaluator"):
        assert_exact_resume(identity(evaluator="e1"), identity(evaluator="e2"))


def test_resume_rejects_changed_command_and_device():
    with pytest.raises(ResumeMismatch, match="command"):
        assert_exact_resume(
            identity(command="run", device="cuda:0"),
            identity(command="run --resume", device="cuda:0"),
        )
    with pytest.raises(ResumeMismatch, match="device"):
        assert_exact_resume(
            identity(command="run", device="cuda:0"),
            identity(command="run", device="cuda:1"),
        )


@pytest.mark.parametrize("key", RESUME_IDENTITY_KEYS)
def test_resume_refuses_an_absent_identity_key_rather_than_defaulting(key):
    saved = identity()
    current = identity()
    del current[key]
    with pytest.raises(ResumeMismatch, match=key):
        assert_exact_resume(saved, current)


@pytest.mark.parametrize("key", RESUME_IDENTITY_KEYS)
def test_resume_refuses_an_empty_identity_value_rather_than_treating_it_as_same(key):
    with pytest.raises(ResumeMismatch, match=key):
        assert_exact_resume(identity(), identity(**{key: ""}))


def test_resume_reports_every_differing_key_not_just_the_first():
    differences = resume_identity_differences(
        identity(), identity(model="m2", device="cuda:3", code="c2"),
    )
    assert set(differences) == {"model", "device", "code"}


def test_resume_refuses_a_non_mapping_identity():
    with pytest.raises(ResumeMismatch):
        assert_exact_resume(identity(), ["not", "a", "mapping"])


# --- cursor ---------------------------------------------------------------


def test_cursor_round_trip_keeps_microbatch_boundary():
    cursor = TrainingCursor(
        examples_seen=17, microbatch_position=2,
        accumulation_boundary=3, epoch=1,
    )
    assert TrainingCursor.from_dict(cursor.to_dict()) == cursor


@pytest.mark.parametrize("field", [
    "examples_seen", "microbatch_position", "accumulation_boundary", "epoch",
])
def test_cursor_refuses_a_negative_position(field):
    with pytest.raises(ValueError, match=field):
        TrainingCursor(**{field: -1})


def test_cursor_refuses_a_microbatch_position_past_the_accumulation_boundary():
    with pytest.raises(ValueError, match="microbatch_position"):
        TrainingCursor(microbatch_position=4, accumulation_boundary=3)


def test_cursor_refuses_a_non_integer_position():
    with pytest.raises(ValueError, match="epoch"):
        TrainingCursor(epoch=1.5)


def test_cursor_refuses_a_missing_or_unexpected_field():
    with pytest.raises(ValueError, match="missing"):
        TrainingCursor.from_dict({"examples_seen": 1})
    with pytest.raises(ValueError, match="unexpected"):
        TrainingCursor.from_dict(TrainingCursor().to_dict() | {"extra": 1})


def test_cursor_reports_whether_it_sits_on_an_accumulation_boundary():
    assert TrainingCursor(2, 2, 2).at_boundary
    assert not TrainingCursor(2, 1, 2).at_boundary


# --- checkpoint state -----------------------------------------------------


def full_state() -> CheckpointState:
    return CheckpointState(
        cursor=TrainingCursor(examples_seen=8, microbatch_position=2,
                           accumulation_boundary=2, epoch=1),
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


def test_checkpoint_state_names_every_required_role():
    files = full_state().files()
    for role in STATE_ROLES:
        if role == "worker_rng":
            assert "worker_rng.0" in files
        else:
            assert role in files, role


@pytest.mark.parametrize("role", [
    "model", "optimizer", "scheduler", "global_rng", "sampler",
    "dataloader_generator", "early_stop", "evaluation", "lifecycle",
])
def test_checkpoint_state_refuses_to_drop_a_required_role(role):
    values = full_state().to_dict()
    values[role] = ""
    with pytest.raises(ValueError, match=role):
        CheckpointState.from_dict(values)


def test_checkpoint_state_refuses_an_unknown_role():
    with pytest.raises(ValueError, match="unknown checkpoint state role"):
        full_state().file_for("not_a_role")


# --- atomic checkpoint publication -----------------------------------------


def payloads(state: CheckpointState) -> dict:
    return {role: (role + "-payload").encode("utf-8") for role in state.files()}


def manifest(step: int = 100) -> CheckpointManifest:
    return CheckpointManifest(
        schema_version=1,
        run_id="run-1",
        config_hash="c" * 64,
        tokenizer_hash="t" * 64,
        model_hash="m" * 64,
        code_hash="k" * 64,
        global_step=step,
        epoch=1.0,
        files=(),
        complete=False,
    )


def test_a_published_checkpoint_is_complete_and_verifiable(tmp_path):
    state = full_state()
    destination = publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    assert is_complete(tmp_path, 1)
    assert verify_checkpoint(tmp_path, 1) == []
    assert (destination / "COMPLETE.json").is_file()


def test_a_published_checkpoint_records_a_hash_and_size_for_every_state_file(tmp_path):
    state = full_state()
    from slm_train.training.selection import read_manifest

    publish_checkpoint(tmp_path, 2, state, payloads(state), manifest(200))
    published = read_manifest(tmp_path, 2)
    assert published.complete is True
    assert published.global_step == 200
    roles = {entry.role for entry in published.files}
    assert roles == set(state.files())
    for entry in published.files:
        assert len(entry.sha256) == 64
        assert entry.size_bytes > 0


def test_a_checkpoint_missing_a_state_payload_is_never_published(tmp_path):
    state = full_state()
    incomplete = payloads(state)
    del incomplete["model"]
    with pytest.raises(ValueError, match="model"):
        publish_checkpoint(tmp_path, 3, state, incomplete, manifest())
    assert not is_complete(tmp_path, 3), "no marker may exist after a refusal"


def test_a_checkpoint_with_an_undeclared_payload_is_refused(tmp_path):
    state = full_state()
    extra = payloads(state)
    extra["surprise.pt"] = b"x"
    with pytest.raises(ValueError, match="surprise"):
        publish_checkpoint(tmp_path, 4, state, extra, manifest())
    assert not is_complete(tmp_path, 4)


def test_republishing_a_checkpoint_number_fails_instead_of_overwriting(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 5, state, payloads(state), manifest())
    with pytest.raises(FileExistsError):
        publish_checkpoint(tmp_path, 5, state, payloads(state), manifest())


@pytest.mark.parametrize("bad", [-1, "3", 1.5, True])
def test_a_non_integer_checkpoint_number_is_refused(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        checkpoint_dir("root", bad)


# --- selection ------------------------------------------------------------


def test_selection_prefers_the_highest_verified_checkpoint(tmp_path):
    state = full_state()
    for number in (1, 2, 3):
        publish_checkpoint(tmp_path, number, state, payloads(state), manifest(number * 10))
    result = select_checkpoint(tmp_path)
    assert result.number == 3
    assert result.reason == "verified"


def test_a_checkpoint_with_no_completion_marker_is_never_selected(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 7, state, payloads(state), manifest())
    (checkpoint_dir(tmp_path, 7) / "COMPLETE.json").unlink()
    result = select_checkpoint(tmp_path)
    assert result.number is None
    assert "no completion marker" in " ".join(result.rejected)


def test_a_partial_directory_with_no_manifest_is_invisible_to_selection(tmp_path):
    partial = checkpoint_dir(tmp_path, 9)
    (partial / "state").mkdir(parents=True)
    (partial / "state" / "model.pt").write_bytes(b"leftover")
    result = select_checkpoint(tmp_path)
    assert result.number is None
    assert result.reason == "no complete checkpoint has been published"


def test_a_corrupted_state_file_is_skipped_and_reported(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    publish_checkpoint(tmp_path, 2, state, payloads(state), manifest())
    # same length, different bytes: exercises the hash check, not the size check
    original = (checkpoint_dir(tmp_path, 2) / "state" / "model.pt").read_bytes()
    (checkpoint_dir(tmp_path, 2) / "state" / "model.pt").write_bytes(b"X" * len(original))
    result = select_checkpoint(tmp_path)
    assert result.number == 1, "a corrupt newest checkpoint must fall back, not fail"
    assert any("hash mismatch" in line for line in result.rejected)


def test_a_truncated_state_file_is_reported_as_a_size_mismatch(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    publish_checkpoint(tmp_path, 2, state, payloads(state), manifest())
    (checkpoint_dir(tmp_path, 2) / "state" / "model.pt").write_bytes(b"short")
    result = select_checkpoint(tmp_path)
    assert result.number == 1
    assert any("size mismatch" in line for line in result.rejected)


def test_selection_blocks_rather_than_restarting_when_every_candidate_fails(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    (checkpoint_dir(tmp_path, 1) / "state" / "optimizer.pt").unlink()
    result = select_checkpoint(tmp_path)
    assert result.number is None
    assert result.found is False
    assert "every candidate" in result.reason


def test_a_manifest_edited_to_drop_a_state_file_is_not_selectable(tmp_path):
    """The marker binds the manifest bytes, so an edited manifest cannot verify."""
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    (checkpoint_dir(tmp_path, 1) / "state" / "optimizer.pt").unlink()
    document = json.loads((checkpoint_dir(tmp_path, 1) / "manifest.json").read_text("utf-8"))
    document["files"] = [f for f in document["files"] if f["role"] != "optimizer"]
    (checkpoint_dir(tmp_path, 1) / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    problems = verify_checkpoint(tmp_path, 1)
    assert problems, "an edited manifest must never verify"
    assert any("completion marker" in problem for problem in problems)
    assert select_checkpoint(tmp_path).number is None


def test_a_manifest_emptied_of_its_file_list_is_not_selectable(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    document = json.loads((checkpoint_dir(tmp_path, 1) / "manifest.json").read_text("utf-8"))
    document["files"] = []
    (checkpoint_dir(tmp_path, 1) / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    assert verify_checkpoint(tmp_path, 1) != []
    assert select_checkpoint(tmp_path).number is None


def test_a_deleted_state_file_is_caught_by_the_markers_certified_hashes(tmp_path):
    state = full_state()
    publish_checkpoint(tmp_path, 1, state, payloads(state), manifest())
    (checkpoint_dir(tmp_path, 1) / "state" / "scheduler.pt").unlink()
    problems = verify_checkpoint(tmp_path, 1)
    assert any("completion marker" in problem for problem in problems)
    assert select_checkpoint(tmp_path).number is None


def test_selection_on_an_empty_run_root_is_a_reason_not_a_silent_restart(tmp_path):
    result = select_checkpoint(tmp_path)
    assert result.number is None
    assert result.found is False
    assert result.reason

def test_a_rerun_reclaims_a_debris_destination_missing_its_marker(tmp_path):
    """An interrupted publish leaves a directory without the completion marker.

    That residue is not user evidence; a rerun must be able to rebuild its own
    checkpoint at the same step, while a complete checkpoint is never touched.
    """
    state = full_state()

    # normal publish, then simulate debris: strip the marker and manifest.
    first = publish_checkpoint(tmp_path, 3, state, payloads(state), manifest(300))
    (first / "COMPLETE.json").unlink()

    second = publish_checkpoint(tmp_path, 3, state, payloads(state), manifest(300))
    assert is_complete(tmp_path, 3)
    assert second == first
    assert (second / "COMPLETE.json").is_file()


def test_a_complete_checkpoint_is_never_recycled(tmp_path):
    state = full_state()
    destination = publish_checkpoint(tmp_path, 5, state, payloads(state), manifest(500))
    import pytest as _pytest

    with _pytest.raises(FileExistsError):
        publish_checkpoint(tmp_path, 5, state, payloads(state), manifest(500))
    assert is_complete(tmp_path, 5)
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        p.name for p in destination.iterdir()
    )
