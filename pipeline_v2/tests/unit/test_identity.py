"""Unit tests for source/code/runtime/invocation/run identity."""
import json
from pathlib import Path

import pytest

from slm_train.identity.code import code_identity
from slm_train.identity.invocation import invocation_identity
from slm_train.identity.run_id import InputFingerprint, attempt_id, final_run_id
from slm_train.identity.source import hf_source_identity, local_merkle_manifest


def _fingerprint() -> InputFingerprint:
    return InputFingerprint(
        config_hash="c", data_hash="d", tokenizer_hash="t", tokenized_hash="td",
        model_hash="m", evaluator_hash="e", code_hash="s", release="r1",
        launch_profile_hash="lp", runtime_hash="rt", invocation_hash="iv",
    )


def test_fingerprint_is_field_sensitive_and_stable() -> None:
    """Brief step 1: every fingerprint field (and the release literal) must
    participate in the digest; identical fingerprints give identical run ids."""
    fp = _fingerprint()
    same = InputFingerprint(**fp.__dict__)
    assert fp.digest() == same.digest()
    assert final_run_id("exp", fp) == final_run_id("exp", same) == "exp-" + fp.digest()
    fields = ("config_hash", "data_hash", "tokenizer_hash", "tokenized_hash",
              "model_hash", "evaluator_hash", "code_hash",
              "launch_profile_hash", "runtime_hash", "invocation_hash")
    for field in fields:
        changed = InputFingerprint(**{**fp.__dict__, field: "changed"})
        assert fp.digest() != changed.digest(), field
        assert final_run_id("exp", fp) != final_run_id("exp", changed), field
    changed_release = InputFingerprint(**{**fp.__dict__, "release": "r2"})
    assert fp.digest() != changed_release.digest()


def test_final_run_id_rejects_unsafe_experiment_id() -> None:
    for bad in ("../escape", "", "a/b", " lead", "a" * 65):
        with pytest.raises(ValueError, match="safe"):
            final_run_id(bad, _fingerprint())


def test_attempt_id_is_derived_from_launch_inputs_and_excluded_from_final_identity() -> None:
    """Design 10.3: nonce + PID + process start time; never part of the final id."""
    a = attempt_id("exp", launch_nonce="nonce", pid=1234, process_start_time=123.0)
    assert a == attempt_id("exp", launch_nonce="nonce", pid=1234, process_start_time=123.0)
    for varied in ({"launch_nonce": "nonce2"}, {"pid": 999}, {"process_start_time": 124.0}):
        args = {"launch_nonce": "nonce", "pid": 1234, "process_start_time": 123.0}; args.update(varied)
        assert a != attempt_id("exp", **args)
    with pytest.raises(ValueError):
        attempt_id("exp", launch_nonce="", pid=1234, process_start_time=123.0)
    fp = _fingerprint()
    # the attempt id is not an input to the final run id at all
    assert final_run_id("exp", fp) == "exp-" + fp.digest()
    assert a not in str(fp.digest())


def test_local_merkle_manifest_is_exact_file_set(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_bytes(b"world")
    m1 = local_merkle_manifest(tmp_path, ["a.txt", "sub/b.txt"])
    m2 = local_merkle_manifest(tmp_path, ["a.txt", "sub/b.txt"])
    assert m1.root_hash == m2.root_hash
    assert m1.file_digests == m2.file_digests
    # content change changes the hash
    (tmp_path / "a.txt").write_bytes(b"jello")
    m3 = local_merkle_manifest(tmp_path, ["a.txt", "sub/b.txt"])
    assert m3.root_hash != m1.root_hash
    assert "sub/b.txt" in m3.file_digests
    with pytest.raises(KeyError):
        local_merkle_manifest(tmp_path, ["a.txt", " missing.txt"])


def test_hf_source_identity_is_pinned(tmp_path: Path) -> None:
    files = {"a.json": "hash-a", "b/c.json": "hash-b"}
    i1 = hf_source_identity("org/repo", revision="abc123", file_sha256=files)
    i2 = hf_source_identity("org/repo", revision="abc123", file_sha256=dict(files))
    assert i1 == i2
    assert i1.revision == "abc123"
    assert i1 != hf_source_identity("org/repo", revision="abc124", file_sha256=files)
    with pytest.raises(ValueError):
        hf_source_identity("org/repo", revision="", file_sha256=files)
    assert "hf" in json.dumps(i1.model_dump())


def test_source_identity_semantic_hash_exists() -> None:
    i1 = hf_source_identity("org/repo", revision="abc", file_sha256={"a": "h"})
    assert i1.semantic_manifest_hash() == i1.semantic_manifest_hash()
    assert i1.semantic_manifest_hash() != hf_source_identity(
        "org/repo", revision="abd", file_sha256={"a": "h"}).semantic_manifest_hash()


def test_code_identity_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    src = root / "pipeline_v2" / "slm_train"
    (src / "contracts").mkdir(parents=True)
    (src / "__init__.py").write_bytes(b"")
    (src / "contracts" / "base.py").write_bytes(b"# base")
    (root / "uv.lock").write_bytes(b"lock")
    c1 = code_identity(root)
    c2 = code_identity(root)
    assert c1 == c2
    # a changed .py source file changes the root hash (I4)
    (src / "contracts" / "base.py").write_bytes(b"# base v2")
    c3 = code_identity(root)
    assert c3.root_hash != c1.root_hash
    assert "pipeline_v2/slm_train/contracts/base.py" in c3.file_digests
    # a changed uv.lock changes the root hash (I4)
    (root / "uv.lock").write_bytes(b"lock v2")
    c4 = code_identity(root)
    assert c4.root_hash != c3.root_hash
    assert c4.file_digests["uv.lock"] != c3.file_digests["uv.lock"]


def test_code_identity_fails_closed_on_missing_uv_lock(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    src = root / "pipeline_v2" / "slm_train"
    (src / "contracts").mkdir(parents=True)
    (src / "__init__.py").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="uv.lock"):
        code_identity(root)


def test_runtime_identity_records_supplied_facts() -> None:
    from slm_train.identity.runtime import RuntimeIdentity
    r1 = RuntimeIdentity.model_validate({
        "schema_version": 1,
        "python_version": "3.12.8",
        "venv_path": ".venv",
        "package_lock": {"pinned": True},
        "os": "Windows-11",
        "cuda": "12.6",
        "driver": "560.0",
        "device_class": "capable-e",
        "device_name": "consumer-gpu",
    })
    assert r1.schema_version == 1
    assert r1.device_class == "capable-e"
    r2 = RuntimeIdentity.model_validate({
        "schema_version": 1,
        "python_version": "3.12.8", "venv_path": ".venv",
        "package_lock": {"pinned": True}, "os": "Windows-11",
        "cuda": "12.6", "driver": "560.0",
        "device_class": "not-recorded", "device_name": "not-recorded",
    })
    assert r1 != r2  # device facts matter for runtime identity


def test_invocation_working_directory_is_repository_relative() -> None:
    """Design 6.3: the invocation WD must be an exact repository-relative path."""
    good = invocation_identity(
        command=["python"], working_directory="runs/v2", environment={})
    assert good.working_directory == "runs/v2"
    nested = invocation_identity(
        command=["python"], working_directory="pipeline_v2/slm_train", environment={})
    assert nested.working_directory == "pipeline_v2/slm_train"
    for bad in ("", "   ", "..", "../x", "C:foo", "C:/x", "/abs", "//server/share/x",
                "a/../b", "..\\x", ".", "./runs", "runs/./x", "runs/.", "runs/..",
                "runs/", "runs\\", "runs//v2", "/", "runs\x00/v2", "runs/v2\x00",
                "runs/./"):
        with pytest.raises(ValueError):
            invocation_identity(command=["python"], working_directory=bad, environment={})


def test_invocation_identity_excludes_non_allowlisted_env() -> None:
    i1 = invocation_identity(
        command=["python", "-m", "slm_train.cli", "train"],
        working_directory="runs/v2",
        environment={"PATH": "z", "TOKEN": "secret", "POSE_LENGTH": "4"},
    )
    i2 = invocation_identity(
        command=["python", "-m", "slm_train.cli", "train"],
        working_directory="runs/v2",
        environment={"PATH": "z", "TOKEN": "leaked", "POSE_LENGTH": "4"},
    )
    assert i1 == i2  # secret env value must not change identity
    i3 = invocation_identity(
        command=["python", "-m", "slm_train.cli", "train"],
        working_directory="runs/v2",
        environment={"PATH": "x", "POSE_LENGTH": "4"},
    )
    assert i1 != i3  # allowlisted values do participate
