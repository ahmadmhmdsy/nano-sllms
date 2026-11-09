"""Unit tests for versioned V2 contracts (immutable/mutable envelopes)."""
import typing

import pytest
from pydantic import ValidationError

from slm_train.contracts import (
    CheckpointManifest,
    DataManifest,
    EvaluationReport,
    EvaluatorManifest,
    FinalPublicationManifest,
    LogIndexContract,
    ModelManifest,
    RunIdentityManifest,
    SourceManifest,
    StatusContract,
    TokenizerManifest,
    TokenizedDataManifest,
    ValidationLabel,
)

IMMUTABLE_RUN = {
    "schema_version": 1,
    "run_id": "core-abc",
    "experiment_id": "core",
    "release": "r1",
    "config_path": "pipeline_v2/configs/base/r1.yaml",
    "config_hash": "c",
    "data_hash": "d",
    "tokenizer_hash": "t",
    "tokenized_hash": "td",
    "model_hash": "m",
    "evaluator_hash": "e",
    "code_hash": "s",
    "dependency_hash": "dep",
    "runtime_fingerprint": "rf",
    "working_directory": "runs/v2/core/core-abc",
    "launch_profile": "pilot-smoke",
    "exact_command": ("python", "-m", "slm_train.cli", "train"),
    "evidence_root": "runs/v2/core/core-abc",
}


class TestEnvelopeRules:
    def test_validation_vocabulary_is_exact(self) -> None:
        for label in ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"):
            StatusContract.model_validate({"schema_version": 1, "label": label})
        for bad in ("pass", "OK", "PENDING", "", None):
            with pytest.raises(ValidationError):
                StatusContract.model_validate({"schema_version": 1, "label": bad})

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SourceManifest.model_validate({"schema_version": 1, "surprise": 1})

    def test_models_are_frozen(self) -> None:
        m = DataManifest.model_validate(
            {"schema_version": 1, "content_hash": "h"}
        )
        with pytest.raises(ValidationError):
            m.content_hash = "y"

    def test_schema_version_required_and_positive(self) -> None:
        with pytest.raises(ValidationError):
            SourceManifest.model_validate({"source_ref": "x"})
        with pytest.raises(ValidationError):
            SourceManifest.model_validate({"schema_version": 0, "source_ref": "x"})


class TestRunIdentity:
    def test_full_identity_field_set(self) -> None:
        m = RunIdentityManifest.model_validate({
            **IMMUTABLE_RUN,
            "git_commit": "18106191a2a0d33b3fc8a9635166dedf54a486a6",
            "git_worktree_state": {"dirty": "user-changes"},
            "python_identity": {"version": "3.12"},
            "package_identity": {"uv.lock": "sha256:lock"},
            "device_identity": {"class": "capable-e"},
            "driver_identity": {"version": "560.0"},
            "parent_model_identity": "base",
            "parent_checkpoint_identity": "ckpt-100",
            "promotion_predecessor": {
                "predecessor_run_id": "core-old",
                "predecessor_checkpoint_identity": "ckpt-90",
                "predecessor_data_hash": "d0",
                "predecessor_evaluator_hash": "e0",
                "predecessor_gate_set": "smoke",
            },
        })
        assert m.evidence_root == "runs/v2/core/core-abc"
        assert m.promotion_predecessor.predecessor_run_id == "core-old"

    def test_identity_hash_is_immutable_excluding_created_at(self) -> None:
        a = RunIdentityManifest.model_validate({
            **IMMUTABLE_RUN, "created_at": "2026-01-01T00:00:00Z"})
        b = RunIdentityManifest.model_validate({
            **IMMUTABLE_RUN, "created_at": "2099-01-01T00:00:00Z"})
        assert a.semantic_manifest_hash() == b.semantic_manifest_hash() == a.semantic_manifest_hash()

    def test_lease_and_status_are_not_identity_fields(self) -> None:
        data = {**IMMUTABLE_RUN, "lease_state": "held", "current_status": "running"}
        with pytest.raises(ValidationError):
            RunIdentityManifest.model_validate(data)

    def test_any_identity_change_changes_the_hash(self) -> None:
        base = RunIdentityManifest.model_validate(IMMUTABLE_RUN)
        for field in ("config_hash", "data_hash", "tokenizer_hash", "tokenized_hash",
                      "model_hash", "evaluator_hash", "code_hash", "dependency_hash",
                      "runtime_fingerprint", "working_directory", "launch_profile",
                      "exact_command", "evidence_root"):
            changed_data = dict(IMMUTABLE_RUN)
            changed_data[field] = ("x",) if field == "exact_command" else "changed"
            changed = RunIdentityManifest.model_validate(changed_data)
            assert base.semantic_manifest_hash() != changed.semantic_manifest_hash(), field

    def test_evidence_root_must_be_repository_relative(self) -> None:
        for bad in ("/abs/evidence", "C:/evidence", "../evidence", "..\\evidence"):
            with pytest.raises(ValidationError):
                RunIdentityManifest.model_validate({**IMMUTABLE_RUN, "evidence_root": bad})
        assert RunIdentityManifest.model_validate({
            **IMMUTABLE_RUN, "evidence_root": "runs/v2/core/r"}).evidence_root


class TestDataManifest:
    def test_full_field_set(self) -> None:
        m = DataManifest.model_validate({
            "schema_version": 1,
            "source_ids": ("mes-3k",),
            "source_roles": {"mes-3k": "train"},
            "source_identity": {"kind": "hf", "repo_id": "org/repo", "revision": "abc123"},
            "source_fingerprint": "fp",
            "freshness_policy": "pinned-revision",
            "supersession_policy": "explicit-generation",
            "dataset_config": {"name": "mes-3k"},
            "split_policy": {"train": 0.9},
            "filters": {"min_len": 10},
            "normalization": "nfkc",
            "dedupe_mode": "simhash",
            "dedupe_salt": "salt",
            "document_id_fields": ("doc_id",),
            "document_counts": {"train": 100},
            "license_note": "CC-BY",
            "provenance_note": "downloaded",
            "overlap_report": {"leakage": 0},
            "truncation_counts": {"train": 0},
            "rejection_counts": {"train": 3},
            "regeneration_command": ("python", "-m", "slm_train.cli", "prepare"),
            "content_hash": "h",
        })
        assert m.semantic_manifest_hash()


class TestTokenizer:
    def test_full_field_set(self) -> None:
        TokenizerManifest.model_validate({
            "schema_version": 1,
            "repository_identity": "pinned-repo",
            "file_identity": "tokenizer.json",
            "content_hash": "h",
            "vocabulary_size": 65536,
            "max_token_id": 65535,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "special_tokens": ("<bos>", "<eos>", "<pad>"),
            "context_behavior": "truncate",
            "offset_behavior": "none",
            "package_version": "tokenizers==0.23.2",
        })


class TestTokenized:
    def _shards(self):
        return ({"path": "data/v2/tokens/shard-0.bin", "size_bytes": 10, "sha256": "h0"},)

    def test_mask_path_required_for_document_masked(self) -> None:
        with pytest.raises(ValidationError):
            TokenizedDataManifest.model_validate({
                "schema_version": 1,
                "packing_mode": "document_masked",
                "mask_path": None,
                "sequence_length": 512,
                "shards": self._shards(),
                "source_data_hash": "d",
            })
        ok = TokenizedDataManifest.model_validate({
            "schema_version": 1,
            "packing_mode": "document_masked",
            "mask_path": "data/v2/tokens/masks.npz",
            "sequence_length": 512,
            "shards": self._shards(),
            "dropped_tail_tokens": 0,
            "overflow_blocks": 0,
            "token_range": (0, 65535),
            "document_boundaries_path": "data/v2/tokens/boundaries.npy",
            "source_data_hash": "d",
        })
        assert ok.mask_path == "data/v2/tokens/masks.npz"

    def test_shard_and_mask_paths_must_be_repository_relative(self) -> None:
        for bad_mask in ("/abs/masks.npz", "C:/masks.npz", "../masks.npz"):
            with pytest.raises(ValidationError):
                TokenizedDataManifest.model_validate({
                    "schema_version": 1,
                    "packing_mode": "document_masked",
                    "mask_path": bad_mask,
                    "sequence_length": 512,
                    "shards": self._shards(),
                    "source_data_hash": "d",
                })
        with pytest.raises(ValidationError):
            TokenizedDataManifest.model_validate({
                "schema_version": 1,
                "packing_mode": "none",
                "sequence_length": 512,
                "shards": ({"path": "/abs/0.bin", "size_bytes": 1, "sha256": "h"},),
                "source_data_hash": "d",
            })


class TestModelAndEvaluator:
    def test_model_full_field_set(self) -> None:
        ModelManifest.model_validate({
            "schema_version": 1,
            "architecture": "nano-dec-preview-25m",
            "config_hash": "cfg",
            "initialization": "from-foundation",
            "base_model_identity": "model:smoke-e-tiny",
            "parent_checkpoint_identity": "smoke/ckpt-2000",
            "parent_checkpoint_content_hash": "phash",
            "capability_metadata": {"context": 4096},
        })

    def test_evaluator_manifest_hash_excludes_hash_field_itself(self) -> None:
        base = {"schema_version": 1, "evaluator_ids": ("e1",), "evaluator_versions": {"e1": "v1"}}
        m = EvaluatorManifest.model_validate(base)
        digest = m.semantic_manifest_hash()
        published = EvaluatorManifest.model_validate({**base, "manifest_hash": digest})
        assert published.manifest_hash == digest
        assert published.semantic_manifest_hash() == m.semantic_manifest_hash()

    def test_evaluator_full_field_set(self) -> None:
        EvaluatorManifest.model_validate({
            "schema_version": 1,
            "evaluator_ids": ("abdou",),
            "evaluator_versions": {"abdou": "v1"},
            "suite_gates": ("smoke-suite",),
            "metric_schemas": {"asis_valid_5": "float"},
            "cadence": "every-checkpoint",
            "thresholds": {"asis_valid_5": 0.85},
            "threshold_directions": {"asis_valid_5": "ge"},
            "manifest_hash": "h",
        })


class TestCheckpointAndEvaluation:
    def test_checkpoint_full_field_set(self) -> None:
        CheckpointManifest.model_validate({
            "schema_version": 1,
            "run_id": "run",
            "config_hash": "c", "data_hash": "d", "tokenizer_hash": "t",
            "tokenized_hash": "td", "model_hash": "m", "evaluator_hash": "e",
            "code_hash": "s", "dependency_hash": "dep",
            "global_step": 100, "epoch": 2.0,
            "model_state_identity": "ms", "optimizer_state_identity": "os",
            "scheduler_state_identity": "ss", "scaler_state_identity": "cs",
            "rng_state_identity": "rs", "sampler_state_identity": "ps",
            "files": ({"path": "optimizer.pt", "size_bytes": 1, "sha256": "h"},),
            "complete": True,
            "evaluation_state": "evaluated",
            "selection_state": "selected",
        })

    def test_checkpoint_is_immutable_and_requires_files(self) -> None:
        m = CheckpointManifest.model_validate({
            "schema_version": 1, "config_hash": "c", "tokenizer_hash": "t",
            "model_hash": "m", "code_hash": "s", "global_step": 1,
            "files": (), "complete": False,
        })
        with pytest.raises(ValidationError):
            m.complete = True

    def test_evaluation_report_full_field_set(self) -> None:
        report = EvaluationReport.model_validate({
            "schema_version": 1,
            "evaluator": "abdou", "evaluator_version": "v1",
            "expected_suite_gates": ("smoke",), "actual_suite_gates": ("smoke",),
            "expected_evaluator_manifest_hash": "h0", "actual_evaluator_manifest_hash": "h0",
            "run_id": "run", "checkpoint_identity": "ckpt-100",
            "data_hash": "d", "tokenizer_hash": "t",
            "device_identity": "capable-e", "seed": 7, "split_identity": "abdou-test",
            "metric_schema_identity": "schema-1",
            "metric_definitions": {"asis_valid_5": "float"},
            "metric_values": {"asis_valid_5": 0.9},
            "baseline": 0.8, "gate_threshold": 0.85, "threshold_direction": "ge",
            "status": "PASS",
            "artifact_paths": ("evaluations/abdou.json",),
        })
        assert report.status == "PASS"

    def test_evaluation_status_uses_exact_vocabulary(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationReport.model_validate({
                "schema_version": 1,
                "evaluator": "e", "evaluator_version": "v1",
                "run_id": "r", "status": "ok",
            })


class TestFinalPublication:
    def test_full_field_set(self) -> None:
        FinalPublicationManifest.model_validate({
            "schema_version": 1,
            "run_id": "run",
            "evidence_root": "runs/v2/core/run",
            "selected_checkpoint": "ckpt-2000",
            "final_files": ({"path": "model.safetensors", "size_bytes": 10, "sha256": "h"},),
            "read_back_passed": True,
            "payload_hashes": {"model.safetensors": "h"},
        })


class TestSemanticManifestSubclasses:
    """C1: every manifest subclass must inherit a valid MUTABLE_FIELDS and
    compute its semantic manifest hash without AttributeError."""

    @staticmethod
    def _all_subclasses(cls) -> list[type]:
        found: list[type] = []
        for sub in cls.__subclasses__():
            found.append(sub)
            found.extend(TestSemanticManifestSubclasses._all_subclasses(sub))
        return found

    @staticmethod
    def _filler(annotation: object) -> object:
        from pydantic import BaseModel

        if annotation is int:
            return 0
        if annotation is bool:
            return False
        if annotation is str:
            return "x"
        origin = typing.get_origin(annotation)
        if origin is tuple or origin is list or origin is set or origin is frozenset:
            return ()
        if origin is dict or annotation is dict:
            return {}
        for candidate in typing.get_args(annotation):
            if candidate is not type(None) and isinstance(candidate, type) and issubclass(candidate, BaseModel):
                return None  # nested model placeholder: hashing tolerates JSON-none
        return None

    def test_every_manifest_subclass_hashes(self) -> None:
        import slm_train.identity.code  # noqa: F401
        import slm_train.identity.invocation  # noqa: F401
        import slm_train.identity.runtime  # noqa: F401
        from slm_train.contracts.base import SemanticManifest

        subclasses = self._all_subclasses(SemanticManifest)
        assert subclasses
        for cls in subclasses:
            assert isinstance(cls.MUTABLE_FIELDS, tuple), cls.__name__
            payload: dict[str, object] = {
                "schema_version": 1,
                "mutable_fields": (),
            }
            for name, field in cls.model_fields.items():
                if name == "schema_version":
                    continue
                payload[name] = self._filler(field.annotation)
            instance = cls.model_construct(**payload)
            assert isinstance(instance.semantic_manifest_hash(), str) and instance.semantic_manifest_hash(), cls.__name__

    def test_source_manifest_hash_distinguishes_content(self) -> None:
        a = SourceManifest.model_validate({"schema_version": 1, "kind": "hf",
                                           "repo_id": "org/repo", "revision": "abc",
                                           "file_digests": {"a.json": "h1"}})
        b = SourceManifest.model_validate({"schema_version": 1, "kind": "hf",
                                           "repo_id": "org/repo", "revision": "abc",
                                           "file_digests": {"a.json": "h2"}})
        assert a.semantic_manifest_hash() != b.semantic_manifest_hash()


class TestStatusIsOperationalOnly:
    def test_status_timestamps_and_lease_are_mutable(self) -> None:
        s = StatusContract.model_validate({"schema_version": 1, "label": "PASS"})
        s.label = "FAIL"  # mutable by design
        from slm_train.contracts import LeaseContract, HeartbeatContract, RecoveryContract
        LeaseContract.model_validate({"schema_version": 1, "holder": "gpu-0"})
        HeartbeatContract.model_validate({"schema_version": 1})
        RecoveryContract.model_validate({"schema_version": 1, "note": "recovering"})
