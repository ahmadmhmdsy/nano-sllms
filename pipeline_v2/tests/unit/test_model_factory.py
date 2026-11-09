"""Task 6 capability tests: constrained R1 causal-LM factory."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from slm_train.config.models import ModelConfig
from slm_train.contracts.model import ModelManifest
from slm_train.errors import PipelineError
from slm_train.models.factory import ModelCapabilityError, build_causal_lm
from slm_train.models.manifest import build_model_manifest

from slm_train.training.events import (
    TRAINING_EVENT_NAMES,
    EventEmitter,
    TrainingEvent,
)
from slm_train.training.hf_causal_lm import resolve_output_dir
from slm_train.training.spec import TrainingResult



def tiny_model_config(**overrides) -> ModelConfig:
    """Tiny valid R1 GQA model configuration used across these tests."""
    values = {
        "architecture": "llama",
        "hidden": 32,
        "intermediate": 64,
        "layers": 1,
        "heads": 4,
        "kv_heads": 2,
        "context": 16,
        "tied_embeddings": False,
        "precision": "fp16",
        "attention": "sdpa",
        "base_model": None,
    }
    return ModelConfig(**{**values, **overrides})


# --- capability gates (brief names first) ------------------------------------


def test_bf16_is_rejected():
    with pytest.raises(ModelCapabilityError, match="bf16"):
        build_causal_lm(tiny_model_config(precision="bf16"))


def test_flash_attention_is_rejected():
    with pytest.raises(ModelCapabilityError, match="flash_attention_2"):
        build_causal_lm(tiny_model_config(attention="flash_attention_2"))


def test_gqa_uses_sdpa():
    model = build_causal_lm(tiny_model_config())
    assert model.config.model_type == "llama"


def test_unsupported_architecture_is_rejected():
    from slm_train.models.causal_lm import validate_model_capabilities

    for architecture in ("gpt2", "mistral", "qwen2"):
        with pytest.raises(ModelCapabilityError, match="architecture"):
            validate_model_capabilities(
                hidden=32, heads=4, kv_heads=2, context=16,
                layers=1, intermediate=64,
                precision="fp16", attention="sdpa",
                architecture=architecture,
            )


def test_bad_head_divisibility_is_rejected():
    with pytest.raises(ModelCapabilityError, match="head"):
        build_causal_lm(tiny_model_config(hidden=66, heads=4, kv_heads=2))
    with pytest.raises(ModelCapabilityError, match="head"):
        build_causal_lm(tiny_model_config(heads=3, kv_heads=2))


def test_data_context_mismatch_is_rejected():
    with pytest.raises(ModelCapabilityError, match="context"):
        build_causal_lm(tiny_model_config(), data_context=8)


def test_r2_capability_is_rejected():
    from slm_train.config.models import R1_CAPABILITIES, R2_CAPABILITIES

    assert not (R2_CAPABILITIES & R1_CAPABILITIES)
    with pytest.raises(ModelCapabilityError, match="sft"):
        build_causal_lm(tiny_model_config(), capabilities=("pretrain", "sft"))
    with pytest.raises(ModelCapabilityError, match="lora"):
        build_causal_lm(tiny_model_config(), capabilities=("lora",))


def test_base_model_mounting_is_r2_only():
    with pytest.raises(ModelCapabilityError, match="base_model"):
        build_causal_lm(tiny_model_config(base_model="org/tiny-base"))


# --- acceptance --------------------------------------------------------------


def test_valid_fp16_sdpa_config_is_accepted():
    model = build_causal_lm(tiny_model_config())
    assert model.config.hidden_size == 32
    assert model.config.num_hidden_layers == 1
    assert model.config.intermediate_size == 64
    assert model.config.max_position_embeddings == 16
    assert model.config.tie_word_embeddings is False
    assert model.dtype == torch.float16


def test_built_model_has_real_gqa_head_counts():
    model = build_causal_lm(tiny_model_config())
    assert model.config.num_attention_heads == 4
    assert model.config.num_key_value_heads == 2
    attention = model.model.layers[0].self_attn
    assert attention.num_key_value_groups == 2  # real GQA grouping


def test_built_model_stays_on_cpu_and_never_touches_cuda():
    model = build_causal_lm(tiny_model_config())
    for param in model.parameters():
        assert param.device.type == "cpu"


# --- model manifest ----------------------------------------------------------


def test_model_manifest_records_every_required_field():
    model_config = tiny_model_config()
    model = build_causal_lm(model_config)
    manifest = build_model_manifest(
        model_config=model_config,
        model=model,
        attn_implementation="sdpa",
        source_hash="0" * 64,
    )
    assert isinstance(manifest, ModelManifest)
    info = dict(manifest.capability_metadata)
    assert manifest.architecture == "llama"
    for field in (
        "hidden", "intermediate", "layers", "heads", "kv_heads",
        "kv_head_groups", "context", "tied_embeddings", "precision",
        "attention", "attn_implementation", "package_versions",
        "source_hash", "num_parameters",
    ):
        assert field in info, "missing manifest field: " + field
    assert info["heads"] == 4
    assert info["kv_heads"] == 2
    assert info["context"] == 16
    assert info["tied_embeddings"] is False
    assert info["precision"] == "fp16"
    assert info["attention"] == "sdpa"
    assert "torch" in info["package_versions"]
    assert "transformers" in info["package_versions"]
    # from-scratch model: no invented base or parent identity
    assert manifest.base_model_identity is None
    assert manifest.parent_checkpoint_identity is None
    assert manifest.parent_checkpoint_content_hash is None


def test_base_model_identity_recorded_only_when_given():
    model = build_causal_lm(tiny_model_config())
    manifest = build_model_manifest(
        model_config=tiny_model_config(),
        model=model,
        attn_implementation="sdpa",
        source_hash="1" * 64,
        base_model="org/parent",
    )
    assert manifest.base_model_identity == "org/parent"


# --- typed training events ----------------------------------------------------


def test_event_names_are_typed_and_monotonic():
    assert TRAINING_EVENT_NAMES == (
        "prepare", "step", "evaluation", "checkpoint", "health", "finalization",
    )
    received = []

    def sink(event: TrainingEvent) -> None:
        received.append(event)

    emitter = EventEmitter(sink)
    for kind in sorted(TRAINING_EVENT_NAMES):  # arbitrary order stays monotonic
        emitter.emit(kind, {"n": 1})
    sequences = [event.sequence for event in received]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    assert all(event.kind in TRAINING_EVENT_NAMES for event in received)


# --- utility functions from hf_causal_lm ------------------------------------


def test_resolve_output_dir():
    assert resolve_output_dir("checkpoints/r1/run-1") == "checkpoints/r1/run-1"
    with pytest.raises(PipelineError):
        resolve_output_dir(None)
    with pytest.raises(PipelineError):
        resolve_output_dir("../outside")
    with pytest.raises(PipelineError):
        resolve_output_dir("C:/absolute/path")


def test_spec_has_no_hf_imports():
    import slm_train.training.spec as spec_module

    source = Path(spec_module.__file__).read_text(encoding="utf-8")
    assert "transformers" not in source
    assert "TrainingArguments" not in source


def test_training_result_carries_validation_label():
    result = TrainingResult(
        label="PASS", events=(), checkpoint=None, evaluations=(), diagnostic=None,
    )
    assert result.label in ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION")


def test_state_loading_uses_weights_only():
    from slm_train.training.hf_causal_lm import load_state_dict_safely
    import tempfile, os

    payload = {"layer.weight": torch.arange(4, dtype=torch.float32)}
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "state.pt")
        torch.save(payload, state_path)
        loaded = load_state_dict_safely(Path(state_path))
    assert torch.equal(loaded["layer.weight"], payload["layer.weight"])


# --- review fix I-1: positivity guards survive a model_construct bypass ---------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden", 0),
        ("heads", 0),
        ("kv_heads", 0),
        ("layers", 0),
        ("intermediate", 0),
        ("context", 0),
    ],
)
def test_non_positive_dimensions_are_rejected(field: str, value: int) -> None:
    """A zero dimension must fail closed naming the dimension, not build or ZeroDivision."""
    config = tiny_model_config()
    bypassed = ModelConfig.model_construct(**{**config.model_dump(), field: value})
    with pytest.raises(ModelCapabilityError, match=field):
        build_causal_lm(bypassed)


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden", -32), ("layers", -1), ("context", -16), ("intermediate", 0)],
)
def test_negative_or_zero_dimensions_name_the_offender(field: str, value: int) -> None:
    config = tiny_model_config()
    bypassed = ModelConfig.model_construct(**{**config.model_dump(), field: value})
    with pytest.raises(ModelCapabilityError) as excinfo:
        build_causal_lm(bypassed)
    assert field in str(excinfo.value)
    assert not isinstance(excinfo.value, ZeroDivisionError)


# --- review fix I-2: typed result and event contracts are enforced ---------------


def test_training_result_rejects_a_bogus_label() -> None:
    with pytest.raises(ValueError, match="invalid validation label"):
        TrainingResult(
            label="bogus", events=(), checkpoint=None, evaluations=(), diagnostic=None,
        )


@pytest.mark.parametrize("label", ["PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"])
def test_training_result_accepts_every_vocabulary_label(label: str) -> None:
    result = TrainingResult(
        label=label, events=(), checkpoint=None, evaluations=(), diagnostic=None,
    )
    assert result.label == label


def _event(name: str, sequence: int) -> TrainingEvent:
    return TrainingEvent(name=name, sequence=sequence, payload={})


@pytest.mark.parametrize(
    "sequences",
    [(9, 1), (1, 1), (2, 1, 3), (3, 2)],
)
def test_training_result_rejects_non_monotonic_event_sequences(sequences: tuple[int, ...]) -> None:
    events = tuple(_event("step", sequence) for sequence in sequences)
    with pytest.raises(ValueError, match="strictly increasing"):
        TrainingResult(
            label="PASS", events=events, checkpoint=None, evaluations=(), diagnostic=None,
        )


@pytest.mark.parametrize("bad_sequence", [0, -1])
def test_training_result_rejects_non_positive_event_sequences(bad_sequence: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        TrainingResult(
            label="PASS",
            events=(_event("step", bad_sequence),),
            checkpoint=None,
            evaluations=(),
            diagnostic=None,
        )


def test_training_result_accepts_a_strictly_increasing_sequence() -> None:
    events = (_event("prepare", 1), _event("step", 2), _event("checkpoint", 3))
    result = TrainingResult(
        label="PASS", events=events, checkpoint=None, evaluations=(), diagnostic=None,
    )
    assert [event.sequence for event in result.events] == [1, 2, 3]


def test_two_emitters_on_one_sink_do_not_duplicate_sequences() -> None:
    received: list[TrainingEvent] = []

    def sink(event: TrainingEvent) -> None:
        received.append(event)

    first = EventEmitter(sink)
    second = EventEmitter(sink)
    first.emit("step", {"i": 0})
    second.emit("step", {"i": 1})
    first.emit("health", {"i": 2})
    sequences = [event.sequence for event in received]
    assert len(sequences) == 3
    assert len(set(sequences)) == 3, "two emitters on one sink must not reuse a sequence"
    assert sequences == sorted(sequences)


# --- review fix minors ----------------------------------------------------------


def test_events_module_imports_no_torch_or_transformers() -> None:
    import subprocess
    import sys

    code = (
        "import sys;"
        "import slm_train.training.events as m;"
        "print(sorted(n for n in sys.modules if n.split('.')[0] in {'torch','transformers'}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "[]", completed.stdout


def test_manifest_hashes_change_with_the_model_config() -> None:
    def manifest_for(**overrides) -> ModelManifest:
        model_config = tiny_model_config(**overrides)
        return build_model_manifest(
            model_config=model_config,
            model=build_causal_lm(model_config),
            attn_implementation="sdpa",
            source_hash="0" * 64,
        )

    first = manifest_for()
    second = manifest_for(hidden=64)
    assert first.config_hash != second.config_hash
    assert first.semantic_manifest_hash() != second.semantic_manifest_hash()


def test_pickled_state_is_rejected_by_the_weights_only_loader(tmp_path: Path) -> None:
    import pickle

    from slm_train.training.hf_causal_lm import load_state_dict_safely

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo should-never-run",))

    state_path = tmp_path / "evil.pt"
    state_path.write_bytes(pickle.dumps({"payload": Exploit()}))
    with pytest.raises(Exception) as excinfo:
        load_state_dict_safely(state_path)
    assert "should-never-run" not in str(excinfo.value)

