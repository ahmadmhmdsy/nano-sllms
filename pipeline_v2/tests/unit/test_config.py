import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from slm_train.config.loader import ConfigError, load_config
from slm_train.config.canonical import semantic_hash
from slm_train.config.overrides import apply_safe_overrides
from slm_train.config.models import (
    ConfigDocument,
    DataConfig,
    EvaluatorSuiteConfig,
    FeaturesConfig,
    FreshnessPolicy,
    ModelConfig,
    PromotionReference,
    PackingConfig,
    RunConfig,
    RuntimeConfig,
    SourceDeclaration,
    StrictModel,
    TokenizerConfig,
    TrainingConfig,
    validate_materialized,
)
from slm_train.config.paths import PathPolicy

ROOT = Path(__file__).resolve().parents[3]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, "schema_version: 2\nschema_version: 2\n"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, "schema_version: 2\nunknown: true\n"))


def test_r2_capability_is_rejected(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    text = base.read_text(encoding="utf-8").replace(
        "capabilities: [pretrain]",
        "capabilities: [pretrain, sft]",
    )
    with pytest.raises(ConfigError, match="R1"):
        load_config(write(tmp_path, text))


def test_base_template_is_not_runnable_until_materialized() -> None:
    with pytest.raises(ValueError, match="materialized"):
        validate_materialized(load_config(ROOT / "pipeline_v2/configs/base/r1.yaml"))


def test_only_documented_cli_overrides_are_accepted() -> None:
    config = load_config(ROOT / "pipeline_v2/configs/base/r1.yaml")
    changed = apply_safe_overrides(config, {"run.seed": 43})
    assert changed.run.seed == 43
    with pytest.raises(ValueError, match="unsafe"):
        apply_safe_overrides(config, {"train.batch_size": 2})


def test_comments_do_not_change_semantic_hash(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    commented = tmp_path / "base-with-comment.yaml"
    commented.write_text("# formatting-only comment\n" + base.read_text(encoding="utf-8"), encoding="utf-8")
    assert semantic_hash(load_config(base).model_dump()) == semantic_hash(load_config(commented).model_dump())


def test_inheritance_cycle_and_ambiguous_list_merge_are_rejected(tmp_path: Path) -> None:
    from slm_train.config.inheritance import resolve_inheritance
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: [b.yaml]\n", encoding="utf-8")
    b.write_text("extends: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cycle"):
        resolve_inheritance(a, root=tmp_path)
    a.write_text("extends: [b.yaml]\ncapabilities: [pretrain]\n", encoding="utf-8")
    b.write_text("capabilities: [sft]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="merge"):
        resolve_inheritance(a, root=tmp_path)


def test_cross_field_context_mismatch_is_rejected(tmp_path: Path) -> None:
    """A tokenizer context that disagrees with the model context is rejected.

    The test bumps whatever the first declared context is rather than matching a
    literal, so it keeps testing the cross-field rule after the shipped base
    profile is retuned instead of quietly passing on a no-op replacement.
    """
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    source = base.read_text(encoding="utf-8")
    match = re.search(r"^(\s*context: )(\d+)$", source, flags=re.MULTILINE)
    assert match is not None, "the base profile declares no tokenizer context"
    bumped = int(match.group(2)) + 1
    text = source[: match.start()] + match.group(1) + str(bumped) + source[match.end() :]
    with pytest.raises(ConfigError, match="context"):
        load_config(write(tmp_path, text))


def test_nested_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, "schema_version: 2\nmodel: {not_a_model_field: 1}\n"))


def test_missing_required_nested_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Field required"):
        load_config(write(tmp_path, "schema_version: 2\nmodel: {hidden: 32}\n"))


def test_s_and_promotion_relationship_is_explicit(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    text = base.read_text(encoding="utf-8").replace(
        "  promotion: null",
        "  promotion: {predecessor_stage: S, predecessor_run_id: s-run, checkpoint: checkpoints/s, data_manifest: manifests/data.json, evaluator_manifest: manifests/evaluator.json, gate_report: reports/s-gate.json}",
    )
    with pytest.raises(ConfigError, match="S cannot reference"):
        load_config(write(tmp_path, text))


def test_safe_overrides_return_a_new_frozen_document() -> None:
    config = load_config(ROOT / "pipeline_v2/configs/base/r1.yaml")
    changed = apply_safe_overrides(config, {"run.seed": 43})

    assert config.run.seed == 42
    assert changed.run.seed == 43
    assert "apply_overrides_record" not in changed.__dict__
    with pytest.raises(ValidationError):
        changed.run.seed = 44


@pytest.mark.parametrize(
    "model_type",
    [
        RunConfig,
        FreshnessPolicy,
        SourceDeclaration,
        DataConfig,
        TokenizerConfig,
        PackingConfig,
        ModelConfig,
        TrainingConfig,
        EvaluatorSuiteConfig,
        RuntimeConfig,
        PromotionReference,
        FeaturesConfig,
        ConfigDocument,
    ],
)
def test_configuration_models_are_strict_and_frozen(model_type: type[StrictModel]) -> None:
    assert model_type.model_config["extra"] == "forbid"
    assert model_type.model_config["strict"] is True
    assert model_type.model_config["frozen"] is True


def test_committed_schema_constrains_capabilities_to_r1() -> None:
    schema_path = ROOT / "pipeline_v2/configs/schemas/config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    capabilities = schema["properties"]["capabilities"]

    assert capabilities["items"]["const"] == "pretrain"
    assert capabilities["minItems"] == 1
    assert capabilities["maxItems"] == 1
    assert "default" not in capabilities


def test_generated_schema_documents_runtime_cross_field_validation_boundary() -> None:
    schema = ConfigDocument.model_json_schema()

    assert schema["$comment"] == (
        "Pydantic model_json_schema cannot express model-level cross-field validators "
        "(context, precision, evaluator, and promotion relationships). "
        "Runtime Pydantic validation is authoritative."
    )


def test_r1_rejects_bf16_and_flash_attention(tmp_path: Path) -> None:
    base = (ROOT / "pipeline_v2/configs/base/r1.yaml").read_text(encoding="utf-8")
    cases = (
        (base.replace("precision: fp16", "precision: bf16"), "R1"),
        (base.replace("attention: sdpa", "attention: flash_attention_2"), "R1"),
    )
    for text, message in cases:
        with pytest.raises(ConfigError, match=message):
            load_config(write(tmp_path, text))


def test_inheritance_rejects_paths_outside_declared_root(tmp_path: Path) -> None:
    from slm_train.config.inheritance import resolve_inheritance

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("schema_version: 2\n", encoding="utf-8")
    child = root / "child.yaml"
    child.write_text("extends: [../outside.yaml]\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="root"):
        resolve_inheritance(child, root=root)


def test_path_policy_rejects_absolute_traversal_and_symlink_paths(tmp_path: Path) -> None:
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        policy.resolve(str(tmp_path / "outside.txt"))
    with pytest.raises(ValueError, match="traversal"):
        policy.resolve("../outside.txt")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        policy.resolve("link")
    with pytest.raises(ValueError, match="symlink"):
        policy.relative(link)


def test_materialized_validation_rejects_empty_sources_and_thresholds() -> None:
    config = load_config(ROOT / "pipeline_v2/configs/base/r1.yaml")
    source = SourceDeclaration(
        source_id="train",
        kind="local",
        path_or_repo="data/source",
        revision=None,
        file_set=("train.jsonl",),
        role="train",
        license="test",
        expected_rows=1,
        expected_tokens=1,
        freshness_policy=FreshnessPolicy(max_age="1d", supersession="manual"),
    )
    data = config.data.model_copy(
        update={
            "dataset_id": "dataset",
            "sources": (source,),
            "regeneration_command": ("./run.py",),
        }
    )
    tokenizer = config.tokenizer.model_copy(
        update={"source": "tokenizer", "revision_or_files": "tokenizer.json"}
    )
    materialized = config.model_copy(
        update={
            "run": config.run.model_copy(update={"name": "run"}),
            "data": data,
            "tokenizer": tokenizer,
        }
    )

    validate_materialized(materialized)
    with pytest.raises(ValueError, match="source"):
        validate_materialized(materialized.model_copy(update={"data": data.model_copy(update={"sources": ()})}))
    with pytest.raises(ValueError, match="threshold"):
        validate_materialized(
            materialized.model_copy(
                update={"evaluation": materialized.evaluation.model_copy(update={"thresholds": {}})}
            )
        )
