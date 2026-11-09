"""Task 7 contract tests: R1 registries, trusted interfaces, extension boundary."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from slm_train.adapters import registry as adapter_registry
from slm_train.adapters.registry import entries, get as get_adapter
from slm_train.config.models import R1_CAPABILITIES, R2_CAPABILITIES
from slm_train.contracts import interfaces
from slm_train.contracts.base import ValidationLabel
from slm_train.features import registry as feature_registry
from slm_train.features.registry import validate_capabilities

EXPECTED_PROTOCOLS = (
    "SourceReader",
    "RowTransformer",
    "Deduplicator",
    "Splitter",
    "TokenizerAdapter",
    "Packer",
    "DatasetValidator",
    "ModelFactory",
    "FeatureModule",
    "CheckpointAdapter",
    "Evaluator",
    "MetricSuite",
    "Gate",
    "SelectionPolicy",
    "ArtifactPublisher",
    "ResourceProbe",
    "HealthMonitor",
)

FORBIDDEN_CAPABILITIES = (
    "acquire_lease",
    "rewrite_immutable_manifest",
    "mark_run_successful",
    "publish_outside_allowlist",
)


# --- brief tests ------------------------------------------------------------


def test_r1_rejects_each_r2_capability():
    for capability in ("sft", "lora", "kd", "gdn", "mounting"):
        with pytest.raises(ValueError, match="R1"):
            validate_capabilities("r1", ["pretrain", capability])


def test_registry_has_no_dynamic_discovery():
    import slm_train.adapters.registry as registry

    assert not hasattr(registry, "discover")
    assert registry.entries()


# --- adapter registry --------------------------------------------------------


def test_r1_registry_is_explicit_and_nonempty():
    values = entries()
    assert values
    assert all(value.release == "r1" for value in values)
    assert all(value.maturity in {"experimental", "validated", "retired"} for value in values)


def test_r1_rejects_all_release_two_capabilities():
    for capability in ("sft", "lora", "kd", "gdn", "mounting"):
        try:
            validate_capabilities("r1", ["pretrain", capability])
        except ValueError as error:
            assert "R1" in str(error)
        else:
            raise AssertionError(capability)


def test_every_r2_capability_name_is_rejected():
    for capability in sorted(R2_CAPABILITIES):
        with pytest.raises(ValueError):
            validate_capabilities("r1", ["pretrain", capability])


def test_capability_gate_rejects_bad_release_and_malformed_sets():
    with pytest.raises(ValueError, match="r1"):
        validate_capabilities("r2", ["pretrain"])
    with pytest.raises(ValueError, match="r1"):
        validate_capabilities("", ["pretrain"])
    with pytest.raises(ValueError):
        validate_capabilities("r1", [])
    with pytest.raises(ValueError):
        validate_capabilities("r1", ["pretrain", "pretrain"])
    with pytest.raises(ValueError):
        validate_capabilities("r1", ["supervised"])
    assert R1_CAPABILITIES == {"pretrain"}
    assert not (R1_CAPABILITIES & R2_CAPABILITIES)


def test_r1_only_capability_is_accepted():
    validate_capabilities("r1", ["pretrain"])
    validate_capabilities("r1", tuple(R1_CAPABILITIES))


def test_adapter_entries_are_well_formed():
    for entry in entries():
        for field in ("name", "version", "schema", "stage", "test_contract"):
            assert getattr(entry, field).strip(), entry.name
        assert entry.maturity in {"experimental", "validated", "retired"}
        assert entry.release == "r1"
        assert "r1" in entry.compatibility


def test_adapter_entry_rejects_empty_and_out_of_range_fields():
    base = {
        "name": "x",
        "version": "1",
        "schema": "S",
        "stage": "prepare",
        "maturity": "validated",
        "compatibility": ("r1",),
        "release": "r1",
        "test_contract": "unit/test_x.py",
    }
    for field, value in (
        ("name", ""),
        ("version", ""),
        ("schema", ""),
        ("stage", ""),
        ("test_contract", ""),
        ("maturity", "production"),
        ("release", "r2"),
        ("compatibility", ()),
    ):
        with pytest.raises(ValueError):
            adapter_registry.RegistryEntry(**{**base, field: value})


def test_unknown_adapter_name_fails_closed():
    with pytest.raises(ValueError):
        get_adapter("no_such_adapter")
    with pytest.raises(ValueError):
        get_adapter("")
    with pytest.raises(ValueError):
        get_adapter(None)


def test_known_adapter_name_resolves_to_its_own_entry():
    for entry in entries():
        assert get_adapter(entry.name) is entry


def test_unknown_feature_name_fails_closed():
    with pytest.raises(ValueError):
        feature_registry.get("no_such_feature")
    with pytest.raises(ValueError):
        feature_registry.get("")


def test_feature_registry_is_static_and_the_r2_boundary_is_empty():
    values = feature_registry.entries()
    assert values
    assert all(entry.release == "r1" for entry in values)
    assert all("pretrain" in entry.capabilities for entry in values)
    assert all("r2" not in entry.capabilities for entry in values)
    assert feature_registry.r2_boundary() == ()


# --- no dynamic discovery anywhere -------------------------------------------


@pytest.mark.parametrize("module", [adapter_registry, feature_registry])
def test_no_dynamic_registration_symbol_is_exported(module) -> None:
    for name in (
        "discover",
        "register",
        "load",
        "scan",
        "load_entry_points",
        "plugins",
        "iter_modules",
        "import_module",
    ):
        assert not hasattr(module, name), f"{module.__name__}.{name}"


def test_registry_modules_never_use_dynamic_import_mechanisms() -> None:
    for module in (adapter_registry, feature_registry):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in (
            "importlib",
            "pkgutil",
            "entry_points",
            "subprocess",
            "__import__",
            "eval(",
            "exec(",
        ):
            assert banned not in source, f"{module.__name__} uses {banned}"


# --- trusted interfaces ------------------------------------------------------


def test_every_expected_protocol_is_defined():
    for name in EXPECTED_PROTOCOLS:
        assert hasattr(interfaces, name), name
        assert isinstance(getattr(interfaces, name), type), name


@pytest.mark.parametrize("name", EXPECTED_PROTOCOLS)
def test_protocol_declares_input_output_label_and_allowlist(name: str) -> None:
    from slm_train.contracts.validation import require_repository_relative

    declaration = interfaces.protocol_declaration(name)
    assert declaration.input_models, name
    assert declaration.output_models, name
    assert declaration.failure_label in ValidationLabel.__args__, name
    assert declaration.artifact_allowlist, name
    for root in declaration.artifact_allowlist:
        # each allowlist entry is a writable root: repository-relative, and it
        # must admit a child while the root itself is not an artifact
        require_repository_relative(root)
        interfaces.require_allowed_artifact(name, root.rstrip("/") + "/artifact.bin")
        with pytest.raises(ValueError):
            interfaces.require_allowed_artifact(name, root.rstrip("/"))


@pytest.mark.parametrize("name", EXPECTED_PROTOCOLS)
def test_protocol_declares_exactly_the_r1_methods(name: str) -> None:
    protocol = getattr(interfaces, name)
    expected = interfaces.R1_METHOD_MAP[name]
    assert set(expected) <= set(protocol.__protocol_attrs__), name
    for method in expected:
        assert callable(getattr(protocol, method, None)), f"{name}.{method}"


def test_interface_modules_import_no_torch_or_transformers() -> None:
    import ast

    banned = {"torch", "transformers", "TrainingArguments"}
    for module in (interfaces, adapter_registry, feature_registry):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            for name in names:
                root = name.split(".")[0]
                assert root not in banned, f"{module.__name__} imports {name}"


# --- extension boundary ------------------------------------------------------


@pytest.mark.parametrize("capability", FORBIDDEN_CAPABILITIES)
def test_every_protocol_forbids_the_four_extension_capabilities(capability: str) -> None:
    for name in EXPECTED_PROTOCOLS:
        declaration = interfaces.protocol_declaration(name)
        assert capability in declaration.forbidden_capabilities, f"{name}/{capability}"


def test_an_extension_requesting_a_forbidden_capability_is_rejected() -> None:
    assert interfaces.request_capability("SourceReader", "read_rows") is True
    for capability in FORBIDDEN_CAPABILITIES:
        with pytest.raises(ValueError):
            interfaces.request_capability("SourceReader", capability)


def test_an_unknown_capability_fails_closed() -> None:
    with pytest.raises(ValueError):
        interfaces.request_capability("SourceReader", "teleport")
    with pytest.raises(ValueError):
        interfaces.request_capability("NoSuchProtocol", "read_rows")


@pytest.mark.parametrize(
    "path",
    ["/abs/path", "C:/win", "../escape", "..\\escape", "", "logs/", "C:relative"],
)
def test_artifact_allowlist_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("SourceReader", path)


@pytest.mark.parametrize(
    "path",
    ["data/v2/prepared", "data/v2/prepared/", "data/v2/prepared/./x", "data//v2//prepared//x"],
)
def test_artifact_allowlist_rejects_the_root_itself_and_dot_segments(path: str) -> None:
    """An allowlist root names a directory, not a writable artifact."""
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("SourceReader", path)


def test_artifact_allowlist_rejects_a_path_outside_the_allowlist() -> None:
    declaration = interfaces.protocol_declaration("SourceReader")
    allowed = declaration.artifact_allowlist[0].rstrip("/") + "/allowed.bin"
    interfaces.require_allowed_artifact("SourceReader", allowed)
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("SourceReader", "not/allowed/anywhere.bin")
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("NoSuchProtocol", allowed)


def test_allowed_artifact_must_be_a_prefix_of_a_declared_root() -> None:
    declaration = interfaces.protocol_declaration("SourceReader")
    root = declaration.artifact_allowlist[0].rstrip("/")
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("SourceReader", root + "-other/file.bin")
    with pytest.raises(ValueError):
        interfaces.require_allowed_artifact("SourceReader", root + "/../elsewhere.bin")


# --- typed models the protocols reference ------------------------------------


def test_typed_models_are_strict_and_frozen() -> None:
    for name in interfaces.TYPED_MODEL_NAMES:
        model = getattr(interfaces, name)
        assert dataclasses.is_dataclass(model) or hasattr(model, "model_config"), name
        config = getattr(model, "model_config", {})
        assert config.get("frozen") is True, name
        assert config.get("extra") == "forbid", name


def test_typed_models_are_all_importable() -> None:
    for name in interfaces.TYPED_MODEL_NAMES:
        assert hasattr(interfaces, name), name


def test_provenance_record_and_runtime_profile_carry_real_fields() -> None:
    provenance = interfaces.ProvenanceRecord.model_fields
    for field in ("dataset_id", "source_manifest_hash", "generation_id"):
        assert field in provenance, field
    runtime = interfaces.RuntimeProfile.model_fields
    for field in ("python_version", "os", "device_class", "device_name"):
        assert field in runtime, field


def test_selection_policy_protocol_and_config_are_distinguishable() -> None:
    assert interfaces.SelectionPolicy is not interfaces.SelectionPolicyConfig
    assert "select" in interfaces.SelectionPolicy.__protocol_attrs__
    assert interfaces.SelectionPolicyConfig.model_fields


def test_declared_allowlist_and_capabilities_are_immutable() -> None:
    declaration = interfaces.protocol_declaration("SelectionPolicy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.artifact_allowlist = ("artifacts/elsewhere/",)

