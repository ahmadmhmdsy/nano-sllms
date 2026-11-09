"""Security, migration, and the command surface. The refusals are the subject."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "pipeline_v2")

from slm_train.adapters.legacy_v1 import MigrationError, migrate_v1_config
from slm_train.security import (
    ExportError,
    ExportRequest,
    ModelSecurityError,
    PathSecurityError,
    export_allowlist,
    is_contained,
    redact,
    redact_mapping,
    require_contained,
    require_no_dynamic_type,
    require_regular_file,
    require_safe_weight_suffix,
)


# --- path containment -----------------------------------------------------


def test_a_child_path_is_contained(tmp_path):
    child = tmp_path / "runs" / "a" / "file.json"
    child.parent.mkdir(parents=True)
    child.write_text("{}", encoding="utf-8")
    assert is_contained(tmp_path, child) is True
    assert require_contained(tmp_path, child) == child.resolve()


def test_a_traversal_is_refused(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    assert is_contained(tmp_path, tmp_path / ".." / "outside.txt") is False
    with pytest.raises(PathSecurityError, match="escapes"):
        require_contained(tmp_path, tmp_path / ".." / "outside.txt")


def test_a_string_prefix_that_is_not_a_path_prefix_is_refused(tmp_path):
    """The classic bug: "runs-evil".startswith("runs") is True."""
    root = tmp_path / "runs"
    root.mkdir()
    sibling = tmp_path / "runs-evil" / "file.txt"
    sibling.parent.mkdir()
    sibling.write_text("x", encoding="utf-8")
    assert is_contained(root, sibling) is False


def test_a_symlink_escape_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "secret.txt"
    target.write_text("x", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    assert is_contained(root, link) is False
    with pytest.raises(PathSecurityError, match="symlink"):
        require_regular_file(root, link)


def test_a_root_is_not_contained_in_itself(tmp_path):
    assert is_contained(tmp_path, tmp_path) is False
    with pytest.raises(PathSecurityError):
        require_contained(tmp_path, tmp_path)


def test_a_directory_is_not_a_regular_file(tmp_path):
    child = tmp_path / "sub"
    child.mkdir()
    with pytest.raises(PathSecurityError, match="not a regular file"):
        require_regular_file(tmp_path, child)


# --- model loading --------------------------------------------------------


def test_only_safetensors_is_accepted():
    assert require_safe_weight_suffix(Path("m.safetensors")).suffix == ".safetensors"


@pytest.mark.parametrize("name", ["m.pt", "m.pth", "m.bin", "m.ckpt", "m", "m.SAFE"])
def test_another_weight_format_is_refused_before_it_is_opened(name):
    if name == "m.SAFE":
        pytest.skip("case is handled by the lower() comparison")
    with pytest.raises(ModelSecurityError, match="safetensors"):
        require_safe_weight_suffix(Path(name))


def test_an_uppercase_suffix_is_accepted_because_the_check_is_case_folded():
    assert require_safe_weight_suffix(Path("m.SAFETENSORS"))


@pytest.mark.parametrize("key", [
    "auto_map", "architectures", "quantization_config", "module",
    "custom_pipelines",
])
def test_a_config_that_names_a_type_is_refused(key):
    with pytest.raises(ModelSecurityError, match="execute"):
        require_no_dynamic_type({"model_type": "llama", key: {"x": "y"}})


def test_a_nested_auto_map_is_refused_too():
    """A top-level-only scan is bypassed by nesting one level down."""
    with pytest.raises(ModelSecurityError, match="nested.auto_map"):
        require_no_dynamic_type({"nested": {"auto_map": {"X": "os.system"}}})


def test_an_unsafe_key_deep_inside_a_list_is_refused():
    with pytest.raises(ModelSecurityError, match="architectures"):
        require_no_dynamic_type({"a": [{"b": [{"architectures": ["X"]}]}]})


def test_the_refusal_names_where_the_offending_key_was():
    with pytest.raises(ModelSecurityError) as error:
        require_no_dynamic_type({"outer": {"inner": {"auto_map": 1}}})
    assert "outer.inner.auto_map" in str(error.value)


def test_a_plain_config_is_accepted():
    assert require_no_dynamic_type({"model_type": "llama", "hidden": 8})


def test_a_non_mapping_config_is_refused():
    with pytest.raises(ModelSecurityError, match="mapping"):
        require_no_dynamic_type(["architectures"])


# --- secrets --------------------------------------------------------------


def test_a_secret_value_is_redacted_wherever_it_appears():
    environ = {"HF_TOKEN": "hf_supersecretvalue"}
    assert "supersecret" not in redact("token=hf_supersecretvalue", environ)
    assert "[REDACTED]" in redact("token=hf_supersecretvalue", environ)


def test_a_secret_pasted_into_a_notes_field_is_still_redacted():
    environ = {"WANDB_API_KEY": "abcd1234efgh5678"}
    document = {"notes": "the key is abcd1234efgh5678 keep it safe"}
    cleaned = redact_mapping(document, environ)
    assert "abcd1234efgh5678" not in json.dumps(cleaned)


def test_a_sensitive_key_is_masked_even_when_its_value_is_innocent():
    document = {"api_key": "not-a-real-secret", "steps": 10}
    cleaned = redact_mapping(document, {})
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["steps"] == 10


def test_redaction_does_not_mutate_the_live_document():
    document = {"api_key": "value"}
    redact_mapping(document, {})
    assert document["api_key"] == "value"


def test_nested_structures_are_redacted():
    environ = {"HF_TOKEN": "hf_abcdefghij"}
    document = {"a": [{"b": "see hf_abcdefghij"}], "c": ("hf_abcdefghij",)}
    rendered = json.dumps(redact_mapping(document, environ), default=str)
    assert "hf_abcdefghij" not in rendered


def test_a_secret_hidden_in_a_key_name_is_redacted():
    """A dict of tokens keyed by token is a real shape."""
    environ = {"HF_TOKEN": "hf_SECRETVALUE123"}
    cleaned = redact_mapping({"note_hf_SECRETVALUE123": "x"}, environ)
    assert "hf_SECRETVALUE123" not in json.dumps(cleaned)


def test_a_very_short_value_is_not_treated_as_a_secret():
    """A two-character token would redact half the document."""
    assert redact("abc", {"HF_TOKEN": "ab"}) == "abc"


# --- export ---------------------------------------------------------------


def _run_tree(tmp_path):
    (tmp_path / "final").mkdir(parents=True)
    (tmp_path / "final" / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "final" / "tokenizer.json").write_bytes(b"tokens")
    (tmp_path / "COMPLETE.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_an_export_copies_exactly_the_allowlist(tmp_path):
    run = _run_tree(tmp_path)
    copied = export_allowlist(
        ExportRequest(run, tmp_path / "out", ["final/model.safetensors"])
    )
    assert copied == ["final/model.safetensors"]
    assert (tmp_path / "out/final/model.safetensors").read_bytes() == b"weights"
    assert not (tmp_path / "out/COMPLETE.json").exists()


def test_an_export_without_an_allowlist_is_refused(tmp_path):
    with pytest.raises(ExportError, match="allowlist"):
        ExportRequest(tmp_path, tmp_path / "out", [])
    with pytest.raises(ExportError, match="allowlist"):
        ExportRequest(tmp_path, tmp_path / "out", None)


def test_an_allowlist_entry_that_escapes_the_run_is_refused(tmp_path):
    run = _run_tree(tmp_path)
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ExportError, match="refused"):
        export_allowlist(ExportRequest(run, tmp_path / "out", ["../secret.txt"]))


def test_an_allowlist_entry_that_does_not_exist_is_refused(tmp_path):
    run = _run_tree(tmp_path)
    with pytest.raises(ExportError, match="refused"):
        export_allowlist(ExportRequest(run, tmp_path / "out", ["final/absent.bin"]))


def test_an_allowlist_naming_a_directory_is_refused(tmp_path):
    run = _run_tree(tmp_path)
    with pytest.raises(ExportError, match="refused"):
        export_allowlist(ExportRequest(run, tmp_path / "out", ["final"]))


# --- V1 migration ---------------------------------------------------------


def test_migration_preserves_the_input_and_writes_a_new_file(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\nseed: 7\n", encoding="utf-8")
    before = source.read_bytes()
    result = migrate_v1_config(source, tmp_path / "v2.yaml")
    assert source.read_bytes() == before
    assert result.source_sha256_before == result.source_sha256_after
    import yaml as _yaml

    written = _yaml.safe_load(result.target.read_text(encoding="utf-8"))
    assert written["run"]["name"] == "pilot"
    assert written["schema_version"] == 2


def test_migration_never_launches_anything(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\n", encoding="utf-8")
    result = migrate_v1_config(source, tmp_path / "v2.yaml")
    written = result.target.read_text(encoding="utf-8")
    assert "not an adoption" in written
    assert "cuda" not in written
    assert "device" not in written


def test_migration_refuses_to_overwrite_an_existing_output(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\n", encoding="utf-8")
    target = tmp_path / "v2.yaml"
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(MigrationError, match="overwrite"):
        migrate_v1_config(source, target)
    assert target.read_text(encoding="utf-8") == "existing"


def test_migration_refuses_to_write_over_its_own_input(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="may not be"):
        migrate_v1_config(source, source)


def test_migration_refuses_a_missing_input(tmp_path):
    with pytest.raises(MigrationError, match="does not exist"):
        migrate_v1_config(tmp_path / "absent.yaml", tmp_path / "v2.yaml")


def test_a_checkpoint_reference_is_never_carried_across(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\npretrained_checkpoint: /some/model.pt\n", encoding="utf-8")
    result = migrate_v1_config(source, tmp_path / "v2.yaml")
    assert "pretrained_checkpoint" in result.notes
    assert "model.base_model" not in result.target.read_text(encoding="utf-8")


def test_an_unmapped_key_is_reported_rather_than_guessed(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\nsome_unknown_v1_knob: 3\n", encoding="utf-8")
    result = migrate_v1_config(source, tmp_path / "v2.yaml")
    assert "some_unknown_v1_knob" in result.notes


# --- commands -------------------------------------------------------------


def test_migrate_v1_command_preserves_the_input_and_says_it_will_not_launch(tmp_path, capsys):
    from slm_train.cli import main

    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\n", encoding="utf-8")
    code = main([
        "migrate-v1", "--input", str(source), "--output", str(tmp_path / "v2.yaml"),
    ])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert source.read_text(encoding="utf-8") == "name: pilot\n"
    assert output["source_unchanged"] is True
    assert "will not launch" in output["note"]


def test_export_command_refuses_to_run_without_an_allowlist(tmp_path, capsys):
    from slm_train.cli import main

    code = main(["export", "--run", str(tmp_path), "--out", str(tmp_path / "o")])
    output = json.loads(capsys.readouterr().out)
    assert code != 0
    assert output["status"] == "FAIL"
    assert "allowlist" in output["reason"]


def test_status_on_a_run_with_no_published_status_is_blocked(tmp_path, capsys):
    from slm_train.cli import main

    code = main(["status", "--run", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert code != 0
    assert output["status"] == "BLOCKED"


def test_init_refuses_to_overwrite_an_existing_file(tmp_path, capsys):
    from slm_train.cli import main

    target = tmp_path / "new.yaml"
    target.write_text("keep me", encoding="utf-8")
    code = main(["init", "--out", str(target)])
    output = json.loads(capsys.readouterr().out)
    assert code != 0
    assert output["status"] == "FAIL"
    assert target.read_text(encoding="utf-8") == "keep me"


def test_doctor_never_touches_a_device(capsys):
    from slm_train.cli import main

    assert main(["doctor"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["gpu_touched"] is False


def test_preflight_without_gpu_never_probes_a_device(capsys):
    from slm_train.cli import main

    main(["preflight"])
    output = json.loads(capsys.readouterr().out)
    assert "gpu" not in output or output.get("gpu_probed") is not True


def test_every_experiment_template_validates(capsys):
    from slm_train.cli import main

    for name in ("smoke", "pilot", "target"):
        path = Path("pipeline_v2/configs/experiments") / (name + ".yaml")
        assert main(["validate", "--config", str(path)]) == 0, name
        assert json.loads(capsys.readouterr().out)["status"] == "PASS"
