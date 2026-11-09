"""The V1 boundary, checked as a boundary and not as a promise.

Three independent checks, because each catches a different mistake:

1. No V2 source imports V1, by module name, at all.
2. No V2 source reaches V1 through a dynamic import or a plugin scan.
3. The frozen V1 inventory still hashes to exactly what it hashed to, so a test
   run cannot have written to V1 even by accident.

A claim that "V2 never touches V1" is only worth something if something verifies
it on every run, so these run in the ordinary CPU suite.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
V2_PACKAGE = ROOT / "pipeline_v2" / "slm_train"

#: Top-level names that would mean V2 is calling into the frozen V1 tree.
V1_MODULE_NAMES = frozenset({
    "laya",
    "src",
    "scripts",
    "train_v1",
    "legacy",
    "v1",
    "nano_slms",
})

#: Callables that discover or execute code at runtime.
DYNAMIC_CALLS = frozenset({
    "importlib.import_module",
    "__import__",
    "pkgutil.iter_modules",
    "walk_packages",
})


def _v2_sources():
    return sorted(V2_PACKAGE.rglob("*.py"))


def test_the_v2_package_is_present():
    assert _v2_sources(), "no V2 sources found; the boundary scan would pass vacuously"


def test_no_v2_source_imports_a_v1_module():
    offenders = []
    for path in _v2_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in V1_MODULE_NAMES:
                    offenders.append(path.relative_to(ROOT).as_posix() + " -> " + name)
    assert not offenders, "V2 must not import V1: " + "; ".join(offenders)


def test_no_v2_source_reaches_v1_through_a_dynamic_import():
    offenders = []
    for path in _v2_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id == "__import__":
                offenders.append(path.relative_to(ROOT).as_posix() + " -> __import__")
            if isinstance(function, ast.Attribute):
                rendered = ast.unparse(function)
                if rendered in DYNAMIC_CALLS:
                    offenders.append(path.relative_to(ROOT).as_posix() + " -> " + rendered)
    assert not offenders, "V2 must not discover or load V1 at runtime: " + "; ".join(offenders)


def test_no_v2_source_uses_entry_point_plugin_discovery():
    offenders = []
    for path in _v2_sources():
        text = path.read_text(encoding="utf-8")
        for forbidden in ("entry_points", "importlib.metadata", "pkg_resources"):
            if forbidden in text:
                offenders.append(path.relative_to(ROOT).as_posix() + " -> " + forbidden)
    assert not offenders, "no runtime plugin discovery is allowed: " + "; ".join(offenders)


def test_frozen_v1_inventory_is_unchanged():
    inventory = json.loads(
        (ROOT / "pipeline_v2/tests/fixtures/v1_inventory.json").read_text(encoding="utf-8")
    )
    for row in inventory["files"]:
        payload = (ROOT / row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"], row["path"]
        assert len(payload) == row["bytes"], row["path"]


def test_the_inventory_fixture_is_not_empty():
    """An empty inventory would make the hash test pass for the wrong reason."""
    inventory = json.loads(
        (ROOT / "pipeline_v2/tests/fixtures/v1_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["files"], "the V1 inventory lists no files"


def test_migration_is_not_adoption_and_says_so():
    readme = (ROOT / "pipeline_v2/migrations/README.md").read_text(encoding="utf-8")
    assert "not adoption" in readme.lower()
    assert "no v1 file" in readme.lower()


@pytest.mark.parametrize("command", [
    "init", "validate", "plan", "doctor", "preflight", "run",
    "status", "explain", "diff", "export", "migrate-v1",
])
def test_every_command_in_the_plan_is_reachable(command):
    from slm_train.cli import build_parser

    actions = build_parser()._subparsers._group_actions[0].choices
    assert command in actions, "no such command: " + command
