"""Tests for the deterministic current-state renderer (Task 2a).

The renderer must produce byte-stable output from the same registry
inputs, must never inspect processes, and must never write anything in
check mode. All fixtures live in temporary directories so the live
repository state files are never touched by these tests.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.render_current_state import (
    RenderResult,
    collect_git_metadata,
    load_registries,
    render_state,
)

REGISTRY_LABELS = ("tasks", "experiments", "decisions", "environment", "jobs")


def write_registries(root: Path, *, jobs: list | None = None) -> None:
    """Create the five canonical registry files under a temporary root."""

    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "research").mkdir(parents=True, exist_ok=True)
    (root / "state" / "tasks.yaml").write_text(
        "schema_version: 1\n"
        "tasks:\n"
        "  - id: ui/task-u1\n"
        "    scope: ui\n"
        "    status: done\n"
        "    evidence: []\n"
        "  - id: ui/task-u1b\n"
        "    scope: ui\n"
        "    status: pending\n"
        "    evidence: []\n",
        encoding="utf-8",
    )
    (root / "research" / "experiments.yaml").write_text(
        "schema_version: 1\n"
        "experiments:\n"
        "  - id: ui/E-001\n"
        "    scope: ui\n"
        "    status: closed\n"
        "    verdict: accepted\n"
        "    evidence: []\n",
        encoding="utf-8",
    )
    (root / "state" / "decisions.yaml").write_text(
        "schema_version: 1\n"
        "decisions:\n"
        "  - id: meta/d-001\n"
        "    decision: keep fp16 on sm_75\n"
        "    status: accepted\n",
        encoding="utf-8",
    )
    (root / "state" / "environment.json").write_text(
        json.dumps({"python": "3.12.9", "gpu": "Turing sm_75"}), encoding="utf-8"
    )
    if jobs is None:
        jobs = [{"id": "diacritizer/train-e63", "status": "done"}]
    (root / "state" / "jobs.json").write_text(
        json.dumps({"schema_version": 1, "jobs": jobs}), encoding="utf-8"
    )


def tree(root: Path) -> set[str]:
    """Return every file path under root as a /-joined relative path."""

    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
    }


class RendererTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "fixture_root"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_render_is_deterministic_for_same_inputs(self):
        write_registries(self.root)
        first = render_state(fixture_root=self.root)
        second = render_state(fixture_root=self.root)
        self.assertIsInstance(first, RenderResult)
        self.assertEqual(first, second)
        self.assertEqual(first.text, second.text)
        self.assertTrue(first.ok)

    def test_conflicting_job_state_renders_unknown(self):
        write_registries(
            self.root,
            jobs=[
                {"id": "laya/train-e75", "status": "active"},
                {"id": "diacritizer/train-e63", "status": "active"},
            ],
        )
        result = render_state(fixture_root=self.root)
        self.assertIn("UNKNOWN", result.text)
        payload = json.loads((self.root / "state" / "current.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(payload["active_job"], "UNKNOWN")

    def test_check_mode_does_not_write(self):
        write_registries(self.root)
        (self.root / "CURRENT_STATE.md").write_bytes(b"stale snapshot\n")
        before = (self.root / "CURRENT_STATE.md").read_bytes()
        result = render_state(fixture_root=self.root, mode="check")
        after = (self.root / "CURRENT_STATE.md").read_bytes()
        self.assertEqual(before, after)
        self.assertFalse(result.ok)

    def test_check_mode_creates_no_new_files(self):
        write_registries(self.root)
        (self.root / "CURRENT_STATE.md").write_bytes(b"stale\n")
        before = tree(self.root)
        render_state(fixture_root=self.root, mode="check")
        self.assertEqual(tree(self.root), before)
        self.assertFalse((self.root / "state" / "current.json").exists())

    def test_check_passes_after_write(self):
        write_registries(self.root)
        rendered = render_state(fixture_root=self.root, mode="write")
        self.assertTrue(rendered.ok)
        verified = render_state(fixture_root=self.root, mode="check")
        self.assertTrue(verified.ok)
        self.assertEqual(rendered.text, verified.text)

    def test_write_touches_only_the_two_output_paths(self):
        write_registries(self.root)
        before = tree(self.root)
        render_state(fixture_root=self.root, mode="write")
        created = tree(self.root) - before
        self.assertEqual(created, {"CURRENT_STATE.md", "state/current.json"})

    def test_missing_registries_render_deterministically(self):
        first = render_state(fixture_root=self.root)
        second = render_state(fixture_root=self.root)
        self.assertEqual(first, second)
        self.assertIn("UNKNOWN", first.text)

    def test_identical_fixtures_in_different_dirs_render_equal(self):
        write_registries(self.root)
        other_root = Path(self._tmp.name) / "other_root"
        other_root.mkdir()
        write_registries(other_root)
        self.assertEqual(
            render_state(fixture_root=self.root).text,
            render_state(fixture_root=other_root).text,
        )

    def test_single_active_job_is_reported_by_id(self):
        write_registries(
            self.root, jobs=[{"id": "laya/train-e75", "status": "active"}]
        )
        result = render_state(fixture_root=self.root)
        self.assertIn("laya/train-e75", result.text)

    def test_load_registries_empty_root_is_deterministic(self):
        first = load_registries(self.root)
        second = load_registries(self.root)
        self.assertEqual(first, second)
        for label in REGISTRY_LABELS:
            self.assertIn(label, first)
            self.assertIsNone(first[label])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            render_state(fixture_root=self.root, mode="deploy")

    def test_render_result_is_frozen(self):
        result = RenderResult(text="x", ok=True)
        with self.assertRaises(Exception):
            result.text = "y"


class GitMetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_non_repo_directory_reports_unknown(self):
        meta = collect_git_metadata(self.root)
        self.assertEqual(meta["branch"], "UNKNOWN")
        self.assertEqual(meta["commit"], "UNKNOWN")
        self.assertEqual(meta["dirty_count"], "UNKNOWN")
        self.assertFalse((self.root / ".git").exists())

    def test_repo_directory_reports_facts(self):
        if shutil.which("git") is None:  # pragma: no cover - environment guard
            self.skipTest("git not available")
        env_repo = self.root / "repo"
        env_repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                cwd=env_repo,
                check=True,
                capture_output=True,
            )

        git("init")
        (env_repo / "tracked.txt").write_text("x\n", encoding="utf-8")
        git("add", "tracked.txt")
        git("commit", "-m", "init")
        meta = collect_git_metadata(env_repo)
        self.assertNotEqual(meta["branch"], "UNKNOWN")
        self.assertRegex(meta["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(meta["dirty_count"], 0)
        (env_repo / "untracked.txt").write_text("y\n", encoding="utf-8")
        self.assertEqual(collect_git_metadata(env_repo)["dirty_count"], 1)

    def test_write_then_check_ignores_generated_outputs(self):
        if shutil.which("git") is None:  # pragma: no cover - environment guard
            self.skipTest("git not available")
        env_repo = self.root / "snapshot_repo"
        env_repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                cwd=env_repo,
                check=True,
                capture_output=True,
            )

        git("init")
        git("commit", "--allow-empty", "-m", "init")
        write_registries(env_repo)
        before = collect_git_metadata(env_repo)["dirty_count"]
        rendered = render_state(env_repo, mode="write")
        self.assertTrue(rendered.ok)
        self.assertEqual(collect_git_metadata(env_repo)["dirty_count"], before)
        self.assertTrue(render_state(env_repo, mode="check").ok)


if __name__ == "__main__":
    unittest.main()
