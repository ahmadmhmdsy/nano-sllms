import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import patch

import yaml

# Direct execution (python tests\agent_context\test_check_agent_context.py)
# must resolve scripts.check_agent_context; discovery with -t . gets the
# repository root on sys.path for free, direct runs do not.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.check_agent_context import (
    _FORMAT_DIAGNOSTIC as _FORMAT_DIAGNOSTIC_PATTERN,
    check_markdown_links,
    check_records,
    check_static_policy,
    lint_tree,
    main,
)


FIXTURES = Path(__file__).parent / "fixtures"

# Placeholder and bound expected from the linter's redacting diagnostic boundary.
REDACTED_DIAGNOSTIC_PLACEHOLDER = "<redacted-diagnostic>"
MAX_ISSUE_CHARS = 320


class ContextLinterTests(unittest.TestCase):
    def test_duplicate_task_ids_are_reported(self):
        issues = check_records(
            {"tasks": [{"id": "ui/u1"}, {"id": "ui/u1"}], "experiments": []},
            root=Path("."),
        )
        self.assertTrue(any("duplicate task id" in issue for issue in issues))

    def test_invalid_task_status_is_reported(self):
        issues = check_records(
            {"tasks": [{"id": "ui/u1", "status": "finished"}], "experiments": []},
            root=Path("."),
        )
        self.assertTrue(any("invalid task status" in issue for issue in issues))

    def test_duplicate_experiment_ids_are_reported(self):
        issues = check_records(
            {
                "tasks": [],
                "experiments": [
                    {"id": "laya/E-1"},
                    {"id": "laya/E-1"},
                ],
            },
            root=Path("."),
        )
        self.assertTrue(any("duplicate experiment id" in issue for issue in issues))

    def test_invalid_experiment_status_and_verdict_are_reported(self):
        issues = check_records(
            {
                "tasks": [],
                "experiments": [
                    {"id": "laya/E-1", "status": "finished", "verdict": "maybe"},
                ],
            },
            root=Path("."),
        )
        self.assertTrue(any("invalid experiment status" in issue for issue in issues))
        self.assertTrue(any("invalid experiment verdict" in issue for issue in issues))

    def test_malformed_table_row_is_reported(self):
        issues = check_records(
            {"tasks": [], "experiments": [], "tables": {"TASKS.md": "| a || b |"}},
            root=Path("."),
        )
        self.assertTrue(any("malformed table row" in issue for issue in issues))

    def test_missing_evidence_path_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            issues = check_records(
                {
                    "tasks": [
                        {
                            "id": "ui/u1",
                            "status": "done",
                            "evidence": ["missing/evidence.json"],
                        }
                    ],
                    "experiments": [],
                },
                root=root,
            )
        self.assertTrue(any("missing evidence path" in issue for issue in issues))
        self.assertTrue(any("missing/evidence.json" in issue for issue in issues))

    def test_evidence_diagnostics_redact_secrets_and_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = (
                ("missing/sk-" + ("A" * 24) + ".txt", "<redacted-secret-evidence>"),
                ("missing/evidence\x1f.txt", "<redacted-control-evidence>"),
            )
            for value, marker in cases:
                with self.subTest(marker=marker):
                    issues = check_records(
                        {
                            "tasks": [{
                                "id": "ui/u1",
                                "status": "done",
                                "evidence": [value],
                            }],
                            "experiments": [],
                        },
                        root=root,
                    )
                    rendered = "\n".join(issues)
                    self.assertTrue(any(marker in issue for issue in issues))
                    self.assertNotIn(value, rendered)
                    self.assertNotIn("Traceback", rendered)

    def test_missing_evidence_diagnostic_is_bounded_for_long_safe_value(self):
        value = "missing/" + ("x" * 250) + ".txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            issues = check_records(
                {
                    "tasks": [{
                        "id": "ui/u1",
                        "status": "done",
                        "evidence": [value],
                    }],
                    "experiments": [],
                },
                root=Path(temp_dir),
            )
        issue = next(issue for issue in issues if "missing evidence path" in issue)
        self.assertIn("<truncated-evidence>", issue)
        self.assertLessEqual(len(issue), 320)
        self.assertNotIn(value, issue)

    def test_evidence_paths_are_lexical_and_no_follow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "context-root"
            root.mkdir()
            regular = root / "evidence.txt"
            regular.write_text("ok\n", encoding="utf-8")
            directory = root / "runs" / "u1_full"
            directory.mkdir(parents=True)
            outside = workspace / "outside-secret.txt"
            outside.write_text("LEAKED_SECRET_VALUE\n", encoding="utf-8")
            link = root / "linked-evidence.txt"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            def issues_for(path):
                return check_records(
                    {
                        "tasks": [{
                            "id": "ui/u1",
                            "status": "done",
                            "evidence": [path],
                        }],
                        "experiments": [],
                    },
                    root=root,
                )

            self.assertFalse(any("evidence" in issue for issue in issues_for("evidence.txt")))
            self.assertFalse(any("evidence" in issue for issue in issues_for("runs/u1_full")))
            linked_issues = issues_for("linked-evidence.txt")
            self.assertTrue(any("unsafe evidence" in issue for issue in linked_issues))
            self.assertNotIn("LEAKED_SECRET_VALUE", "\n".join(linked_issues))

    def test_evidence_paths_reject_lexical_escapes_and_aliases(self):
        invalid = (
            " ../outside.txt",
            "../outside.txt",
            " docs/file.txt",
            "docs/file.txt ",
            "docs//file.txt",
            "./docs/file.txt",
            r"docs\file.txt",
            r"state\tasks.yaml",
            "https://example.invalid/file.txt",
            "C:/outside.txt",
            "C:foo",
            "//server/share/file.txt",
            "docs/file.txt:stream",
            "README.md. ",
            "docs/mixed\\separator/file.txt",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for value in invalid:
                with self.subTest(value=value):
                    issues = check_records(
                        {
                            "tasks": [{
                                "id": "ui/u1",
                                "status": "done",
                                "evidence": [value],
                            }],
                            "experiments": [],
                        },
                        root=root,
                    )
                    self.assertTrue(issues)

    def test_non_scalar_status_and_verdict_are_controlled_issues(self):
        issues = check_records(
            {
                "tasks": [{
                    "id": "ui/u1",
                    "status": ["bad"],
                    "evidence": [],
                }],
                "experiments": [{
                    "id": "ui/e1",
                    "status": "running",
                    "verdict": {"bad": True},
                }],
            },
            root=Path("."),
        )
        self.assertTrue(any("invalid task status" in issue for issue in issues))
        self.assertTrue(any("invalid experiment verdict" in issue for issue in issues))

    def test_valid_fixtures_pass(self):
        valid_root = FIXTURES / "valid_state"
        tasks = yaml.safe_load((valid_root / "tasks.yaml").read_text(encoding="utf-8"))
        experiments = yaml.safe_load(
            (valid_root / "experiments.yaml").read_text(encoding="utf-8")
        )
        issues = check_records(
            {"tasks": tasks["tasks"], "experiments": experiments["experiments"]},
            root=valid_root,
        )
        self.assertEqual(issues, [])

    def test_invalid_fixtures_report_expected_issues(self):
        invalid_root = FIXTURES / "invalid_state"
        tasks = yaml.safe_load((invalid_root / "tasks.yaml").read_text(encoding="utf-8"))
        experiments = yaml.safe_load(
            (invalid_root / "experiments.yaml").read_text(encoding="utf-8")
        )
        issues = check_records(
            {"tasks": tasks["tasks"], "experiments": experiments["experiments"]},
            root=invalid_root,
        )
        self.assertTrue(any("duplicate task id" in issue for issue in issues))
        self.assertTrue(any("invalid task status" in issue for issue in issues))
        self.assertTrue(any("missing evidence path" in issue for issue in issues))
        self.assertTrue(any("duplicate experiment id" in issue for issue in issues))
        self.assertTrue(any("invalid experiment status" in issue for issue in issues))
        self.assertTrue(any("invalid experiment verdict" in issue for issue in issues))

    def test_markdown_links_check_local_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Target\n", encoding="utf-8")
            source = docs / "source.md"
            source.write_text(
                "[good](./target.md)\n[bad](./missing.md)\n",
                encoding="utf-8",
            )
            issues = check_markdown_links(source, root)
        self.assertEqual(len(issues), 1)
        self.assertIn("broken markdown link", issues[0])

    def test_markdown_link_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            docs = root / "docs"
            docs.mkdir(parents=True)
            outside = root.parent / "outside.md"
            outside.write_text("outside\n", encoding="utf-8")
            source = docs / "source.md"
            source.write_text("[escape](../outside.md)\n", encoding="utf-8")
            issues = check_markdown_links(source, root)
        self.assertTrue(any("outside" in issue or "escape" in issue for issue in issues))
        self.assertNotIn("outside\n", "\n".join(issues))

    def test_markdown_link_rejects_single_backslash_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Target\n", encoding="utf-8")
            source = root / "source.md"
            source.write_text("[target](docs\\target.md)\n", encoding="utf-8")
            issues = check_markdown_links(source, root)
        self.assertTrue(any("unsafe" in issue for issue in issues))

    def test_markdown_link_nul_target_is_a_controlled_issue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.md"
            source.write_text("[bad](bad\x00name.md)\n", encoding="utf-8")
            issues = check_markdown_links(source, root)
        self.assertTrue(any("unsafe markdown link" in issue for issue in issues))
        self.assertNotIn("\x00", "\n".join(issues))
        self.assertNotIn("embedded null", "\n".join(issues).lower())

    def test_markdown_link_symlink_target_is_rejected_without_reading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "root"
            docs = root / "docs"
            docs.mkdir(parents=True)
            secret = workspace / "secret.md"
            secret.write_text("LEAKED_SECRET_VALUE\n", encoding="utf-8")
            link = docs / "linked.md"
            try:
                link.symlink_to(secret)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            source = docs / "source.md"
            source.write_text("[linked](linked.md)\n", encoding="utf-8")
            issues = check_markdown_links(source, root)
        self.assertTrue(any("unsafe" in issue for issue in issues))
        self.assertNotIn("LEAKED_SECRET_VALUE", "\n".join(issues))

    def test_static_policy_rejects_live_process_claim(self):
        issues = check_static_policy(
            {"AGENTS.md": "PID 123 is live as of now.\n"},
            root=Path("."),
        )
        self.assertTrue(issues)

    def test_static_policy_rejects_bare_python_and_pip(self):
        issues = check_static_policy(
            {"AGENTS.md": "python script.py\npip install package\n"},
            root=Path("."),
        )
        self.assertTrue(any("bare Python" in issue for issue in issues))
        self.assertTrue(any("bare pip" in issue for issue in issues))

    def test_static_policy_rejects_archived_agents_reference(self):
        issues = check_static_policy(
            {"AGENTS.md": "See data/snapshot_row13_row16_wip/docs/AGENTS.md\n"},
            root=Path("."),
        )
        self.assertTrue(any("archived AGENTS.md" in issue for issue in issues))


ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = ROOT / "rules"
RULES_INDEX = RULES_DIR / "index.yaml"
RULES_README = RULES_DIR / "README.md"
AGENT_DOCS = ROOT / "docs" / "agent"

# Design-required fields plus the brief fields (design adds 'supersedes').
RULE_FIELDS = (
    "id",
    "status",
    "scope",
    "applies_when",
    "owner",
    "last_reviewed",
    "supersedes",
    "evidence",
)
RULE_STATUSES = frozenset({"draft", "active", "superseded", "retired"})

# rule file -> substrings that must appear in the named policy rule's body so
# the rules stay genuinely conditional rather than generic prose.
POLICY_CONTENT_ANCHORS = {
    "gpu.md": ("co-run", "shared-memory", "oom"),
    "experiments.md": ("pre-register", "anchor", "hard-fail", "smoke", "partial"),
    "data.md": ("atomic", "mmap", "regenerat"),
    "evidence.md": ("remembered numbers", "relative"),
    "context.md": ("UNKNOWN", "historical"),
}

COMPONENT_IDS = (
    "core",
    "diacritizer",
    "laya",
    "ui",
    "mex",
    "langid",
    "flow",
    "viz",
    "webui",
)

# Root-policy sentences owned by the top-level rules; component files must not
# repeat them (they reference the parent rule instead).
ROOT_POLICY_OWNED_PHRASES = (
    "one protected repository gpu training job",
    "never kill a foreign process",
)


def _rule_front_matter(testcase: unittest.TestCase, rule_file: str) -> dict:
    """Parse and return the YAML front matter block of a rule file."""
    path = ROOT / "rules" / rule_file
    lines = path.read_text(encoding="utf-8").splitlines()
    testcase.assertTrue(
        lines and lines[0].strip() == "---",
        f"rules/{rule_file} must begin with a --- front matter fence",
    )
    end = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
    )
    testcase.assertIsNotNone(
        end, f"rules/{rule_file} front matter is not closed with ---"
    )
    data = yaml.safe_load("\n".join(lines[1:end]))
    testcase.assertIsInstance(
        data, dict, f"rules/{rule_file} front matter must parse to a mapping"
    )
    return data


class RuleContextTests(unittest.TestCase):
    '''Task 3: conditional rules, SOUL, and docs/agent structure tests.'''

    def _index(self) -> list[dict]:
        self.assertTrue(
            RULES_INDEX.is_file(), "rules/index.yaml must exist and list rules"
        )
        data = yaml.safe_load(RULES_INDEX.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        entries = data.get("rules", [])
        self.assertIsInstance(entries, list, "index rules must be a list")
        for entry in entries:
            self.assertIsInstance(entry, dict)
            self.assertTrue(
                isinstance(entry.get("path"), str) and entry["path"].strip(),
                "each index entry needs a rule path",
            )
        self.assertTrue(
            RULES_README.is_file(), "rules/README.md must exist"
        )
        return list(entries)

    def test_index_points_only_to_existing_rule_files(self):
        for entry in self._index():
            self.assertTrue(
                (RULES_DIR / entry["path"]).is_file(),
                f"index path does not exist: rules/{entry['path']}",
            )

    def test_index_paths_match_the_on_disk_rules_exactly(self):
        index_paths = [entry["path"] for entry in self._index()]
        self.assertEqual(
            len(index_paths),
            len(set(index_paths)),
            "index entries must not repeat a rule path",
        )
        on_disk = sorted(
            path.relative_to(RULES_DIR).as_posix()
            for path in RULES_DIR.glob("**/*.md")
            if path.name != "README.md"
        )
        self.assertEqual(sorted(index_paths), on_disk)

    def test_every_rule_has_required_front_matter_fields(self):
        for entry in self._index():
            front = _rule_front_matter(self, entry["path"])
            for field in RULE_FIELDS:
                self.assertIn(field, front, f"{entry['path']} missing {field}")
            for field in RULE_FIELDS:
                if field in ("supersedes",):
                    continue
                value = front[field]
                self.assertTrue(
                    isinstance(value, str) and value.strip() or isinstance(value, list) and value,
                    f"{entry['path']} field {field} must be non-empty",
                )
            self.assertIn(
                front["status"],
                RULE_STATUSES,
                f"{entry['path']} invalid rule status: {front['status']}",
            )

    def test_rule_evidence_paths_exist_and_are_relative(self):
        for entry in self._index():
            front = _rule_front_matter(self, entry["path"])
            for evidence_path in front["evidence"]:
                self.assertIsInstance(evidence_path, str)
                self.assertFalse(
                    Path(evidence_path).is_absolute()
                    or re.match(r"^[A-Za-z]:[\\/]", evidence_path)
                    or evidence_path.startswith("/"),
                    f"{entry['path']} evidence path must be relative: {evidence_path}",
                )
                self.assertTrue(
                    (ROOT / evidence_path).exists(),
                    f"{entry['path']} missing evidence path: {evidence_path}",
                )

    def test_policy_rules_carry_their_conditions(self):
        for rule_file, anchors in POLICY_CONTENT_ANCHORS.items():
            body = (RULES_DIR / rule_file).read_text(encoding="utf-8").lower()
            for anchor in anchors:
                self.assertIn(
                    anchor.lower(),
                    body,
                    f"rules/{rule_file} must mention {anchor!r} (conditional policy)",
                )

    def test_component_rules_do_not_duplicate_root_policy(self):
        self.assertTrue(self._index())
        for component_id in COMPONENT_IDS:
            front = _rule_front_matter(self, f"components/{component_id}.md")
            self.assertEqual(
                front["scope"],
                component_id,
                f"component rule {component_id} scope must equal its id",
            )
            body = (RULES_DIR / "components" / f"{component_id}.md").read_text(
                encoding="utf-8"
            ).lower()
            for phrase in ROOT_POLICY_OWNED_PHRASES:
                self.assertNotIn(
                    phrase,
                    body,
                    f"components/{component_id}.md duplicates root policy: {phrase!r}",
                )

    def test_agent_docs_document_operations_and_schema(self):
        for required in (
            AGENT_DOCS / "OPERATIONS.md",
            AGENT_DOCS / "SCHEMA.md",
        ):
            self.assertTrue(
                required.is_file(), f"{required.name} must exist under docs/agent/"
            )
        operations = (AGENT_DOCS / "OPERATIONS.md").read_text(encoding="utf-8").lower()
        for anchor in ("launch gate", "read-back", "smoke", "resume", "gpu"):
            self.assertIn(
                anchor.lower(), operations,
                f"OPERATIONS.md must cover {anchor!r}",
            )
        schema = (AGENT_DOCS / "SCHEMA.md").read_text(encoding="utf-8").lower()
        for anchor in ("record fields", "status vocabular", "front matter", "UNKNOWN"):
            self.assertIn(
                anchor.lower(), schema,
                f"SCHEMA.md must cover {anchor!r}",
            )
        README = ROOT / "rules" / "README.md"
        for source in (README, AGENT_DOCS / "OPERATIONS.md", AGENT_DOCS / "SCHEMA.md"):
            self.assertEqual(
                check_markdown_links(source, ROOT), [],
                f"broken links in {source.name}",
            )
            issues = check_static_policy({source.name: source.read_text(encoding="utf-8")}, ROOT)
            self.assertEqual(issues, [], f"static policy issues in {source.name}")


# ---------------------------------------------------------------- Task 4 ----

ACTIVE_COMPONENT_DIRS = (
    "diacritizer",
    "flow",
    "langid",
    "laya",
    "mex",
    "ui",
    "viz",
    "webui",
)

# Design layering (section 4): the root router must hand an arriving agent
# to the identity, current-state, conditional-rules, and operations layers.
REQUIRED_ROOT_LINKS = (
    "SOUL.md",
    "CURRENT_STATE.md",
    "rules/README.md",
    "docs/agent/OPERATIONS.md",
)

# docs/agent/SCHEMA.md owns this vocabulary; the router must declare exactly
# it, exactly once.
CANONICAL_VOCABULARY = frozenset(
    {"PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"}
)

CLAUDE_REQUIRED_ITEMS = (
    "AGENTS.md",
    "CURRENT_STATE.md",
    "direct user",
    "dirty",
    "GPU co-run",
    "destructive",
    "concise",
)

# Heavy manual sections the thin adapter must not resurrect.
CLAUDE_REMOVED_SECTIONS = (
    "priority order",
    "core principles",
    "security rules",
    "error handling and recovery",
    "tool-use economy",
    "hardware utilisation protocol",
    "definition of done",
)


class RootAdapterTests(unittest.TestCase):
    """Task 4: root router AGENTS.md + thin CLAUDE.md adapter regressions."""

    def _root_text(self, name):
        path = ROOT / name
        self.assertTrue(path.is_file(), f"root {name} must exist")
        return path.read_text(encoding="utf-8")

    def test_root_agents_md_contains_no_live_or_machine_claims(self):
        issues = check_static_policy(
            {"AGENTS.md": self._root_text("AGENTS.md")}, ROOT
        )
        self.assertEqual([], issues)

    def test_root_agents_md_routes_every_active_component_directory(self):
        text = self._root_text("AGENTS.md")
        for component in ACTIVE_COMPONENT_DIRS:
            self.assertIn(
                component, text, f"root AGENTS.md must route {component}/"
            )

    def test_active_component_directories_own_scoped_agents_files(self):
        for component in ACTIVE_COMPONENT_DIRS:
            self.assertTrue(
                (ROOT / component / "AGENTS.md").is_file(),
                f"{component}/AGENTS.md must exist as the nearest scoped guide",
            )

    def test_active_component_agents_files_are_policy_light(self):
        for component in ACTIVE_COMPONENT_DIRS:
            path = ROOT / component / "AGENTS.md"
            issues = check_static_policy(
                {path.as_posix(): path.read_text(encoding="utf-8")}, ROOT
            )
            self.assertEqual([], issues, f"static policy issues in {path}")

    def test_root_agents_md_links_required_policy_targets(self):
        text = self._root_text("AGENTS.md")
        for target in REQUIRED_ROOT_LINKS:
            self.assertIn(
                target, text, f"root AGENTS.md must reference {target}"
            )
        self.assertEqual(
            [], check_markdown_links(ROOT / "AGENTS.md", ROOT)
        )

    def test_root_agents_md_uses_one_validation_vocabulary(self):
        text = self._root_text("AGENTS.md")
        lines = [
            line
            for line in text.splitlines()
            if re.search(r"validation\s+label", line, re.I)
        ]
        self.assertEqual(
            1,
            len(lines),
            "root AGENTS.md must declare the validation vocabulary exactly once",
        )
        self.assertEqual(
            set(CANONICAL_VOCABULARY),
            set(re.findall(r"`([A-Z][A-Za-z_]*)`", lines[0])),
        )

    def test_root_agents_md_preserves_platform_then_user_authority(self):
        text = self._root_text("AGENTS.md")
        lowered = text.lower()
        platform = lowered.index("platform and system safety always come first")
        user = lowered.index("direct user instructions")
        self.assertLess(platform, user)
        self.assertIn("outrank all other repository guidance", lowered)
        self.assertIn("within platform and system safety", self._root_text("SOUL.md"))

    def test_claude_md_is_a_thin_scoped_adapter(self):
        text = self._root_text("CLAUDE.md")
        lowered = text.lower()
        normalized = re.sub(r"\s+", " ", lowered)
        self.assertLess(
            normalized.index("platform and system safety remain first"),
            normalized.index("direct user instructions outrank"),
        )
        self.assertLessEqual(
            len(text.splitlines()),
            90,
            "CLAUDE.md must stay a thin adapter, not a duplicate manual",
        )
        for item in CLAUDE_REQUIRED_ITEMS:
            self.assertIn(
                item.lower(), lowered, f"thin CLAUDE.md must mention {item!r}"
            )
        for section in CLAUDE_REMOVED_SECTIONS:
            self.assertNotIn(
                section, lowered,
                f"thin CLAUDE.md must not restate {section!r}",
            )
        self.assertEqual(
            [], check_markdown_links(ROOT / "CLAUDE.md", ROOT)
        )

    def test_archived_snapshot_agents_is_not_discoverable_as_active_scope(self):
        source = ROOT / "data" / "snapshot_row13_row16_wip" / "docs" / "AGENTS.md"
        destination = (
            ROOT
            / "data"
            / "snapshot_row13_row16_wip"
            / "docs"
            / "AGENTS.archive.md"
        )
        self.assertFalse(
            source.exists(),
            "the snapshot instruction file must be archived, not active scope",
        )
        self.assertTrue(
            destination.is_file(), "the archived instruction file must exist"
        )

class LintTreeHelperTestCase(unittest.TestCase):
    """Focused tests for the public, read-only lint_tree helper."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_lint_tree_reports_record_issues_read_only(self):
        self._write(
            "state/tasks.yaml",
            "tasks:\n  - id: mock/tiny\n    status: nonsense\n    evidence: []\n",
        )
        issues = lint_tree(self.root, include_policy=False)
        self.assertTrue(any("invalid task status" in i for i in issues))
        # read-only: no file was created or modified besides the fixture
        self.assertEqual(
            sorted(p.name for p in self.root.rglob("*")),
            ["state", "tasks.yaml"],
        )

    def test_lint_tree_contains_deep_yaml_recursion(self):
        nested = "tasks: " + "[" * 2000 + "0" + "]" * 2000 + "\n"
        self._write("state/tasks.yaml", nested)
        try:
            issues = lint_tree(self.root, include_policy=False)
        except RecursionError as exc:
            self.fail(f"lint_tree leaked RecursionError: {exc}")

        joined = "\n".join(issues)
        self.assertTrue(any("cannot parse state/tasks.yaml" in issue for issue in issues))
        self.assertNotIn("RecursionError", joined)
        self.assertNotIn(nested, joined)

    def test_lint_tree_clean_root_returns_no_issues(self):
        self._write(
            "state/tasks.yaml",
            "tasks: []\nexperiments: []\n",
        )
        self._write("README.md", "# Clean\n")
        self.assertEqual(lint_tree(self.root, include_policy=False), [])

    def test_malformed_dedicated_experiments_file_does_not_use_embedded_fallback(self):
        self._write(
            "state/tasks.yaml",
            "tasks: []\nexperiments:\n"
            "  - id: mock/one\n"
            "    status: planned\n"
            "    evidence: []\n"
            "  - id: mock/one\n"
            "    status: planned\n"
            "    evidence: []\n",
        )
        self._write("research/experiments.yaml", "not-a-mapping\n")
        issues = lint_tree(self.root, include_policy=False)
        self.assertTrue(any("research/experiments.yaml must contain a mapping" in i for i in issues))
        self.assertFalse(any("duplicate experiment id" in i for i in issues))

    def test_lint_tree_rejects_symlink_before_reading_any_files(self):
        secret = Path(self._tmp.name) / "secret.txt"
        secret.write_text("LEAKED_SECRET_VALUE\n", encoding="utf-8")
        link = self.root / "linked-secret.txt"
        try:
            link.symlink_to(secret)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with patch(
            "scripts.check_agent_context._current_records",
            side_effect=AssertionError("records must not be read"),
        ):
            issues = lint_tree(self.root)
        self.assertTrue(any("unsafe" in issue for issue in issues))
        self.assertNotIn("LEAKED_SECRET_VALUE", "\n".join(issues))

    def test_lint_tree_rejects_oversized_context_file_before_full_read(self):
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        oversized = self.root / "README.md"
        oversized.write_bytes(b"x" * ((1 << 20) + 1))
        issues = lint_tree(self.root, include_policy=False)
        self.assertTrue(any("size limit" in issue for issue in issues))
        self.assertFalse(any("mapping" in issue for issue in issues))

    def test_lint_tree_ignores_node_modules_descendant_links(self):
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        node_modules = self.root / "node_modules"
        node_modules.mkdir()
        target = self.root / "outside-dependency"
        target.mkdir()
        try:
            (node_modules / "react").symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlink creation unavailable: {exc}")
        self.assertEqual(lint_tree(self.root, include_policy=False), [])

    def test_lint_tree_ignores_skillopt_control_descendant_links(self):
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        staging = self.root / ".skillopt-sleep"
        staging.mkdir()
        target = self.root / "outside-staging"
        target.mkdir()
        try:
            (staging / "latest").symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlink creation unavailable: {exc}")
        self.assertEqual(lint_tree(self.root, include_policy=False), [])

    def _write_exception_registry(self, text):
        self._write("state/context_exceptions.yaml", text)

    def _empty_exception_registry(self):
        self._write_exception_registry("schema_version: 1\nexceptions: []\n")

    def _write_required_policy_files(self):
        self._write("AGENTS.md", "# agents\n")
        self._write("CLAUDE.md", "# claude\n")
        self._write("SOUL.md", "# soul\n")
        self._write("rules/nested.md", "# rule\n")
        self._empty_exception_registry()

    def _write_clean_records(self):
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")

    def test_lint_tree_enforces_policy_checks_by_default(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("AGENTS.md", "run python -m tool\n")
        self._write("SOUL.md", "pip freeze > file.txt\n")
        self._write("rules/nested.md", "a bare python call lives here\n")
        default_issues = lint_tree(self.root)
        without_policy = lint_tree(self.root, include_policy=False)
        self.assertFalse(any("bare" in i for i in without_policy))
        self.assertFalse(any("missing required policy file" in i for i in without_policy))
        self.assertTrue(any("bare Python command: AGENTS.md" in i for i in default_issues))
        self.assertTrue(any("bare Python command: rules/nested.md" in i for i in default_issues))
        self.assertTrue(any("bare pip command: SOUL.md" in i for i in default_issues))

    def test_lint_tree_default_requires_policy_files(self):
        self._write_clean_records()
        issues = lint_tree(self.root)
        for required in ("AGENTS.md", "CLAUDE.md", "SOUL.md"):
            self.assertTrue(
                any(f"missing required policy file: {required}" in issue for issue in issues),
                f"default lint must require {required}: {issues}",
            )
        self.assertTrue(
            any("missing required policy file: rules/*.md" in issue for issue in issues),
            issues,
        )
        self._write_required_policy_files()
        self.assertEqual(lint_tree(self.root), [])

    def test_lint_tree_default_reports_live_and_archived_policy_claims(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("AGENTS.md", "PID 123 is live as of now.\n")
        self._write(
            "rules/nested.md",
            "See data/snapshot_row13_row16_wip/docs/AGENTS.md\n",
        )
        issues = lint_tree(self.root)
        self.assertTrue(any("live process claim: AGENTS.md:1" in issue for issue in issues))
        self.assertTrue(
            any("archived AGENTS.md reference: rules/nested.md:1" in issue for issue in issues)
        )

    def test_main_enforces_policy_checks_by_default(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("AGENTS.md", "run python -m tool\n")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("ISSUE: bare Python command: AGENTS.md:1", stdout.getvalue())

    def test_main_no_policy_flag_opts_out_of_policy_checks(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("AGENTS.md", "run python -m tool\n")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--root", str(self.root), "--no-policy"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        with redirect_stdout(io.StringIO()) as default_stdout:
            default_code = main(["--root", str(self.root)])
        self.assertEqual(default_code, 1)
        self.assertIn("ISSUE: bare Python command: AGENTS.md:1", default_stdout.getvalue())


    def _accepted_registry(self, *records):
        return "schema_version: 1\nexceptions:\n" + "".join(records)

    def _table_exception(self, identifier, line, target=None):
        record = (
            "  - id: context-exception/" + identifier + "\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: " + str(line) + "\n"
            "    kind: malformed_table_row\n"
            "    reason: accepted fixture\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        return record

    def _link_exception(self, identifier, line, target):
        return (
            "  - id: context-exception/" + identifier + "\n"
            "    status: accepted_historical\n"
            "    path: research/EXPERIMENTS.md\n"
            "    line: " + str(line) + "\n"
            "    kind: broken_markdown_link\n"
            "    target: " + target + "\n"
            "    reason: accepted fixture\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )

    def test_default_policy_accepts_exact_historical_findings(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write(
            "research/EXPERIMENTS.md",
            "[missing](research/_template_experiment.md)\n",
        )
        self._write_exception_registry(
            self._accepted_registry(
                self._table_exception("tasks-1", 1),
                self._link_exception("link-1", 1, "research/_template_experiment.md"),
            )
        )
        accepted = []
        issues = lint_tree(self.root, accepted_exceptions=accepted)
        self.assertEqual([], issues)
        self.assertEqual(2, len(accepted))
        self.assertTrue(any("TASKS.md:1" in item for item in accepted))
        self.assertTrue(any("research/EXPERIMENTS.md:1" in item for item in accepted))

    def test_changed_exception_finding_remains_an_issue(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write_exception_registry(
            self._accepted_registry(self._table_exception("tasks-2", 2))
        )
        accepted = []
        issues = lint_tree(self.root, accepted_exceptions=accepted)
        self.assertEqual([], accepted)
        self.assertTrue(
            any("malformed table row: TASKS.md:1" in issue for issue in issues)
        )

    def test_changed_exception_link_target_remains_an_issue(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write(
            "research/EXPERIMENTS.md",
            "[missing](research/OTHER_PLACEHOLDER.md)\n",
        )
        self._write_exception_registry(
            self._accepted_registry(
                self._link_exception("link-1", 1, "research/_template_experiment.md")
            )
        )
        accepted = []
        issues = lint_tree(self.root, accepted_exceptions=accepted)
        self.assertEqual([], accepted)
        self.assertTrue(
            any(
                "broken markdown link: research/EXPERIMENTS.md:1" in issue
                for issue in issues
            )
        )

    def test_record_only_mode_does_not_apply_exceptions(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write_exception_registry(
            self._accepted_registry(self._table_exception("tasks-1", 1))
        )
        issues = lint_tree(self.root, include_policy=False)
        self.assertTrue(
            any("malformed table row: TASKS.md:1" in issue for issue in issues)
        )

    def test_missing_registry_is_a_controlled_issue_in_policy_mode(self):
        self._write_clean_records()
        self._write("AGENTS.md", "# agents\n")
        self._write("CLAUDE.md", "# claude\n")
        self._write("SOUL.md", "# soul\n")
        self._write("rules/nested.md", "# rule\n")
        accepted = []
        issues = lint_tree(self.root, accepted_exceptions=accepted)
        self.assertIn(
            "missing required context exception registry: state/context_exceptions.yaml",
            issues,
        )
        self.assertEqual([], accepted)

    def test_malformed_registry_does_not_accept_anything(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write_exception_registry("schema_version: 1\nexceptions: {}\n")
        accepted = []
        issues = lint_tree(self.root, accepted_exceptions=accepted)
        self.assertEqual([], accepted)
        self.assertTrue(
            any("malformed table row: TASKS.md:1" in issue for issue in issues)
        )
        self.assertTrue(any("context exception" in issue for issue in issues))

    def test_main_prints_accepted_historical_lines_and_exits_zero(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write_exception_registry(
            self._accepted_registry(self._table_exception("tasks-1", 1))
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--root", str(self.root)])
        output = stdout.getvalue()
        self.assertEqual(0, code)
        self.assertIn("ACCEPTED HISTORICAL: TASKS.md:1", output)
        self.assertNotIn("ISSUE: malformed table row", output)

    def test_main_max_issues_bounds_accepted_and_issue_output(self):
        self._write_clean_records()
        self._write_required_policy_files()
        self._write("TASKS.md", "| broken || row |\n")
        self._write("AGENTS.md", "run python -m tool\n")
        self._write_exception_registry(
            self._accepted_registry(self._table_exception("tasks-1", 1))
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--root", str(self.root), "--max-issues", "1"])
        lines = stdout.getvalue().splitlines()
        self.assertEqual(1, code)
        self.assertTrue(lines[0].startswith("ACCEPTED HISTORICAL: TASKS.md:1"))
        self.assertTrue(
            any("additional issues omitted: 1" in line for line in lines), lines
        )

    def test_real_repository_lints_clean_with_accepted_exceptions(self):
        root = Path(__file__).resolve().parents[2]
        accepted = []
        issues = lint_tree(root, accepted_exceptions=accepted)
        self.assertEqual([], issues)
        self.assertEqual(4, len(accepted))
        for line in accepted:
            self.assertTrue(line.startswith("ACCEPTED HISTORICAL: "), line)


def _controlled_issue(issues, needle):
    """Return the single issue containing *needle* or fail with a bounded dump."""
    matches = [issue for issue in issues if needle in issue]
    if not matches:
        raise AssertionError(f"no {needle!r} diagnostic in {issues!r}")
    return matches[0]


class LinterDiagnosticBoundaryTests(unittest.TestCase):
    """The linter must never echo untrusted paths or exception strings."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _assert_not_disclosed(self, issues, token):
        rendered = "\n".join(issues)
        self.assertTrue(issues)
        self.assertNotIn(token, rendered)
        self.assertNotIn("Traceback", rendered)
        for issue in issues:
            self.assertNotIn("simulated", issue)
            self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def _assert_redacted(self, issues, token):
        self._assert_not_disclosed(issues, token)
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, "\n".join(issues))

    def test_preflight_filesystem_error_diagnostic_is_controlled(self):
        token = "github_pat_" + "R" * 30
        with patch(
            "scripts.check_agent_context._lstat_component",
            side_effect=OSError(f"simulated lstat failure for {token}"),
        ):
            issues = lint_tree(self.root)
        self._assert_not_disclosed(issues, token)  # no raw exception escapes
        issue = _controlled_issue(issues, "metadata could not be inspected")
        self.assertIn("unsafe context path:", issue)

    def test_induced_read_error_diagnostic_is_controlled(self):
        token = "github_pat_" + "Q" * 30
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        self._write("README.md", f"notes for {token}\n")
        rules = self.root / "rules"
        rules.mkdir()
        (rules / f"rule-{token}.md").write_bytes(b"policy: \xff\xfe\x00not-utf8")
        (rules / "clean.md").write_text("# clean\n", encoding="utf-8")
        self._write("AGENTS.md", f"notes for {token}\n")
        self._write("CLAUDE.md", "# claude\n")
        self._write("SOUL.md", "# soul\n")
        self._write("state/context_exceptions.yaml", "schema_version: 1\nexceptions: []\n")
        real_open = Path.open

        def open_with_induced_failures(path, *args, **kwargs):
            if path.name == f"rule-{token}.md":
                raise OSError(f"simulated read failure for {token}")
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=open_with_induced_failures):
            issues = lint_tree(self.root)
        rendered = "\n".join(issues)
        self.assertEqual(rendered, "cannot read <redacted-diagnostic>")
        self.assertNotIn(token, rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertNotIn("simulated", rendered)

    def test_utf8_decode_error_diagnostic_is_controlled(self):
        token = "github_pat_" + "D" * 30
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        self._write("README.md", f"notes for {token}\n")
        rules = self.root / "rules"
        rules.mkdir()
        (rules / f"rule-{token}.md").write_bytes(b"policy: \xff\xfe\x00not-utf8")
        (rules / "clean.md").write_text("# clean\n", encoding="utf-8")
        self._write("AGENTS.md", f"notes for {token}\n")
        self._write("CLAUDE.md", "# claude\n")
        self._write("SOUL.md", "# soul\n")
        issues = lint_tree(self.root)
        rendered = "\n".join(issues)
        self.assertIn("cannot decode", rendered)
        self.assertNotIn(token, rendered)
        self.assertNotIn("\xff", rendered)
        self.assertNotIn(self.root.name, rendered)

    def test_markdown_missing_secret_filename_is_controlled(self):
        token = "github_pat_" + "F" * 30
        issues = check_markdown_links(self.root / f"{token}.md", self.root)
        self._assert_redacted(issues, token)
        self.assertIn("missing markdown file:", issues[0])

    def test_jwt_and_bearer_diagnostics_are_redacted(self):
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        )
        bearer = "mcp_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
        for target in (
            f"{jwt}.md",
            f"missing-{jwt}.md",
            f"Bearer:{bearer}",
            f"Bearer:{bearer}.md",
        ):
            with self.subTest(target_length=len(target)):
                self._write("docs/note.md", f"# source\n[leak]({target})\n")
                issues = check_markdown_links(self.root / "docs/note.md", self.root)
                self._assert_redacted(issues, target)
        evidence = check_records(
            {
                "tasks": [
                    {
                        "id": "ui/u1",
                        "status": "done",
                        "evidence": [f"missing/{jwt}.json"],
                    }
                ],
                "experiments": [],
            },
            self.root,
        )
        self.assertIn("<redacted-secret-evidence>", "\n".join(evidence))
        self.assertNotIn(jwt, "\n".join(evidence))

    def test_unicode_tag_characters_are_redacted(self):
        for codepoint in (0xE0041, 0xE007F, 0xE0001):
            with self.subTest(codepoint=f"U+{codepoint:04X}"):
                tag = chr(codepoint)
                self._write(f"docs/{tag}-suffix.md", "# diagnostic source\n")
                issues = check_markdown_links(self.root / f"docs/{tag}.md", self.root)
                self._assert_redacted(issues, tag)
                issues = check_records(
                    {
                        "tasks": [],
                        "experiments": [],
                        "tables": {f"docs/{tag}.md": "| a || b |\n"},
                    },
                    self.root,
                )
                self._assert_redacted(issues, tag)
        self.assertTrue(_FORMAT_DIAGNOSTIC_PATTERN.search(chr(0xE0041)))
        self.assertFalse(_FORMAT_DIAGNOSTIC_PATTERN.search("scratch/note.txt"))

    def test_absolute_path_normalization_failure_is_controlled(self):
        token = "raw github_pat_" + "Z" * 30
        entries = (
            (lint_tree, (self.root,), {}),
            (check_records, ({"tasks": [], "experiments": []}, self.root), {}),
            (check_markdown_links, (self.root / "docs" / "note.md", self.root), {}),
        )
        for function, args, kwargs in entries:
            with self.subTest(function=function.__name__):
                with patch(
                    "scripts.check_agent_context.os.path.abspath",
                    side_effect=OSError(token),
                ):
                    issues = function(*args, **kwargs)
                rendered = "\n".join(issues)
                self.assertTrue(issues, function.__name__)
                self.assertNotIn(token, rendered)
                self.assertNotIn("Traceback", rendered)
                self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)
                for issue in issues:
                    self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)
        with patch(
            "scripts.check_agent_context.os.path.abspath",
            side_effect=OSError(token),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertNotIn(token, stdout.getvalue())
        self.assertNotIn("Traceback", stdout.getvalue())
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, stdout.getvalue())

    def test_missing_root_returns_controlled_issues(self):
        missing = self.root / "nonexistent"
        issues = lint_tree(missing)
        rendered = "\n".join(issues)
        self.assertTrue(issues)
        self.assertIn("missing context path", rendered)
        self.assertNotIn("TypeError", rendered)
        self.assertNotIn("exceptions must derive", rendered)
        self.assertNotIn("Traceback", rendered)
        for issue in issues:
            self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)
        with patch(
            "scripts.check_agent_context._lstat_component",
            side_effect=OSError("raw github_pat_" + "Y" * 30),
        ):
            issues = lint_tree(missing)
        rendered = "\n".join(issues)
        self.assertTrue(issues)
        self.assertNotIn("github_pat_", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_cli_argument_diagnostics_are_bounded(self):
        token = "github_pat_" + "X" * 30
        cases = (
            [f"--unknown-{token}"],
            [f"--unknown-{token}", "value"],
            ["--root"],
            ["--max-issues", "not-a-number"],
            [f"--unknown-{token}", "--root"],
        )
        for argv in cases:
            with self.subTest(argv_length=len(argv)):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(argv)
                self.assertEqual(raised.exception.code, 2)
                rendered = stderr.getvalue()
                self.assertNotIn(token, rendered)
                self.assertNotIn("Traceback", rendered)
                for line in rendered.splitlines():
                    self.assertLessEqual(len(line), MAX_ISSUE_CHARS)

    def test_pem_private_key_block_diagnostics_are_redacted(self):
        header = "-----BEGIN OPENSSH PRIVATE KEY-----"
        body = (
            "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB"
            "AAAAMwAAAAtzc2gtZWQyNTUxOQAAACB8Y3KQlM6XcHp5nJ2F"
        )
        footer = "-----END OPENSSH PRIVATE KEY-----"
        block = f"{header}\n{body}\n{footer}"
        issues = check_records(
            {
                "tasks": [],
                "experiments": [],
                "tables": {f"docs/{block}": "| a || b |\n"},
            },
            self.root,
        )
        self._assert_redacted(issues, header)
        self._assert_redacted(issues, body)

    def test_ordinary_paths_are_not_redacted(self):
        source = self.root / "scratch" / "note.txt"
        self._write("scratch/note.txt", "scratch/note.txt\n")
        issues = check_records(
            {
                "tasks": [
                    {
                        "id": "ui/u1",
                        "status": "done",
                        "evidence": ["scratch/note.txt"],
                    }
                ],
                "experiments": [],
                "tables": {"scratch/note.txt": "| a | b |\n| --- | --- |\n| 1 | 2 |\n"},
            },
            self.root,
        )
        self.assertEqual(issues, [])
        issues = check_markdown_links(source, self.root)
        self.assertEqual(issues, [])

    def test_unexpected_helper_exception_is_sanitized_in_main(self):
        token = "github_pat_" + "V" * 30
        for name in (
            "lint_tree",
            "check_records",
            "check_markdown_links",
            "check_static_policy",
            "_current_records",
            "_preflight_tree",
):
            with self.subTest(helper=name):
                with patch(
                    f"scripts.check_agent_context.{name}",
                    side_effect=TypeError("raw " + token + chr(0)),
                ):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = main(["--root", str(self.root), "--no-policy"])
                combined = stdout.getvalue() + stderr.getvalue()
                self.assertEqual(code, 1)
                self.assertNotIn(token, combined)
                self.assertNotIn(chr(0), combined)
                self.assertNotIn("Traceback", combined)
                self.assertNotIn("TypeError", combined)
                for line in combined.splitlines():
                    self.assertLessEqual(len(line), MAX_ISSUE_CHARS)

    def test_unexpected_helper_exception_is_sanitized_in_lint_tree(self):
        token = "github_pat_" + "U" * 30
        self._write("README.md", "# notes\n")
        for name in (
            "check_records",
            "check_markdown_links",
            "check_static_policy",
            "_current_records",
):
            with self.subTest(helper=name):
                with patch(
                    f"scripts.check_agent_context.{name}",
                    side_effect=RuntimeError("raw " + token + chr(0)),
                ):
                    issues = lint_tree(self.root, include_policy=False)
                rendered = "\n".join(issues)
                self.assertTrue(issues)
                self.assertNotIn(token, rendered)
                self.assertNotIn(chr(0), rendered)
                self.assertNotIn("Traceback", rendered)
                self.assertNotIn("RuntimeError", rendered)
                for issue in issues:
                    self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def test_metadata_error_diagnostics_expose_no_absolute_path(self):
        self._write("README.md", "# notes\n")
        self._write("state/tasks.yaml", "tasks: []\nexperiments: []\n")
        for target in ("os.scandir", "_lstat_component"):
            with self.subTest(target=target):
                with patch(
                    f"scripts.check_agent_context.{target}",
                    side_effect=OSError("boom"),
                ):
                    issues = lint_tree(self.root, include_policy=False)
                rendered = "\n".join(issues)
                self.assertTrue(issues)
                self.assertIn("metadata could not be inspected", rendered)
                self.assertNotIn(str(self.root), rendered)
                self.assertNotIn(self.root.name, rendered)
                self.assertNotIn("boom", rendered)
                self.assertNotIn("Traceback", rendered)
                self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)
                for issue in issues:
                    self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def test_direct_lstat_oserror_diagnostic_hides_absolute_path(self):
        self._write("README.md", "# notes\n")
        with patch(
            "scripts.check_agent_context.os.lstat",
            side_effect=OSError("boom"),
        ):
            issues = lint_tree(self.root, include_policy=False)
        rendered = "\n".join(issues)
        self.assertIn("metadata could not be inspected", rendered)
        self.assertNotIn(str(self.root), rendered)
        self.assertNotIn(self.root.name, rendered)
        self.assertNotIn("boom", rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)

    def test_reparse_metadata_error_hides_absolute_path(self):
        self._write("README.md", "# notes\n")
        with patch(
            "scripts.check_agent_context._WIN",
            True,
        ), patch(
            "scripts.check_agent_context._windows_file_attributes",
            side_effect=OSError("boom"),
        ):
            issues = lint_tree(self.root, include_policy=False)
        rendered = "\n".join(issues)
        self.assertTrue(issues)
        self.assertIn("reparse metadata could not be inspected", rendered)
        self.assertNotIn(str(self.root), rendered)
        self.assertNotIn(self.root.name, rendered)
        self.assertNotIn("boom", rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)

    def test_cli_unicode_issue_is_stream_safe(self):
        class AsciiStream(io.StringIO):
            encoding = "ascii"

            def write(self, value):
                value.encode("ascii")
                return super().write(value)

        stdout = AsciiStream()
        with patch(
            "scripts.check_agent_context.lint_tree",
            return_value=["café"],
        ), redirect_stdout(stdout):
            code = main(["--root", str(self.root), "--no-policy"])
        output = stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("ISSUE:", output)
        self.assertIn(r"caf\xe9", output)
        self.assertNotIn("UnicodeEncodeError", output)

    def test_sanitized_errors_have_no_raw_cause_chain(self):
        token = "github_pat_" + "T" * 30
        self._write("README.md", "# notes\n")
        with patch(
            "scripts.check_agent_context._lstat_component",
            side_effect=OSError("raw " + token),
        ):
            issues = lint_tree(self.root, include_policy=False)
        rendered = "\n".join(issues)
        self.assertNotIn(token, rendered)
        try:
            check_records({"tasks": [], "experiments": []}, self.root)
        except BaseException as exc:  # pragma: no cover - the public call must not raise
            self.fail(f"check_records raised {type(exc).__name__}")

    def test_invalid_path_types_return_controlled_issues(self):
        for bad in (None, object(), 123, 4.5, True, [str]):
            with self.subTest(bad_type=type(bad).__name__):
                cases = (
                    (lint_tree, (bad,), {}),
                    *(
                        [(lint_tree, (self.root,), {"evidence_root": bad})]
                        if bad is not None
                        else []
                    ),
                    (check_records, ({"tasks": [], "experiments": []}, bad), {}),
                    (check_markdown_links, (bad, self.root), {}),
                    (check_markdown_links, (self.root / "docs" / "note.md", bad), {}),
                )
                for function, args, kwargs in cases:
                    issues = function(*args, **kwargs)
                    rendered = "\n".join(issues)
                    self.assertTrue(issues)
                    self.assertNotIn("Traceback", rendered)
                    self.assertNotIn("TypeError", rendered)
                    self.assertNotIn("__fspath__", rendered)
                    self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)
                    for issue in issues:
                        self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def test_registry_identifier_diagnostics_are_bounded(self):
        token = "github_pat_" + "G" * 30
        issues = check_records(
            {
                "tasks": [
                    {
                        "id": f"ui/{token}",
                        "status": f"finished-{token}",
                        "evidence": [],
                    }
                ],
                "experiments": [
                    {
                        "id": f"laya/{token}",
                        "status": f"running-{token}",
                        "verdict": f"maybe-{token}",
                        "evidence": [],
                    }
                ],
                "tables": {token: "| a || b |\n"},
            },
            self.root,
        )
        rendered = "\n".join(issues)
        self.assertEqual(
            rendered,
            "\n".join(
                (
                    f"invalid task status: {REDACTED_DIAGNOSTIC_PLACEHOLDER}",
                    f"invalid experiment status: {REDACTED_DIAGNOSTIC_PLACEHOLDER}",
                    f"invalid experiment verdict: {REDACTED_DIAGNOSTIC_PLACEHOLDER}",
                    f"malformed table row: {REDACTED_DIAGNOSTIC_PLACEHOLDER}:1",
                )
            ),
        )
        self.assertIn("invalid task status", rendered)
        self.assertIn("invalid experiment status", rendered)
        self.assertIn("invalid experiment verdict", rendered)
        self.assertIn("malformed table row", rendered)
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)
        self.assertNotIn(token, rendered)
        for issue in issues:
            self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def test_markdown_link_target_and_source_diagnostics_are_controlled(self):
        token = "github_pat_" + "L" * 30
        long_target = "missing/" + ("x" * 300) + ".md"
        source = self.root / f"source-{token}.md"
        source.write_text(
            f"[leak]({token}.md)\n[bare](C:/outside.md)\n"
            f"[long]({long_target})\n[control](bad\x00name.md)\n",
            encoding="utf-8",
        )
        issues = check_markdown_links(source, self.root)
        rendered = "\n".join(issues)
        self.assertTrue(issues)
        self.assertIn("broken markdown link", rendered)
        self.assertIn("unsafe markdown link", rendered)
        self.assertIn(REDACTED_DIAGNOSTIC_PLACEHOLDER, rendered)
        self.assertNotIn(token, rendered)
        self.assertNotIn(long_target, rendered)
        self.assertNotIn("\x00", rendered)
        for issue in issues:
            self.assertLessEqual(len(issue), MAX_ISSUE_CHARS)

    def test_secret_shaped_filenames_are_redacted_in_reported_issues(self):
        token = "github_pat_" + "Z" * 30
        source = self.root / f"docs/{token}.md"
        self._assert_not_disclosed(
            check_records(
                {"tasks": [], "experiments": [], "tables": {source: "| a || b |\n"}},
                self.root,
            ),
            token,
        )
        self._assert_not_disclosed(
            check_static_policy({source: "python script.py\n"}, self.root),
            token,
        )
        outside = self.root / f"outside-{token}.txt"
        outside.write_text("content\n", encoding="utf-8")
        linked = self.root / f"linked-{token}.md"
        try:
            os.link(outside, linked)
        except OSError as exc:
            self.skipTest(f"hard link creation unavailable: {exc}")
        issues = lint_tree(self.root)
        self._assert_redacted(issues, token)
        self.assertIn("hard-linked regular file", "\n".join(issues))

    def test_cli_preflight_diagnostic_is_redacted(self):
        token = "github_pat_" + "C" * 30
        stdout = io.StringIO()
        with patch(
            "scripts.check_agent_context._lstat_component",
            side_effect=OSError(f"simulated lstat failure for {token}"),
        ):
            with redirect_stdout(stdout):
                code = main(["--root", str(self.root)])
        self.assertNotEqual(code, 0)
        output = stdout.getvalue()
        self.assertNotIn(token, output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("simulated", output)


# ---------------------------------------------------------------- Task 8 ----

PLAN_README = ROOT / "docs" / "plans" / "README.md"
RESEARCH_README = ROOT / "research" / "README.md"

INDEX_COLUMN_NAMES = ("path", "scope", "status", "owner", "last-verified", "supersedes")
PLAN_STATUSES = frozenset({"active", "completed", "blocked", "historical"})
PLAN_SCOPES = frozenset(
    {"core", "diacritizer", "laya", "ui", "mex", "langid", "flow", "viz", "webui"}
)

# Docs whose links run through the linter's own checker in Task 8.
INDEX_LINK_SOURCES = (
    ROOT / "README.md",
    ROOT / "ENVIRONMENT.md",
    ROOT / "PLAN.md",
    PLAN_README,
    RESEARCH_README,
)


def _index_rows(path: Path, occurrence: int = 0) -> list[list[str]]:
    """Return index table rows (path/scope/status/owner/last-verified/supersedes)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [
        i
        for i, line in enumerate(lines)
        if "|" in line
        and all(column in line.lower() for column in INDEX_COLUMN_NAMES)
    ]
    if occurrence < 0 or occurrence >= len(headers):
        return []
    header_index = headers[occurrence]
    rows = []
    for line in lines[header_index + 1:]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and set(cells[0]) <= {"-", " ", ":"}:
            continue  # markdown separator row
        rows.append(cells)
    return rows


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _clean_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def _valid_owner(value: str) -> bool:
    return bool(_clean_cell(value))


def _valid_verified(value: str) -> bool:
    value = _clean_cell(value)
    if value == "UNKNOWN":
        return True
    if not _ISO_DATE_RE.fullmatch(value):
        return False
    year, month, day = (int(part) for part in value.split("-"))
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


class DocIndexTests(unittest.TestCase):
    """Task 8: plan/research indexes and human-facing documentation map."""

    def test_plan_index_exists_with_required_columns(self):
        rows = _index_rows(PLAN_README)
        self.assertTrue(rows, "docs/plans/README.md must contain a plan index table")
        for row in rows:
            self.assertEqual(len(row), 6, f"plan index row needs exactly 6 columns: {row}")
            entry_path = _clean_cell(row[0])
            self.assertTrue(
                (ROOT / entry_path).is_file(),
                f"plan index path does not exist: {entry_path}",
            )
            self.assertTrue(
                entry_path.startswith("docs/plans/"),
                f"plan index path must be rooted at docs/plans/: {entry_path}",
            )
            self.assertIn(_clean_cell(row[1]), PLAN_SCOPES, f"plan scope invalid: {row}")
            self.assertIn(_clean_cell(row[2]), PLAN_STATUSES, f"plan status invalid: {row}")
            self.assertTrue(_valid_owner(row[3]), f"plan owner empty in row: {row}")
            self.assertTrue(_valid_verified(row[4]), f"plan last-verified invalid in row: {row}")
            self.assertTrue(_clean_cell(row[5]), f"plan supersedes empty in row: {row}")

    def test_plan_index_covers_every_plan_file_exactly_once(self):
        rows = _index_rows(PLAN_README)
        self.assertTrue(rows, "plan index table must exist first")
        indexed = [row[0].strip().strip("`") for row in rows]
        self.assertEqual(
            len(indexed), len(set(indexed)), "plan index entries must be unique"
        )
        on_disk = sorted(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "docs" / "plans").rglob("*.md")
            if p.name != "README.md"
        )
        self.assertEqual(
            sorted(indexed), on_disk, "plan index must classify every plan file exactly once"
        )

    def test_research_index_exists_with_required_columns(self):
        rows = _index_rows(RESEARCH_README)
        self.assertTrue(rows, "research/README.md must contain an index table")
        for row in rows:
            self.assertEqual(len(row), 6, f"research index row needs exactly 6 columns: {row}")
            entry_path = _clean_cell(row[0])
            self.assertFalse(
                entry_path.startswith("research/raw/"),
                "raw payloads must stay in the separate directory catalog",
            )
            self.assertTrue(
                (ROOT / entry_path).is_file(),
                f"research index path does not exist: {entry_path}",
            )
            self.assertTrue(_clean_cell(row[1]), f"research scope empty in row: {row}")
            self.assertIn(_clean_cell(row[2]), PLAN_STATUSES, f"research status invalid: {row}")
            self.assertTrue(_valid_owner(row[3]), f"research owner invalid in row: {row}")
            self.assertTrue(_valid_verified(row[4]), f"research last-verified invalid in row: {row}")
            self.assertTrue(_clean_cell(row[5]), f"research supersedes empty in row: {row}")

    def test_research_index_covers_every_research_markdown_once(self):
        rows = _index_rows(RESEARCH_README)
        self.assertTrue(rows, "research index table must exist first")
        indexed = [row[0].strip().strip("`") for row in rows]
        self.assertEqual(
            len(indexed), len(set(indexed)), "research index entries must be unique"
        )
        on_disk = sorted(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "research").rglob("*.md")
            if p.name != "README.md" and "raw" not in p.parts
        )
        self.assertEqual(
            sorted(indexed), on_disk, "research index must classify every report md exactly once"
        )

    def test_research_index_names_raw_payload_store_without_listing_it(self):
        text = RESEARCH_README.read_text(encoding="utf-8")
        self.assertIn("research/raw/", text)
        self.assertLessEqual(
            text.count("research/raw/"), 6,
            "the index must name the raw store generically, not list its payloads",
        )

    def test_readme_is_a_portfolio_overview_linking_all_scoped_entry_points(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("portfolio", text.lower())
        for component in ACTIVE_COMPONENT_DIRS:
            self.assertIn(
                f"({component}/AGENTS.md)",
                text,
                f"README must link the scoped entry point {component}/AGENTS.md",
            )
        self.assertIn("core", text.lower())
        self.assertIn("S/P/T", text)
        self.assertEqual([], check_markdown_links(ROOT / "README.md", ROOT))

    def test_environment_delegates_volatile_facts_and_keeps_stable_policy(self):
        text = (ROOT / "ENVIRONMENT.md").read_text(encoding="utf-8")
        self.assertIn("state/environment.json", text)
        self.assertIn("CURRENT_STATE.md", text)
        self.assertIn("UNKNOWN", text)
        self.assertIn(".venv", text)
        self.assertIn("fp16", text)
        self.assertNotIn("not bf16", text.lower())
        self.assertIn("Turing", text)
        self.assertNotIn(
            "36.30",
            text,
            "ENVIRONMENT must keep volatile disk/free numbers in CURRENT_STATE.md",
        )

    def test_plan_marked_historical_with_banner_and_portfolio_links(self):
        text = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        self.assertIn("historical", text)
        self.assertIn("approved", text.lower())
        self.assertIn("portfolio", text.lower())
        self.assertIn("CURRENT_STATE.md", text)
        self.assertIn("docs/plans/README.md", text)
        # original specification preserved (approved baseline, not deleted)
        for anchor in ("M0", "M3", "5.3", "AUTO-CHECKPOINT"):
            self.assertIn(anchor, text)

    def test_task8_documents_are_link_clean(self):
        for path in INDEX_LINK_SOURCES:
            self.assertEqual([], check_markdown_links(path, ROOT), f"broken links: {path.name}")

    def test_task8_documents_avoid_bare_interpreter_commands(self):
        for path in INDEX_LINK_SOURCES:
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            issues = check_static_policy({path.name: text}, ROOT)
            live = [i for i in issues if "live process" in i]
            self.assertEqual([], live, f"live-process claims in {path.name}")
            # PLAN.md keeps its original specification verbatim, so its
            # historical prose ("python subset", "lang=python") stays as an
            # explicitly accepted exception; the newest docs stay clean.
            if path.name != "PLAN.md":
                self.assertEqual([], issues, f"static policy issues in {path.name}")

    def test_research_raw_directory_has_one_catalog_row(self):
        rows = _index_rows(RESEARCH_README, occurrence=1)
        self.assertEqual(len(rows), 1, "research/raw/ must have one directory catalog row")
        row = rows[0]
        self.assertEqual(len(row), 6)
        self.assertEqual(_clean_cell(row[0]), "research/raw/")
        self.assertTrue((ROOT / _clean_cell(row[0])).is_dir())
        self.assertTrue(_clean_cell(row[1]))
        self.assertIn(_clean_cell(row[2]), PLAN_STATUSES)
        self.assertTrue(_valid_owner(row[3]))
        self.assertTrue(_valid_verified(row[4]))
        self.assertTrue(_clean_cell(row[5]))

    def test_research_raw_store_exists_and_is_excluded_from_indexing(self):
        self.assertTrue((ROOT / "research" / "raw").is_dir())
        self.assertTrue((ROOT / "research" / "raw" / "laya").is_dir())
        self.assertTrue(any((ROOT / "research" / "raw").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
