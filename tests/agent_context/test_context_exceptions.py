"""Exact-match historical exception registry contract.

Every accepted exception is keyed by (path, line, kind, target). A changed,
shifted, new, or unlisted finding is never accepted, and any malformed,
missing, or unsafe registry fails closed.
"""

import tempfile
import unittest
from pathlib import Path

from scripts.check_agent_context import (
    apply_context_exceptions,
    load_context_exceptions,
)


_TABLE_RECORD = (
    "  - id: context-exception/tasks-60\n"
    "    status: accepted_historical\n"
    "    path: TASKS.md\n"
    "    line: 60\n"
    "    kind: malformed_table_row\n"
    "    reason: Legacy table formatting is preserved as historical evidence.\n"
    "    accepted_at: '2026-09-25'\n"
    "    decision_source: direct-user-instruction\n"
    "    evidence:\n"
    "      - docs/agent/reports/context-validation-2026-09-24.md\n"
)
_LINK_RECORD = (
    "  - id: context-exception/experiments-25\n"
    "    status: accepted_historical\n"
    "    path: research/EXPERIMENTS.md\n"
    "    line: 25\n"
    "    kind: broken_markdown_link\n"
    "    target: research/_template_experiment.md\n"
    "    reason: Historical report link is preserved as an accepted exception.\n"
    "    accepted_at: '2026-09-25'\n"
    "    decision_source: direct-user-instruction\n"
    "    evidence:\n"
    "      - docs/agent/reports/context-validation-2026-09-24.md\n"
)


class ContextExceptionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "state").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write_registry(self, text):
        evidence = self.root / "docs/agent/reports/context-validation-2026-09-24.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("# validation evidence\n", encoding="utf-8")
        (self.root / "state/context_exceptions.yaml").write_text(
            text, encoding="utf-8"
        )

    def write_registry_text(self, exceptions):
        self.write_registry("schema_version: 1\nexceptions:\n" + exceptions)

    def write_registry_without_evidence(self, exceptions):
        """Write the registry only, so declared evidence stays missing."""
        (self.root / "state/context_exceptions.yaml").write_text(
            "schema_version: 1\nexceptions:\n" + exceptions, encoding="utf-8"
        )

    def load(self, exceptions):
        self.write_registry_text(exceptions)
        return load_context_exceptions(self.root)

    def test_exact_table_exception_is_accepted(self):
        exceptions, issues = self.load(_TABLE_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["malformed table row: TASKS.md:60"], exceptions
        )
        self.assertEqual([], remaining)
        self.assertEqual(
            ["ACCEPTED HISTORICAL: TASKS.md:60 (malformed_table_row)"],
            accepted,
        )

    def test_exact_broken_link_exception_is_accepted(self):
        exceptions, issues = self.load(_LINK_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            [
                "broken markdown link: research/EXPERIMENTS.md:25: "
                "research/_template_experiment.md"
            ],
            exceptions,
        )
        self.assertEqual([], remaining)
        self.assertEqual(
            [
                "ACCEPTED HISTORICAL: research/EXPERIMENTS.md:25 "
                "(broken_markdown_link: research/_template_experiment.md)"
            ],
            accepted,
        )

    def test_changed_line_is_not_accepted(self):
        exceptions, issues = self.load(_TABLE_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["malformed table row: TASKS.md:61"], exceptions
        )
        self.assertEqual(["malformed table row: TASKS.md:61"], remaining)
        self.assertEqual([], accepted)

    def test_changed_target_is_not_accepted(self):
        exceptions, issues = self.load(_LINK_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            [
                "broken markdown link: research/EXPERIMENTS.md:25: "
                "research/OTHER.md"
            ],
            exceptions
        )
        self.assertEqual(
            [
                "broken markdown link: research/EXPERIMENTS.md:25: "
                "research/OTHER.md"
            ],
            remaining,
        )
        self.assertEqual([], accepted)

    def test_new_unlisted_finding_is_not_accepted(self):
        exceptions, issues = self.load(_TABLE_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["malformed table row: TASKS.md:999"], exceptions
        )
        self.assertEqual(["malformed table row: TASKS.md:999"], remaining)
        self.assertEqual([], accepted)

    def test_unparsable_finding_is_not_accepted(self):
        exceptions, issues = self.load(_TABLE_RECORD)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["some future finding format: TASKS.md:60"], exceptions
        )
        self.assertEqual(["some future finding format: TASKS.md:60"], remaining)
        self.assertEqual([], accepted)

    def test_all_four_initial_findings_match_exact_keys(self):
        exceptions = {
            ("TASKS.md", 60, "malformed_table_row", None): {},
            ("TASKS.md", 196, "malformed_table_row", None): {},
            (
                "research/EXPERIMENTS.md",
                25,
                "broken_markdown_link",
                "research/_template_experiment.md",
            ): {},
            (
                "research/EXPERIMENTS.md",
                28,
                "broken_markdown_link",
                "research/WHAT_WORKS.md",
            ): {},
        }
        issues = [
            "malformed table row: TASKS.md:60",
            "malformed table row: TASKS.md:196",
            "broken markdown link: research/EXPERIMENTS.md:25: "
            "research/_template_experiment.md",
            "broken markdown link: research/EXPERIMENTS.md:28: research/WHAT_WORKS.md",
        ]
        remaining, accepted = apply_context_exceptions(issues, exceptions)
        self.assertEqual([], remaining)
        self.assertEqual(4, len(accepted))

    def test_real_repository_registry_loads_cleanly(self):
        root = Path(__file__).resolve().parents[2]
        exceptions, issues = load_context_exceptions(root)
        self.assertEqual([], issues)
        self.assertEqual(4, len(exceptions))

    def test_real_repository_registry_accepts_only_the_four_known_findings(self):
        root = Path(__file__).resolve().parents[2]
        exceptions, issues = load_context_exceptions(root)
        self.assertEqual([], issues)
        self.assertEqual(
            {
                ("TASKS.md", 60, "malformed_table_row", None),
                ("TASKS.md", 196, "malformed_table_row", None),
                (
                    "research/EXPERIMENTS.md",
                    25,
                    "broken_markdown_link",
                    "research/_template_experiment.md",
                ),
                (
                    "research/EXPERIMENTS.md",
                    28,
                    "broken_markdown_link",
                    "research/WHAT_WORKS.md",
                ),
            },
            set(exceptions),
        )
        remaining, _ = apply_context_exceptions(
            ["malformed table row: TASKS.md:61"], exceptions
        )
        self.assertEqual(["malformed table row: TASKS.md:61"], remaining)

    def test_malformed_registry_fails_closed(self):
        self.write_registry("schema_version: 1\nexceptions: {}\n")
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertTrue(issues)
        self.assertTrue(any("context exception" in issue for issue in issues))

    def test_missing_registry_fails_closed(self):
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertEqual(
            ["missing required context exception registry: state/context_exceptions.yaml"],
            issues,
        )

    def test_wrong_schema_version_fails_closed(self):
        self.write_registry("schema_version: 2\nexceptions: []\n")
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertTrue(any("schema_version" in issue for issue in issues))

    def test_unsafe_absolute_path_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/unsafe\n"
            "    status: accepted_historical\n"
            "    path: C:/outside.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: unsafe\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("unsafe" in issue for issue in issues))

    def test_unsafe_parent_traversal_path_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/unsafe-parent\n"
            "    status: accepted_historical\n"
            "    path: ../outside.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: unsafe\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("unsafe path" in issue for issue in issues))

    def test_unsafe_target_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/unsafe-target\n"
            "    status: accepted_historical\n"
            "    path: research/EXPERIMENTS.md\n"
            "    line: 25\n"
            "    kind: broken_markdown_link\n"
            "    target: /etc/passwd\n"
            "    reason: unsafe\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("unsafe target" in issue for issue in issues))

    def test_duplicate_id_is_rejected(self):
        exceptions, issues = self.load(_TABLE_RECORD + _TABLE_RECORD)
        self.assertEqual({}, exceptions)
        self.assertTrue(any("duplicate id" in issue for issue in issues))

    def test_duplicate_finding_key_is_rejected(self):
        record = _TABLE_RECORD.replace("context-exception/tasks-60", "context-exception/other")
        exceptions, issues = self.load(_TABLE_RECORD + record)
        self.assertEqual({}, exceptions)
        self.assertTrue(any("duplicate finding key" in issue for issue in issues))

    def test_missing_required_field_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/incomplete\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    kind: malformed_table_row\n"
            "    reason: missing line\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid line" in issue for issue in issues))

    def test_wrong_status_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/wrong-status\n"
            "    status: accepted\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: wrong status\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid status" in issue for issue in issues))

    def test_invalid_line_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bad-line\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 0\n"
            "    kind: malformed_table_row\n"
            "    reason: bad line\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid line" in issue for issue in issues))

    def test_bool_line_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bool-line\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: true\n"
            "    kind: malformed_table_row\n"
            "    reason: bool line\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid line" in issue for issue in issues))

    def test_invalid_accepted_at_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bad-date\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: bad date\n"
            "    accepted_at: '2026-13-45'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid accepted_at" in issue for issue in issues))

    def test_wrong_decision_source_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bad-source\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: bad source\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: agent-inference\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid decision_source" in issue for issue in issues))

    def test_unknown_field_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/extra\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: extra field\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
            "    severity: low\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("unknown field" in issue for issue in issues))

    def test_unknown_kind_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bad-kind\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: whatever\n"
            "    reason: bad kind\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid kind" in issue for issue in issues))

    def test_target_on_table_row_kind_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/table-target\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    target: research/OTHER.md\n"
            "    reason: target not allowed\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("target is not allowed" in issue for issue in issues))

    def test_missing_evidence_artifact_is_rejected(self):
        # No evidence artifact is written, so the record must fail closed.
        self.write_registry_without_evidence(
            "  - id: context-exception/missing-evidence\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: missing evidence\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence:\n"
            "      - docs/agent/reports/context-validation-2026-09-24.md\n"
        )
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertTrue(any("missing evidence path" in issue for issue in issues))

    def test_unsafe_evidence_path_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/unsafe-evidence\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: unsafe evidence\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence:\n"
            "      - ../../outside.md\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid local repo-relative evidence path" in issue for issue in issues))

    def test_invalid_reason_is_rejected(self):
        exceptions, issues = self.load(
            "  - id: context-exception/bad-reason\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 1\n"
            "    kind: malformed_table_row\n"
            "    reason: token=abcdefghijklmnopqrstuvwx\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence: []\n"
        )
        self.assertEqual({}, exceptions)
        self.assertTrue(any("invalid reason" in issue for issue in issues))

    def test_empty_registry_is_valid(self):
        self.write_registry("schema_version: 1\nexceptions: []\n")
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertEqual([], issues)

if __name__ == "__main__":
    unittest.main()
