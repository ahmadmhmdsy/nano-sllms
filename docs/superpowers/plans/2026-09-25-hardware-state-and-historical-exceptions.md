# Historical Exceptions and Selective Hardware State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four known historical context findings explicitly accepted without rewriting their source documents, and add a selective, read-only hardware-state command for CPU, RAM, disk, network, GPU, dedicated VRAM, shared GPU memory, and per-process GPU state.

**Architecture:** 'state/context_exceptions.yaml' is a narrow, exact-match exception registry consumed by the existing linter after findings are generated. 'scripts/hardware_state.py' is an on-demand JSON-only probe: collectors return typed sections, missing sources remain unavailable, and no collector writes repository state or starts/stops workloads. Documentation and the existing validation report describe the local-measurement boundary separately from the unavailable global audit.

**Tech Stack:** Python 3.12 standard library; existing repository '.venv'; PyYAML already used by the linter; 'unittest'; Windows 'nvidia-smi', PowerShell performance counters, and 'Get-CimInstance' as optional read-only sources; UTF-8 Markdown and YAML.

---

## Execution constraints

- Work in the existing dirty checkout at 'E:\python_projects\nano_SLMs'.
- Preserve every pre-existing modified, deleted, staged, and untracked path.
- Never run 'git clean', 'git reset', 'git stash', bulk deletion, history rewriting, commit, merge, or push.
- Do not edit 'TASKS.md' or 'research/EXPERIMENTS.md'; they remain historical source documents.
- Use '& .\.venv\Scripts\python.exe' for every Python command.
- Do not run a live GPU probe during implementation validation. All hardware command calls in tests are faked or mocked.
- Do not launch training, inference, servers, downloads, network requests, or process-management operations.
- Do not hand-edit 'CURRENT_STATE.md' or 'state/current.json'.
- Keep the global process/network/adoption audit explicitly 'BLOCKED'; the new local probe does not satisfy that audit.
- The plan intentionally has no commit steps because the active worktree-preservation boundary forbids staging and committing.

## Planned file map

### Create

- 'state/context_exceptions.yaml' — four exact, user-accepted historical findings.
- 'scripts/hardware_state.py' — read-only collectors, parsers, JSON envelope, and CLI.
- 'tests/agent_context/test_context_exceptions.py' — registry validation and exact-match tests.
- 'tests/agent_context/test_hardware_state.py' — parser, collector, CLI, and safety tests.
- 'docs/agent/HARDWARE_STATE.md' — agent-facing hardware-state usage and limitations.

### Modify

- 'scripts/check_agent_context.py' — load, validate, and apply exact historical exceptions.
- 'tests/agent_context/test_check_agent_context.py' — linter integration and policy-fixture coverage.
- 'docs/agent/SCHEMA.md' — document the exception-registry schema.
- 'AGENTS.md' — add a hardware-state task-routed read pointer.
- 'docs/agent/OPERATIONS.md' — add the user-opened local hardware-probe boundary.
- 'docs/agent/reports/context-validation-2026-09-24.md' — record the new validation evidence and remaining 'BLOCKED' audit boundary.

Do not modify the two historical source ledgers named in the exception registry.

---

## Task 1: Define and test the exact historical-exception contract

**Files:**
- Create: 'tests/agent_context/test_context_exceptions.py'
- Create: 'state/context_exceptions.yaml'
- Modify: 'scripts/check_agent_context.py:38-45, 715-774' (constants, YAML loading, and registry helpers)

### Step 1: Write failing registry and matcher tests

Create the test module with a temporary repository helper and these concrete cases:

~~~python
import tempfile
import unittest
from pathlib import Path

from scripts.check_agent_context import (
    apply_context_exceptions,
    load_context_exceptions,
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

    def test_exact_table_exception_is_accepted(self):
        self.write_registry(
            "schema_version: 1\n"
            "exceptions:\n"
            "  - id: context-exception/tasks-60\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 60\n"
            "    kind: malformed_table_row\n"
            "    reason: Legacy table formatting is preserved.\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence:\n"
            "      - docs/agent/reports/context-validation-2026-09-24.md\n"
        )
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["malformed table row: TASKS.md:60"], exceptions
        )
        self.assertEqual([], remaining)
        self.assertEqual(
            ["ACCEPTED HISTORICAL: TASKS.md:60 (malformed_table_row)"],
            accepted,
        )

    def test_changed_line_is_not_accepted(self):
        self.write_registry(
            "schema_version: 1\n"
            "exceptions:\n"
            "  - id: context-exception/tasks-60\n"
            "    status: accepted_historical\n"
            "    path: TASKS.md\n"
            "    line: 60\n"
            "    kind: malformed_table_row\n"
            "    reason: Legacy table formatting is preserved.\n"
            "    accepted_at: '2026-09-25'\n"
            "    decision_source: direct-user-instruction\n"
            "    evidence:\n"
            "      - docs/agent/reports/context-validation-2026-09-24.md\n"
        )
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual([], issues)
        remaining, accepted = apply_context_exceptions(
            ["malformed table row: TASKS.md:61"], exceptions
        )
        self.assertEqual(["malformed table row: TASKS.md:61"], remaining)
        self.assertEqual([], accepted)

    def test_all_four_initial_findings_match_exact_keys(self):
        exceptions = {
            ("TASKS.md", 60, "malformed_table_row", None): {},
            ("TASKS.md", 196, "malformed_table_row", None): {},
            ("research/EXPERIMENTS.md", 25, "broken_markdown_link", "research/_template_experiment.md"): {},
            ("research/EXPERIMENTS.md", 28, "broken_markdown_link", "research/WHAT_WORKS.md"): {},
        }
        issues = [
            "malformed table row: TASKS.md:60",
            "malformed table row: TASKS.md:196",
            "broken markdown link: research/EXPERIMENTS.md:25: research/_template_experiment.md",
            "broken markdown link: research/EXPERIMENTS.md:28: research/WHAT_WORKS.md",
        ]
        remaining, accepted = apply_context_exceptions(issues, exceptions)
        self.assertEqual([], remaining)
        self.assertEqual(4, len(accepted))

    def test_malformed_registry_fails_closed(self):
        self.write_registry("schema_version: 1\nexceptions: {}\n")
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertTrue(issues)
        self.assertTrue(any("context exception" in issue for issue in issues))

    def test_unsafe_absolute_path_is_rejected(self):
        self.write_registry(
            "schema_version: 1\n"
            "exceptions:\n"
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
        exceptions, issues = load_context_exceptions(self.root)
        self.assertEqual({}, exceptions)
        self.assertTrue(any("unsafe" in issue for issue in issues))
~~~

Add equivalent tests for a broken-link target, duplicate records, a missing required field, a wrong status, an invalid line, and a changed target. The broken-link test must use the exact target string because target changes are not accepted.

### Step 2: Run the new tests and verify they fail

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_exceptions -v
~~~

Expected: import or symbol failures for 'load_context_exceptions' and 'apply_context_exceptions'.

### Step 3: Implement the registry parser and exact matcher

Add these type aliases and regular expressions near the existing linter constants:

~~~python
from datetime import date

FindingKey = tuple[str, int, str, str | None]

_CONTEXT_EXCEPTION_KINDS = frozenset(
    {"malformed_table_row", "broken_markdown_link"}
)
_MALFORMED_TABLE_ISSUE = re.compile(
    r"^malformed table row: (?P<path>[^:]+):(?P<line>[0-9]+)$"
)
_BROKEN_LINK_ISSUE = re.compile(
    r"^broken markdown link: (?P<path>[^:]+):(?P<line>[0-9]+): (?P<target>.+)$"
)
~~~

~~~python
def _context_exception_key(issue: str) -> FindingKey | None:
    table = _MALFORMED_TABLE_ISSUE.fullmatch(issue)
    if table is not None:
        return (table.group("path"), int(table.group("line")), "malformed_table_row", None)
    link = _BROKEN_LINK_ISSUE.fullmatch(issue)
    if link is not None:
        return (link.group("path"), int(link.group("line")), "broken_markdown_link", link.group("target"))
    return None


def _registry_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 240:
        return False
    if _is_absolute_reference(value) or "\\" in value or ":" in value:
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if _WIN and any(part.endswith((".", " ")) for part in parts):
        return False
    return _safe_diagnostic(value) == value


def _registry_text(value: Any, label: str, issues: list[str], index: int) -> bool:
    if not isinstance(value, str) or not value or len(value) > 240:
        issues.append(f"context exception[{index}]: invalid {label}")
        return False
    if _safe_diagnostic(value) != value:
        issues.append(f"context exception[{index}]: invalid {label}")
        return False
    return True


def load_context_exceptions(root: Path) -> tuple[dict[FindingKey, dict[str, Any]], list[str]]:
    issues: list[str] = []
    try:
        root = _absolute_lexical(root)
    except UnsafeContextError:
        return {}, ["context exception registry root is unsafe"]
    data = _load_yaml(root / "state" / "context_exceptions.yaml", "state/context_exceptions.yaml", issues)
    if data is _YAML_MISSING:
        return {}, ["missing required context exception registry: state/context_exceptions.yaml"]
    if issues:
        return {}, issues
    if not isinstance(data, dict) or data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        return {}, ["context exception registry must contain schema_version: 1"]
    records = data.get("exceptions")
    if not isinstance(records, list):
        return {}, ["context exception registry exceptions must be a list"]

    parsed: dict[FindingKey, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    allowed_fields = {
        "id", "status", "path", "line", "kind", "target", "reason",
        "accepted_at", "decision_source", "evidence",
    }
    for index, record in enumerate(records):
        local: list[str] = []
        if not isinstance(record, dict):
            issues.append(f"context exception[{index}]: record must be a mapping")
            continue
        if set(record) - allowed_fields:
            local.append(f"context exception[{index}]: unknown field")
        identifier = record.get("id")
        if not isinstance(identifier, str) or _NAMESPACED_ID.fullmatch(identifier) is None:
            local.append(f"context exception[{index}]: invalid id")
        elif identifier in seen_ids:
            local.append(f"context exception[{index}]: duplicate id")
        else:
            seen_ids.add(identifier)
        if record.get("status") != "accepted_historical":
            local.append(f"context exception[{index}]: invalid status")
        finding_path = record.get("path")
        if not _registry_relative_path(finding_path):
            local.append(f"context exception[{index}]: invalid path")
        line = record.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            local.append(f"context exception[{index}]: invalid line")
        kind = record.get("kind")
        if kind not in _CONTEXT_EXCEPTION_KINDS:
            local.append(f"context exception[{index}]: invalid kind")
        target = record.get("target")
        if kind == "broken_markdown_link":
            if not _registry_relative_path(target):
                local.append(f"context exception[{index}]: invalid target")
        elif "target" in record:
            local.append(f"context exception[{index}]: target is not allowed for malformed_table_row")
        if not _registry_text(record.get("reason"), "reason", local, index):
            pass
        accepted_at = record.get("accepted_at")
        if not isinstance(accepted_at, str):
            local.append(f"context exception[{index}]: invalid accepted_at")
        else:
            try:
                if date.fromisoformat(accepted_at).isoformat() != accepted_at:
                    raise ValueError
            except ValueError:
                local.append(f"context exception[{index}]: invalid accepted_at")
        if record.get("decision_source") != "direct-user-instruction":
            local.append(f"context exception[{index}]: invalid decision_source")
        evidence = record.get("evidence")
        if not isinstance(evidence, list):
            local.append(f"context exception[{index}]: evidence must be a list")
        else:
            evidence_issues: list[str] = []
            _validate_evidence(record, "context exception", index, root, evidence_issues)
            local.extend(evidence_issues)
        if local:
            issues.extend(local)
            continue
        key: FindingKey = (finding_path, line, kind, target if kind == "broken_markdown_link" else None)
        if key in parsed:
            issues.append(f"context exception[{index}]: duplicate finding key")
            continue
        parsed[key] = record
    if issues:
        return {}, issues
    return parsed, []
~~~

Add:

~~~python
def apply_context_exceptions(
    issues: list[str],
    exceptions: dict[FindingKey, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    remaining: list[str] = []
    accepted: list[str] = []
    for issue in issues:
        key = _context_exception_key(issue)
        if key is None or key not in exceptions:
            remaining.append(issue)
            continue
        path, line, kind, target = key
        suffix = f": {target}" if target is not None else ""
        accepted.append(
            f"ACCEPTED HISTORICAL: {path}:{line} ({kind}{suffix})"
        )
    return remaining, accepted
~~~

The matcher must use the exact normalized path and target strings produced by the linter. It must not perform substring matching.

### Step 4: Add the four registry records

Create 'state/context_exceptions.yaml' with schema version 1 and these exact records:

~~~yaml
schema_version: 1
exceptions:
  - id: context-exception/tasks-60
    status: accepted_historical
    path: TASKS.md
    line: 60
    kind: malformed_table_row
    reason: Legacy table formatting is preserved as historical evidence.
    accepted_at: "2026-09-25"
    decision_source: direct-user-instruction
    evidence:
      - docs/agent/reports/context-validation-2026-09-24.md
  - id: context-exception/tasks-196
    status: accepted_historical
    path: TASKS.md
    line: 196
    kind: malformed_table_row
    reason: Legacy table formatting is preserved as historical evidence.
    accepted_at: "2026-09-25"
    decision_source: direct-user-instruction
    evidence:
      - docs/agent/reports/context-validation-2026-09-24.md
  - id: context-exception/experiments-25
    status: accepted_historical
    path: research/EXPERIMENTS.md
    line: 25
    kind: broken_markdown_link
    target: research/_template_experiment.md
    reason: Historical report link is preserved as an accepted exception.
    accepted_at: "2026-09-25"
    decision_source: direct-user-instruction
    evidence:
      - docs/agent/reports/context-validation-2026-09-24.md
  - id: context-exception/experiments-28
    status: accepted_historical
    path: research/EXPERIMENTS.md
    line: 28
    kind: broken_markdown_link
    target: research/WHAT_WORKS.md
    reason: Historical report link is preserved as an accepted exception.
    accepted_at: "2026-09-25"
    decision_source: direct-user-instruction
    evidence:
      - docs/agent/reports/context-validation-2026-09-24.md
~~~

### Step 5: Run the focused registry tests

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_exceptions -v
~~~

Expected: all registry/matcher tests pass, including malformed and unsafe-registry cases.

---

## Task 2: Integrate accepted exceptions into the linter

**Files:**
- Modify: 'scripts/check_agent_context.py:777-914'
- Modify: 'tests/agent_context/test_check_agent_context.py:844-919'

### Step 1: Add failing linter integration tests

Add a helper to the existing 'LintTreeHelperTestCase':

~~~python
def _write_exception_registry(self, text):
    self._write("state/context_exceptions.yaml", text)

def _empty_exception_registry(self):
    self._write_exception_registry(
        "schema_version: 1\nexceptions: []\n"
    )
~~~

Update '_write_required_policy_files()' to call '_empty_exception_registry()' so existing policy-mode fixtures have a valid registry. Add tests:

~~~python
def test_default_policy_accepts_exact_historical_findings(self):
    self._write_clean_records()
    self._write_required_policy_files()
    self._write("TASKS.md", "| broken || row |\n")
    self._write(
        "research/EXPERIMENTS.md",
        "[missing](research/_template_experiment.md)\n",
    )
    self._write_exception_registry(
        "schema_version: 1\n"
        "exceptions:\n"
        "  - id: context-exception/tasks-2\n"
        "    status: accepted_historical\n"
        "    path: TASKS.md\n"
        "    line: 1\n"
        "    kind: malformed_table_row\n"
        "    reason: accepted fixture\n"
        "    accepted_at: '2026-09-25'\n"
        "    decision_source: direct-user-instruction\n"
        "    evidence: []\n"
        "  - id: context-exception/link-1\n"
        "    status: accepted_historical\n"
        "    path: research/EXPERIMENTS.md\n"
        "    line: 1\n"
        "    kind: broken_markdown_link\n"
        "    target: research/_template_experiment.md\n"
        "    reason: accepted fixture\n"
        "    accepted_at: '2026-09-25'\n"
        "    decision_source: direct-user-instruction\n"
        "    evidence: []\n"
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
        "schema_version: 1\n"
        "exceptions:\n"
        "  - id: context-exception/tasks-1\n"
        "    status: accepted_historical\n"
        "    path: TASKS.md\n"
        "    line: 2\n"
        "    kind: malformed_table_row\n"
        "    reason: wrong line fixture\n"
        "    accepted_at: '2026-09-25'\n"
        "    decision_source: direct-user-instruction\n"
        "    evidence: []\n"
    )
    issues = lint_tree(self.root)
    self.assertTrue(any("malformed table row: TASKS.md:1" in issue for issue in issues))

def test_record_only_mode_does_not_apply_exceptions(self):
    self._write_clean_records()
    self._write_required_policy_files()
    self._write("TASKS.md", "| broken || row |\n")
    self._write_exception_registry(
        "schema_version: 1\n"
        "exceptions:\n"
        "  - id: context-exception/tasks-1\n"
        "    status: accepted_historical\n"
        "    path: TASKS.md\n"
        "    line: 1\n"
        "    kind: malformed_table_row\n"
        "    reason: record-only fixture\n"
        "    accepted_at: '2026-09-25'\n"
        "    decision_source: direct-user-instruction\n"
        "    evidence: []\n"
    )
    issues = lint_tree(self.root, include_policy=False)
    self.assertTrue(any("malformed table row: TASKS.md:1" in issue for issue in issues))
~~~

Add a CLI assertion that accepted lines are printed with the 'ACCEPTED HISTORICAL:' prefix and that the process exits 0 when no unaccepted issue remains.

### Step 2: Run the integration tests and verify they fail

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -v
~~~

Expected: failures because 'lint_tree' does not yet accept 'accepted_exceptions' and still returns the two fixture findings.

### Step 3: Extend the public linter entry point without breaking callers

Add an optional keyword-only sink:

~~~python
def lint_tree(
    root: Path,
    *,
    include_policy: bool = True,
    evidence_root: Path | None = None,
    accepted_exceptions: list[str] | None = None,
) -> list[str]:
    try:
        return _lint_tree(
            root,
            include_policy=include_policy,
            evidence_root=evidence_root,
            accepted_exceptions=accepted_exceptions,
        )
    except Exception:
        return ["context validation failed unexpectedly: " + _REDACTED_DIAGNOSTIC]
~~~

Change the internal entry point to:

~~~python
def _lint_tree(
    root: Path,
    *,
    include_policy: bool,
    evidence_root: Path | None,
    accepted_exceptions: list[str] | None = None,
) -> list[str]:
~~~

Retain the existing read-only preflight, record, Markdown-link, and policy checks in that function. After those checks have produced their ordinary issues, add:

~~~python
if include_policy:
    exceptions, exception_issues = load_context_exceptions(root)
    issues.extend(exception_issues)
    if not exception_issues:
        issues, accepted = apply_context_exceptions(issues, exceptions)
        if accepted_exceptions is not None:
            accepted_exceptions.extend(accepted)
return issues
~~~

When 'include_policy=False', do not load or apply the registry; record-only mode must continue to expose the raw historical findings. A missing registry in normal policy mode must add the controlled issue 'missing required context exception registry: state/context_exceptions.yaml' and must not apply any partial records.

### Step 4: Update the CLI to show accepted records

In 'main', create an 'accepted' list, pass it to 'lint_tree', print accepted entries before issues using '_safe_print', and preserve the existing '--max-issues' bound for both accepted and ordinary output:

~~~python
accepted: list[str] = []
issues = lint_tree(
    root,
    include_policy=not args.no_policy,
    accepted_exceptions=accepted,
)
combined = [("accepted", entry) for entry in accepted]
combined.extend(("issue", issue) for issue in issues)
for kind, entry in combined[: args.max_issues]:
    _safe_print(entry if kind == "accepted" else f"ISSUE: {entry}")
if len(combined) > args.max_issues:
    _safe_print(
        f"ISSUE: additional issues omitted: {len(combined) - args.max_issues}"
    )
return 1 if issues else 0
~~~

Keep the existing broad exception boundary around 'lint_tree' and the existing ASCII-safe '_safe_print'. Do not print raw registry values or exception text.

### Step 5: Run focused linter and full context tests

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_exceptions tests.agent_context.test_check_agent_context -v
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t . -q
~~~

Expected: all tests pass. The live repository linter may still report the four findings until the actual registry is loaded in the final validation step; test fixtures must be self-contained.

---

## Task 3: Build and test hardware-state parsers

**Files:**
- Create: 'scripts/hardware_state.py'
- Create: 'tests/agent_context/test_hardware_state.py'

### Step 1: Write failing parser tests

Use these exact CSV fixtures:

~~~python
GPU_CSV = """0, GPU-0000, NVIDIA RTX TEST, 999.9, 8192, 2048, 6144, 37, 52, 80.5
"""

PROCESS_CSV = """1234, C:\\Python\\python.exe, 512
5678, C:\\Python\\worker.exe, 256
"""

SHARED_CSV = """1234\t268435456
5678\t134217728
"""
~~~

Add tests for:

~~~python
from scripts.hardware_state import (
    merge_gpu_processes,
    parse_nvidia_gpu_csv,
    parse_nvidia_process_csv,
    parse_shared_process_csv,
)

def test_gpu_csv_keeps_dedicated_vram_separate(self):
    gpus = parse_nvidia_gpu_csv(GPU_CSV)
    self.assertEqual(1, len(gpus))
    self.assertEqual(8192 * 1024 * 1024, gpus[0]["dedicated_vram_total_bytes"])
    self.assertEqual(2048 * 1024 * 1024, gpus[0]["dedicated_vram_used_bytes"])
    self.assertIsNone(gpus[0]["shared_memory_bytes"])

def test_process_csv_and_shared_counter_join_by_pid(self):
    dedicated = parse_nvidia_process_csv(PROCESS_CSV)
    shared = parse_shared_process_csv(SHARED_CSV)
    processes = merge_gpu_processes(dedicated, shared)
    by_pid = {row["pid"]: row for row in processes}
    self.assertEqual(512 * 1024 * 1024, by_pid[1234]["dedicated_vram_bytes"])
    self.assertEqual(268435456, by_pid[1234]["shared_memory_bytes"])
    self.assertNotEqual(
        by_pid[1234]["dedicated_vram_bytes"],
        by_pid[1234]["shared_memory_bytes"],
    )

def test_unmatched_pid_keeps_missing_side_null(self):
    processes = merge_gpu_processes(
        parse_nvidia_process_csv(PROCESS_CSV),
        {1234: 268435456},
    )
    by_pid = {row["pid"]: row for row in processes}
    self.assertEqual(268435456, by_pid[1234]["shared_memory_bytes"])
    self.assertIsNone(by_pid[5678]["shared_memory_bytes"])
~~~

Add malformed-row, negative-value, duplicate-PID, and oversized-process-name tests. The parser must return controlled empty/partial results or raise a parser-specific exception that the collector converts to an unavailable section; it must never emit a traceback.

### Step 2: Run parser tests and verify they fail

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state.HardwareParserTests -v
~~~

Expected: import failures for the new parser functions.

### Step 3: Implement safe command results and parsers

At the top of 'scripts/hardware_state.py', define:

~~~python
import argparse
import csv
import io
import json
import ntpath
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_BYTES_PER_MIB = 1024 * 1024
_PROCESS_SECRET = re.compile(
    r"(?i)(?:AKIA|ASIA)[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|password|secret|token|api[_-]?key"
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""
    error: str | None = None
~~~

Implement 'parse_nvidia_gpu_csv(text: str) -> list[dict[str, Any]]' with the exact field order used by this query:

~~~python
def _number(value: str, *, integer: bool = False, scale: int = 1) -> int | float | None:
    text = value.strip()
    if not text or text.lower() in {"n/a", "na", "[not supported]", "not supported"}:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError("invalid numeric hardware field") from exc
    if number < 0:
        raise ValueError("negative hardware field")
    if integer:
        return int(number * scale)
    return number


def _bounded_process_name(value: str) -> str:
    name = ntpath.basename(value.replace("/", "\\")).strip()
    if (
        not name
        or _PROCESS_SECRET.search(name)
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        return "<redacted-process-name>"
    return name[:160]


def parse_cuda_version(text: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", text)
    return match.group(1) if match is not None else None


def parse_nvidia_gpu_csv(text: str) -> list[dict[str, Any]]:
    result = []
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 10:
            raise ValueError("invalid NVIDIA GPU row")
        index = _number(row[0], integer=True)
        total = _number(row[4], integer=True, scale=_BYTES_PER_MIB)
        used = _number(row[5], integer=True, scale=_BYTES_PER_MIB)
        free = _number(row[6], integer=True, scale=_BYTES_PER_MIB)
        result.append({
            "index": index,
            "uuid": row[1].strip()[:160],
            "name": row[2].strip()[:160],
            "driver_version": row[3].strip()[:160],
            "dedicated_vram_total_bytes": total,
            "dedicated_vram_used_bytes": used,
            "dedicated_vram_available_bytes": free,
            "dedicated_vram_source": "nvidia-smi",
            "utilization_percent": _number(row[7]),
            "temperature_c": _number(row[8]),
            "power_w": _number(row[9]),
            "cuda_version": None,
            "shared_memory_bytes": None,
            "shared_memory_source": None,
        })
    return result


def parse_nvidia_process_csv(text: str) -> list[dict[str, Any]]:
    result = []
    seen: set[int] = set()
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 3:
            raise ValueError("invalid NVIDIA process row")
        pid = _number(row[0], integer=True)
        if pid is None or pid <= 0 or pid in seen:
            raise ValueError("invalid NVIDIA process pid")
        seen.add(pid)
        used = _number(row[2], integer=True, scale=_BYTES_PER_MIB)
        result.append({
            "pid": pid,
            "gpu_uuid": None,
            "process_name": _bounded_process_name(row[1]),
            "dedicated_vram_bytes": used,
            "dedicated_vram_source": "nvidia-smi",
            "shared_memory_bytes": None,
            "shared_memory_source": None,
            "gpu_utilization_percent": None,
        })
    return result


def parse_shared_process_csv(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid shared-memory row")
        pid = int(fields[0].strip())
        value = int(fields[1].strip())
        if pid <= 0 or value < 0 or pid in result:
            raise ValueError("invalid shared-memory value")
        result[pid] = value
    return result


def merge_gpu_processes(
    dedicated: list[dict[str, Any]],
    shared: dict[int, int],
) -> list[dict[str, Any]]:
    by_pid = {row["pid"]: dict(row) for row in dedicated}
    for pid, value in shared.items():
        if pid not in by_pid:
            by_pid[pid] = {
                "pid": pid,
                "gpu_uuid": None,
                "process_name": "<unavailable>",
                "dedicated_vram_bytes": None,
                "dedicated_vram_source": None,
                "shared_memory_bytes": value,
                "shared_memory_source": "GPU Process Memory Shared Usage",
                "gpu_utilization_percent": None,
            }
        else:
            by_pid[pid]["shared_memory_bytes"] = value
            by_pid[pid]["shared_memory_source"] = "GPU Process Memory Shared Usage"
    return [by_pid[pid] for pid in sorted(by_pid)]
~~~

### Step 4: Run parser tests and verify they pass

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state.HardwareParserTests -v
~~~

Expected: all parser tests pass, including explicit separate-byte assertions.

---

## Task 4: Implement selective collectors and the JSON CLI

**Files:**
- Modify: 'scripts/hardware_state.py'
- Modify: 'tests/agent_context/test_hardware_state.py'

### Step 1: Add failing collector and CLI tests

Create a fake runner after importing the public test surface:

~~~python
from scripts.hardware_state import (
    CommandResult,
    build_payload,
    collect_cpu,
    collect_disk,
    collect_hardware_state,
    collect_memory,
    collect_network,
    main,
    parse_cuda_version,
    parse_disk_json,
    parse_memory_json,
    parse_network_json,
)


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.commands = []

    def __call__(self, command):
        self.commands.append(tuple(command))
        key = tuple(command)
        return self.responses.get(key, CommandResult(1, "", "unavailable"))
~~~

Add tests for:

- default overview does not request 'gpu-processes';
- '--section cpu --section memory' requests only those sections;
- '--all' expands to every section;
- unavailable command output produces a valid section with 'status: unavailable';
- partial GPU data produces 'status: partial';
- the process list keeps dedicated and shared values separate;
- disk JSON reports multiple local volumes and network JSON reports bounded addresses without DNS resolution;
- the optional CUDA header parser leaves 'cuda_version' null when absent;
- the CLI returns 2 for unavailable requested sections and 0 for all available sections;
- invalid arguments return 1 with no probe call; main tests call 'main(argv, runner=FakeRunner(...))' so the default subprocess path is never reached;
- no file under a temporary repository is created or changed;
- a simulated command timeout does not emit raw stderr or a traceback.

Use a temporary directory and pass FakeRunner to every collector; no test may call the default runner, read /proc, read /sys, query WMI, or invoke a live hardware command. Patch only the output stream for CLI tests, never repository state.

### Step 2: Run collector/CLI tests and verify they fail

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state -v
~~~

Expected: failures for missing 'collect_hardware_state', 'main', and collector functions.

### Step 3: Implement the read-only command runner

Implement:

~~~python
def run_command(command: Sequence[str]) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(127, "", error="command unavailable")
    except subprocess.TimeoutExpired:
        return CommandResult(124, "", error="command timed out")
    except OSError:
        return CommandResult(1, "", error="command failed")
    return CommandResult(
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
~~~

Never use 'shell=True', never include environment variables in output, and never return raw 'stderr' to the JSON document.

### Step 4: Implement individual collectors

Use these function signatures:

~~~python
CommandRunner = Callable[[Sequence[str]], CommandResult]


def _section(data: Any, status: str, diagnostic: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "data": data}
    if diagnostic is not None:
        result["diagnostics"] = [diagnostic]
    return result


def collect_cpu(runner: CommandRunner = run_command) -> dict[str, Any]:
    if os.name != "nt":
        return _section({}, "unavailable", "cpu source unsupported")
    result = runner([
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue",
    ])
    if result.returncode != 0:
        return _section({}, "unavailable", "cpu source unavailable")
    return _section(parse_cpu_counter(result.stdout), "available")


def parse_memory_json(text: str) -> dict[str, Any]:
    payload = json.loads(text or "{}")
    if not isinstance(payload, dict):
        raise ValueError("memory source is not a mapping")
    total_kb = _number(str(payload.get("TotalVisibleMemorySize", "")), integer=True)
    free_kb = _number(str(payload.get("FreePhysicalMemory", "")), integer=True)
    total = None if total_kb is None else total_kb * 1024
    available = None if free_kb is None else free_kb * 1024
    used = None if total is None or available is None else total - available
    return {
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "used_percent": None if total in (None, 0) or used is None else round(100.0 * used / total, 2),
        "source": "Get-CimInstance Win32_OperatingSystem",
    }


def collect_memory(runner: CommandRunner = run_command) -> dict[str, Any]:
    if os.name != "nt":
        return _section({}, "unavailable", "memory source unsupported")
    result = runner([
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress",
    ])
    if result.returncode != 0:
        return _section({}, "unavailable", "memory source unavailable")
    try:
        data = parse_memory_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section({}, "unavailable", "memory data invalid")
    status = "available" if data["total_bytes"] is not None else "unavailable"
    return _section(data, status, None if status == "available" else "memory data empty")


def collect_gpu(runner: CommandRunner = run_command) -> dict[str, Any]:
    result = runner([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return _section([], "unavailable", "gpu source unavailable")
    try:
        data = parse_nvidia_gpu_csv(result.stdout)
    except ValueError:
        return _section([], "unavailable", "gpu data invalid")
    header = runner(["nvidia-smi"])
    cuda_version = parse_cuda_version(header.stdout) if header.returncode == 0 else None
    for row in data:
        row["cuda_version"] = cuda_version
    return _section(data, "available" if data else "unavailable", None if data else "gpu data empty")


def collect_gpu_processes(runner: CommandRunner = run_command) -> dict[str, Any]:
    result = runner([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return _section([], "unavailable", "gpu process source unavailable")
    try:
        dedicated = parse_nvidia_process_csv(result.stdout)
    except ValueError:
        return _section([], "unavailable", "gpu process data invalid")
    shared: dict[int, int] = {}
    shared_status = None
    if os.name == "nt":
        shared_result = runner([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { $v=$_.InstanceName; $processId=[int]($v -replace 'pid_','' -replace '_.*$',''); '{0}{1}{2}' -f $processId, [char]9, [uint64]$_.CookedValue }",
        ])
        if shared_result.returncode != 0:
            shared_status = "shared memory source unavailable"
        else:
            try:
                shared = parse_shared_process_csv(shared_result.stdout)
            except ValueError:
                shared_status = "shared memory data invalid"
    data = merge_gpu_processes(dedicated, shared)
    status = "partial" if shared_status is not None else "available"
    return _section(data, status, shared_status)


def parse_disk_json(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("disk source is not a list")
    result = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("invalid disk record")
        mount = item.get("DeviceID", item.get("mount_point"))
        total = _number(str(item.get("Size", "")), integer=True)
        available = _number(str(item.get("FreeSpace", "")), integer=True)
        if not isinstance(mount, str) or total is None or available is None:
            raise ValueError("invalid disk values")
        result.append({
            "mount_point": mount[:160],
            "total_bytes": total,
            "used_bytes": total - available,
            "available_bytes": available,
            "source": "Get-CimInstance Win32_LogicalDisk",
        })
    return result


def collect_disk(runner: CommandRunner = run_command) -> dict[str, Any]:
    if os.name != "nt":
        return _section([], "unavailable", "disk source unsupported")
    result = runner([
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json -Compress",
    ])
    if result.returncode != 0:
        return _section([], "unavailable", "disk source unavailable")
    try:
        data = parse_disk_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section([], "unavailable", "disk data invalid")
    return _section(data, "available" if data else "unavailable", None if data else "disk data empty")


def parse_network_json(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("network source is not a list")
    result = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("invalid network record")
        raw_addresses = item.get("addresses", [])
        if isinstance(raw_addresses, str):
            raw_addresses = [raw_addresses]
        if not isinstance(raw_addresses, list):
            raise ValueError("invalid network addresses")
        addresses = [
            address[:160]
            for address in raw_addresses
            if isinstance(address, str) and address
        ]
        result.append({
            "name": item["name"][:160],
            "addresses": addresses,
            "link_state": str(item.get("status", "unknown"))[:40],
        })
    return result


def collect_network(runner: CommandRunner = run_command) -> dict[str, Any]:
    if os.name != "nt":
        return _section([], "unavailable", "network source unsupported")
    result = runner([
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-NetIPConfiguration -ErrorAction SilentlyContinue | ForEach-Object { [pscustomobject]@{ name=$_.InterfaceAlias; status=$_.NetAdapter.Status; addresses=@($_.IPv4Address | ForEach-Object { $_.IPAddress }) + @($_.IPv6Address | ForEach-Object { $_.IPAddress }) } } | ConvertTo-Json -Compress",
    ])
    if result.returncode != 0:
        return _section([], "unavailable", "network source unavailable")
    try:
        data = parse_network_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section([], "unavailable", "network data invalid")
    return _section(data, "available" if data else "unavailable", None if data else "network interfaces unavailable")
~~~

Each collector returns a mapping with 'status' and either 'data' or a bounded 'diagnostics' list. Implement the platform helpers with these exact local-only rules:

~~~python
def parse_cpu_counter(text: str) -> dict[str, Any]:
    try:
        utilization = float(text.strip().splitlines()[0])
    except (ValueError, IndexError):
        utilization = None
    return {
        "model": platform.processor() or None,
        "logical_cores": os.cpu_count(),
        "utilization_percent": utilization,
        "per_core_percent": [],
        "load_average": list(getattr(os, "getloadavg", lambda: ())()),
        "sampled_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_duration_ms": 0,
        "source": "Get-Counter Processor(_Total)",
    }
~~~

Each helper returns explicit null fields when a value is unavailable and never raises a raw exception to the CLI. Use these concrete rules:

- 'collect_gpu' runs the read-only 'nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits' query and may read the local 'nvidia-smi' header for CUDA Version; a missing header leaves 'cuda_version' null and does not invalidate device data.
- 'collect_gpu_processes' runs the dedicated-process query and, on Windows only, the existing read-only 'Get-Counter \\\\GPU Process Memory(*)\\\\Shared Usage' query.
- A failed GPU query returns '{"status": "unavailable", "data": [], "diagnostics": ["gpu source unavailable"]}'.
- A successful GPU query with no shared counter returns 'shared_memory_bytes: None'; it does not convert the missing value to zero.
- 'collect_memory' uses the read-only Windows 'Win32_OperatingSystem' query and leaves every missing field null; unsupported platforms return an explicit unavailable section.
- 'collect_cpu' uses a bounded current-usage command sample and reports model/core count when available; it does not cache the sample.
- 'collect_disk' uses the read-only Windows logical-disk query for all fixed local volumes; it reports bytes and never deletes or writes.
- 'collect_network' uses the read-only local IP-configuration query for interface names, addresses, and link state; it does not query traffic counters or make a connection.

### Step 5: Implement section selection and JSON output

Define:

~~~python
SECTION_NAMES = (
    "overview",
    "cpu",
    "memory",
    "gpu",
    "gpu-processes",
    "disk",
    "network",
)


def _overview_status(parts: dict[str, dict[str, Any]]) -> str:
    statuses = [part["status"] for part in parts.values()]
    if all(status == "available" for status in statuses):
        return "available"
    if any(status == "available" for status in statuses):
        return "partial"
    return "unavailable"


def collect_hardware_state(
    sections: list[str],
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    selected = set(sections)
    if not selected:
        selected = {"overview"}
    unknown = selected - set(SECTION_NAMES) - {"all"}
    if unknown:
        raise ValueError("unknown hardware section")
    if "all" in selected:
        selected = set(SECTION_NAMES)

    result: dict[str, Any] = {}
    for name in SECTION_NAMES:
        if name not in selected:
            continue
        if name == "overview":
            parts = {
                "cpu": collect_cpu(runner),
                "memory": collect_memory(runner),
                "gpu": collect_gpu(runner),
                "disk": collect_disk(runner),
            }
            result[name] = {
                "status": _overview_status(parts),
                "data": parts,
            }
        elif name == "cpu":
            result[name] = collect_cpu(runner)
        elif name == "memory":
            result[name] = collect_memory(runner)
        elif name == "gpu":
            result[name] = collect_gpu(runner)
        elif name == "gpu-processes":
            result[name] = collect_gpu_processes(runner)
        elif name == "disk":
            result[name] = collect_disk(runner)
        elif name == "network":
            result[name] = collect_network(runner)
    return result
~~~

Expand 'all' to every real section except the alias itself. The default section list is '["overview"]'. 'overview' calls only the low-cost summary collectors and never 'collect_gpu_processes'.

Use this envelope:

~~~python
{
    "schema_version": 1,
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "sections": result,
}
~~~

~~~python
def _requested_statuses(sections: dict[str, Any]) -> list[str]:
    return [
        section["status"]
        for section in sections.values()
        if isinstance(section, dict) and "status" in section
    ]


def build_payload(
    sections: list[str],
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sections": collect_hardware_state(sections, runner),
    }


def _write_json(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except (OSError, UnicodeError):
        return False
    return True


class _CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


def main(
    argv: list[str] | None = None,
    *,
    runner: CommandRunner = run_command,
) -> int:
    parser = _CliParser(description="Read-only hardware state")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--section",
        action="append",
        choices=[*SECTION_NAMES, "all"],
        dest="sections",
    )
    selection.add_argument("--all", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    except ValueError:
        return 1
    requested = ["all"] if args.all else (args.sections or ["overview"])
    try:
        payload = build_payload(requested, runner)
        if not _write_json(payload):
            return 1
    except (OSError, ValueError, TypeError, RuntimeError):
        return 1
    statuses = _requested_statuses(payload["sections"])
    return 2 if any(status in {"unavailable", "partial"} for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~

The exact 'json.dumps(payload, ensure_ascii=True, sort_keys=True)' call keeps output parseable when the stream cannot encode ordinary Unicode. The mutually exclusive argparse group prevents combining '--section' and '--all'; '--help' exits before any collector is called. Return 2 when a requested section is unavailable or partial, 0 when all requested sections are available, and 1 for invalid arguments or an internal output/validation failure.

### Step 6: Run the hardware tests and compile the script

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state -v
& .\.venv\Scripts\python.exe -m py_compile scripts\hardware_state.py tests\agent_context\test_hardware_state.py
~~~

Expected: all tests pass and compilation exits 0. Do not run a live probe.

---

## Task 5: Add agent-facing documentation and repository pointers

**Files:**
- Create: 'docs/agent/HARDWARE_STATE.md'
- Modify: 'docs/agent/SCHEMA.md'
- Modify: 'AGENTS.md'
- Modify: 'docs/agent/OPERATIONS.md'
- Modify: 'tests/agent_context/test_hardware_state.py'
- Modify: 'docs/agent/reports/context-validation-2026-09-24.md'

### Step 1: Add documentation contract tests

Add tests that read 'docs/agent/HARDWARE_STATE.md' and assert that it contains:

- the venv invocation;
- '--section', '--all', and the default overview;
- 'dedicated_vram' and 'shared_memory' as separate names;
- per-process reporting;
- 'unavailable'/'partial' semantics;
- the no-write/no-kill/no-network boundary;
- the explicit user-opened hardware-window requirement;
- the distinction between local measurements and a global audit.

Add a test that 'AGENTS.md' and 'docs/agent/OPERATIONS.md' link to the new document. Do not assert a live machine name, PID, VRAM value, or timestamp in documentation. Use this concrete test:

~~~python
class HardwareDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    def test_hardware_guide_documents_the_contract(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(encoding="utf-8")
        for required in (
            "scripts\\hardware_state.py",
            "--section",
            "--all",
            "dedicated_vram",
            "shared_memory",
            "gpu-processes",
            "unavailable",
            "partial",
            "user-opened",
            "global",
        ):
            self.assertIn(required, text)

    def test_router_and_operations_link_to_hardware_guide(self):
        for relative in ("AGENTS.md", "docs/agent/OPERATIONS.md"):
            text = (self.root / relative).read_text(encoding="utf-8")
            self.assertIn("docs/agent/HARDWARE_STATE.md", text)
~~~

### Step 2: Write 'docs/agent/HARDWARE_STATE.md'

Document the exact commands:

~~~powershell
& .\.venv\Scripts\python.exe scripts\hardware_state.py
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section cpu --section memory
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section gpu --section gpu-processes
& .\.venv\Scripts\python.exe scripts\hardware_state.py --all
~~~

Explain in plain language that dedicated VRAM is memory physically attached to the GPU, while shared GPU memory is system memory used by GPU processes. State that the two values are never added together and that a shared-memory increase can indicate a slower or spill-prone workload. Explain that per-process rows help detect co-running work but do not establish ownership or authorize stopping a process.

State that the command is a local live probe, not a global process/network audit, and must only be run in an explicitly opened hardware window.

### Step 3: Add the schema and router pointers

Add a 'Context exception registry' subsection to 'docs/agent/SCHEMA.md' documenting 'schema_version', required record fields, allowed kinds, exact matching, and fail-closed validation.

Add one row to the 'AGENTS.md' task-routed matrix for hardware-state queries pointing to 'docs/agent/HARDWARE_STATE.md' and 'rules/gpu.md'. Add 'docs/agent/HARDWARE_STATE.md' to the quick links. Do not add any live hardware value.

Add a short 'Hardware state probe' section to 'docs/agent/OPERATIONS.md' stating that the command is read-only, selective, and user-window-gated, and that its output is a local measurement rather than global adoption evidence.

### Step 4: Run documentation and link tests

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state tests.agent_context.test_check_agent_context -v
~~~

Expected: documentation, link, and linter tests pass. If the linter reports a new broken link, fix the new documentation link only; do not alter historical source documents.

### Step 5: Update the validation report after evidence exists

Update 'docs/agent/reports/context-validation-2026-09-24.md' only after running the commands. Record:

- the four accepted historical exception keys;
- the focused and full test commands and their actual exit results;
- the linter result after applying exact exceptions;
- renderer '--check' result;
- the hardware tests as 'PASS';
- the live 'hardware_state.py --all' probe as 'SKIPPED' because it was not run;
- the global process/network/adoption audit as 'BLOCKED';
- the preserved dirty-worktree boundary.

Do not invent test counts, hardware values, or live timestamps. Do not change the global audit label to 'PASS'.

---

## Task 6: Complete validation, independent review, and worktree checks

**Files:**
- No new files except validation evidence already listed above.
- Review all intended changed paths without staging them.

### Step 1: Run focused tests

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_exceptions -v
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state -v
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -v
~~~

Expected: each module exits 0. Negative-test diagnostics must remain controlled and must not contain raw secrets, absolute paths, or tracebacks.

### Step 2: Run the full context suite and compilation

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t . -q
& .\.venv\Scripts\python.exe -m py_compile scripts\check_agent_context.py scripts\hardware_state.py tests\agent_context\test_context_exceptions.py tests\agent_context\test_hardware_state.py tests\agent_context\test_check_agent_context.py
~~~

Expected: discovery exits 0 and compilation exits 0.

### Step 3: Run the linter and renderer checks

Run:

~~~powershell
& .\.venv\Scripts\python.exe scripts\check_agent_context.py
& .\.venv\Scripts\python.exe scripts\render_current_state.py --check
~~~

Expected:

- linter prints four 'ACCEPTED HISTORICAL:' lines and exits 0 when no new unaccepted finding exists;
- renderer check exits 0;
- no generated file is hand-edited.

If the linter prints any unaccepted finding, stop and fix the implementation or registry matching; do not suppress it by broad substring matching.

### Step 4: Run whitespace and protected-path checks

Run:

~~~powershell
git diff --check -- scripts/check_agent_context.py scripts/hardware_state.py state/context_exceptions.yaml tests/agent_context/test_context_exceptions.py tests/agent_context/test_hardware_state.py tests/agent_context/test_check_agent_context.py docs/agent/HARDWARE_STATE.md docs/agent/SCHEMA.md docs/agent/OPERATIONS.md AGENTS.md docs/agent/reports/context-validation-2026-09-24.md
git diff --cached --check
git diff --name-only HEAD -- SOUL.md rules state/lessons_pending.json
~~~

Expected: intended-path and cached checks exit 0; the protected-path command prints no paths. Preserve unrelated existing '.superpowers/sdd/' whitespace findings rather than cleaning them.

### Step 5: Obtain independent review

Ask an independent reviewer to inspect:

- exact exception matching and fail-closed registry validation;
- no broad suppression;
- separate dedicated VRAM/shared-memory fields;
- PID joining and null preservation;
- no writes, process termination, or network operations;
- ASCII-safe partial output;
- documentation and report consistency;
- dirty-worktree preservation.

The review verdict must be recorded as 'CLEAN', or every finding must be fixed and re-reviewed before completion.

### Step 6: Capture final worktree evidence

Run:

~~~powershell
git status --short
git diff --name-only HEAD
git diff --cached --name-only
~~~

Confirm that:

- historical 'TASKS.md' and 'research/EXPERIMENTS.md' are unchanged by this task;
- no protected policy file was changed except the explicitly approved minimal pointer updates listed in Task 5;
- no commit, push, merge, reset, stash, clean, or deletion occurred;
- the live hardware probe remains unclaimed and 'SKIPPED';
- the global audit remains 'BLOCKED'.

## Final acceptance checklist

- [ ] Four historical findings are accepted through exact registry records.
- [ ] New, shifted, changed, and unlisted findings still fail.
- [ ] Historical source documents remain unchanged.
- [ ] Hardware command is selective, read-only, JSON-only, and on-demand.
- [ ] Dedicated VRAM and shared memory are separate at device and process levels.
- [ ] Missing sources are explicit and never guessed.
- [ ] Tests prove no writes, process termination, or network activity.
- [ ] Documentation gives the agent exact usage instructions.
- [ ] Full tests, compilation, linter, renderer, and whitespace checks are recorded.
- [ ] Independent review is 'CLEAN'.
- [ ] Live hardware probe is 'SKIPPED' unless separately user-opened.
- [ ] Global process/network/adoption audit remains 'BLOCKED'.
- [ ] Dirty worktree is preserved with no commit or integration action.
