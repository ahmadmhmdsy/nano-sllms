# Agent Context Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace split-brain agent context with a small stable router, generated current state, conditional rules, task reflections, and a gated self-improvement loop.

**Architecture:** `SOUL.md` owns stable agent identity; `AGENTS.md` owns repository-wide routing and safety; `rules/` owns conditional specialist policy; `state/` and `research/experiments.yaml` own machine-readable lifecycle records; `CURRENT_STATE.md` is generated; `docs/reflections/` stores task-local closeout reports. Reflection is automatic, but permanent rule or persona changes are staged, evaluated, and explicitly adopted.

**Tech Stack:** UTF-8 Markdown; YAML syntax through the existing PyYAML 6.0.3 environment; JSON for generated/runtime records; Python standard library plus existing PyYAML; `unittest`; PowerShell; repository `.venv`.

---

## Execution constraints

- Work in the existing `E:\python_projects\nano_SLMs` checkout only after checking `git status`.
- Preserve every pre-existing deleted, modified, and untracked path.
- Never run `git clean`, `git reset`, `git stash`, bulk deletion, or history rewriting.
- Stage exact paths only; never use `git add .`.
- Context work is CPU-only. Do not launch training, evaluation, inference, probes, downloads, or servers.
- Use `& .\.venv\Scripts\python.exe` for every Python command.
- Do not place secrets, raw transcripts, or user dataset contents in registries or reflections.
- Keep each task independently reviewable and commit only the files listed in that task.

## Planned file map

### Create

- `SOUL.md` — stable identity and reasoning posture.
- `rules/README.md` — conditional-rule usage and lifecycle.
- `rules/index.yaml` — machine-readable rule index.
- `rules/gpu.md` — shared-GPU and process-safety policy.
- `rules/experiments.md` — pre-registration, evidence, and verdict policy.
- `rules/data.md` — preservation, regeneration, and chunking policy.
- `rules/evidence.md` — source-artifact and metric-reporting policy.
- `rules/context.md` — loading, stale-state, and documentation policy.
- `rules/components/core.md`
- `rules/components/diacritizer.md`
- `rules/components/laya.md`
- `rules/components/ui.md`
- `rules/components/mex.md`
- `rules/components/langid.md`
- `rules/components/flow.md`
- `rules/components/viz.md`
- `rules/components/webui.md`
- `docs/agent/OPERATIONS.md`
- `docs/agent/REFLECTION.md`
- `docs/agent/SCHEMA.md`
- `docs/agent/SELF_IMPROVEMENT.md`
- `docs/reflections/README.md`
- `docs/reflections/INDEX.md`
- `docs/reflections/_template.md`
- `state/tasks.yaml`
- `state/decisions.yaml`
- `state/environment.json`
- `state/jobs.json`
- `state/lessons_pending.json`
- `state/current.json` (generated)
- `CURRENT_STATE.md` (generated)
- `research/experiments.yaml`
- `scripts/check_agent_context.py`
- `scripts/render_current_state.py`
- `scripts/closeout_reflection.py`
- `scripts/validate_context_proposal.py`
- `tests/agent_context/__init__.py`
- `tests/agent_context/test_check_agent_context.py`
- `tests/agent_context/test_render_current_state.py`
- `tests/agent_context/test_closeout_reflection.py`
- `tests/agent_context/test_context_proposal.py`
- `tests/agent_context/fixtures/valid_state/`
- `tests/agent_context/fixtures/invalid_state/`
- `docs/plans/README.md`
- `research/README.md`

### Modify

- `AGENTS.md`
- `CLAUDE.md`
- `README.md`
- `ENVIRONMENT.md`
- `PLAN.md`
- `docs/agent/` references from `AGENTS.md`
- `TASKS.md` only after a reviewed migration report
- `research/EXPERIMENTS.md` only after the experiment registry is validated

### Rename/archive

- `data/snapshot_row13_row16_wip/docs/AGENTS.md` → `data/snapshot_row13_row16_wip/docs/AGENTS.archive.md`, only after verifying the target does not exist and the archive contains no required active instructions.

---

## Task 1: Build the read-only context linter

**Files:**
- Create: `scripts/check_agent_context.py`
- Create: `tests/agent_context/__init__.py`
- Create: `tests/agent_context/test_check_agent_context.py`
- Create: `tests/agent_context/fixtures/valid_state/tasks.yaml`
- Create: `tests/agent_context/fixtures/valid_state/experiments.yaml`
- Create: `tests/agent_context/fixtures/invalid_state/tasks.yaml`
- Create: `tests/agent_context/fixtures/invalid_state/experiments.yaml`

### Step 1: Write failing tests

Add tests for these public functions:

```python
from pathlib import Path

from scripts.check_agent_context import (
    check_records,
    check_markdown_links,
    check_static_policy,
)

def test_duplicate_task_ids_are_reported():
    issues = check_records(
        {"tasks": [{"id": "ui/u1"}, {"id": "ui/u1"}], "experiments": []},
        root=Path("."),
    )
    assert any("duplicate task id" in issue for issue in issues)

def test_invalid_status_is_reported():
    issues = check_records(
        {"tasks": [{"id": "ui/u1", "status": "finished"}], "experiments": []},
        root=Path("."),
    )
    assert any("invalid task status" in issue for issue in issues)

def test_static_policy_rejects_live_process_claim():
    issues = check_static_policy(
        {"AGENTS.md": "PID 123 is live as of now.\n"},
        root=Path("."),
    )
    assert issues
```

### Step 2: Run the tests and verify they fail

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
```

Expected: FAIL because `scripts/check_agent_context.py` does not exist.

### Step 3: Implement the minimum linter

Implement these functions with no network, process termination, or GPU access:

```python
def check_records(records: dict, root: Path) -> list[str]:
    """Return stable, human-readable validation issues."""

def check_markdown_links(path: Path, root: Path) -> list[str]:
    """Check local Markdown links and explicitly listed evidence paths."""

def check_static_policy(files: dict[str, str], root: Path) -> list[str]:
    """Reject live-state claims and bare-Python commands in policy files."""

def main(argv: list[str] | None = None) -> int:
    """Print all issues and return 0 only when the context is valid."""
```

Required checks:

- unique namespaced task IDs;
- unique experiment IDs;
- allowed task statuses;
- allowed experiment statuses/verdicts;
- malformed `||` table rows;
- relative Markdown links;
- evidence paths;
- live PID/job/current-machine claims in `AGENTS.md` and `CLAUDE.md`;
- bare `python` and `pip` commands in policy files;
- references to archived active `AGENTS.md` files.

### Step 4: Run focused and full linter tests

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -v
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
```

Expected: all tests PASS.

### Step 5: Run the linter against the current tree

Run:

```powershell
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
```

Expected before migration: a non-zero exit with a finite issue list. Do not “fix” issues by deleting historical files.

### Step 6: Commit only linter files

Run:

```powershell
git add scripts/check_agent_context.py tests/agent_context/__init__.py tests/agent_context/test_check_agent_context.py tests/agent_context/fixtures/valid_state/tasks.yaml tests/agent_context/fixtures/valid_state/experiments.yaml tests/agent_context/fixtures/invalid_state/tasks.yaml tests/agent_context/fixtures/invalid_state/experiments.yaml
git commit -m "test: add agent context linter"
```

---

## Task 2: Add canonical state registries and current-state renderer

**Files:**
- Create: `state/tasks.yaml`
- Create: `state/decisions.yaml`
- Create: `state/environment.json`
- Create: `state/jobs.json`
- Create: `state/lessons_pending.json`
- Create: `state/current.json`
- Create: `research/experiments.yaml`
- Create: `scripts/render_current_state.py`
- Create: `tests/agent_context/test_render_current_state.py`

### Step 1: Write failing renderer tests

Use temporary directories and assert:

```python
from pathlib import Path

def test_render_is_deterministic_for_same_inputs(tmp_path):
    first = render_state(fixture_root=tmp_path)
    second = render_state(fixture_root=tmp_path)
    assert first == second

def test_conflicting_job_state_renders_unknown(tmp_path):
    rendered = render_state(fixture_root=tmp_path)
    assert "UNKNOWN" in rendered

def test_check_mode_does_not_write(tmp_path):
    before = (tmp_path / "CURRENT_STATE.md").read_bytes()
    result = render_state(fixture_root=tmp_path, mode="check")
    after = (tmp_path / "CURRENT_STATE.md").read_bytes()
    assert before == after
    assert result.ok is False
```

### Step 2: Run tests and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_render_current_state -v
```

Expected: FAIL because the renderer and fixtures do not exist.

### Step 3: Create initial registry records

Create records for the currently known active lines without inventing runtime state:

- one completed `ui/task-u1` record matching the completed U-1 evidence;
- one pending `ui/task-u1b` semantic-follow task;
- one pending `ui/task-u1c` diversity task;
- one completed `diacritizer/task-e63` record;
- the active Laya follow-up records as `laya/task-e75` and its explicitly recorded rungs;
- active/blocked records only when supported by an existing evidence path.

Use `UNKNOWN` for unverified ownership, active job, or last-verified time. Preserve old numeric IDs in `aliases`.

### Step 4: Implement the renderer

Implement the following types and functions:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RenderResult:
    text: str
    ok: bool

def load_registries(root: Path) -> dict:
    """Load tasks, experiments, decisions, environment, and jobs."""

def render_state(fixture_root: Path, mode: str = "write") -> RenderResult:
    """Render deterministic CURRENT_STATE.md and state/current.json."""

def collect_git_metadata(root: Path) -> dict:
    """Read branch, commit, and dirty-state counts without changing Git."""
```

`--write` may update only `CURRENT_STATE.md` and `state/current.json`. `--check` must never write. The renderer must not inspect or terminate processes; it reads `state/jobs.json` only.

### Step 5: Run renderer tests and check mode

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_render_current_state -v
& .\.venv\Scripts\python.exe scripts/render_current_state.py --check
```

Expected: tests PASS; repository check initially FAILs until the generated snapshot is created.

### Step 6: Generate and review the first snapshot

Run:

```powershell
& .\.venv\Scripts\python.exe scripts/render_current_state.py --write
& .\.venv\Scripts\python.exe scripts/render_current_state.py --check
```

Expected: both commands PASS, and `CURRENT_STATE.md` contains no invented active job or machine fact.

### Step 7: Commit registry/renderer files

Run:

```powershell
git add scripts/render_current_state.py state/tasks.yaml state/decisions.yaml state/environment.json state/jobs.json state/lessons_pending.json state/current.json CURRENT_STATE.md research/experiments.yaml tests/agent_context/test_render_current_state.py
git commit -m "feat: add generated agent current state"
```

---

## Task 3: Add SOUL and conditional rules

**Files:**
- Create: all files listed under `rules/` in the planned file map.
- Create: `SOUL.md`
- Create: `docs/agent/OPERATIONS.md`
- Create: `docs/agent/SCHEMA.md`
- Modify: `tests/agent_context/test_check_agent_context.py`

### Step 1: Add front-matter and rule-index tests

Test that every rule has:

- `id`;
- `status`;
- `scope`;
- `applies_when`;
- `owner`;
- `last_reviewed`;
- `evidence`.

Test that `rules/index.yaml` points only to existing rule files.

### Step 2: Run tests and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -v
```

Expected: FAIL because the rule files do not exist.

### Step 3: Write `SOUL.md`

Use the approved content boundary:

- mission;
- priorities;
- evidence posture;
- collaboration posture;
- learning posture;
- uncertainty handling;
- no mutable project facts;
- no duplicate repository rules.

### Step 4: Write conditional rules

Implement these concrete policies:

- `rules/gpu.md`: one protected repository GPU training job; no repository GPU co-run; no foreign-process kills; no shared-memory spill; resize job shape on OOM.
- `rules/experiments.md`: pre-register, anchor evaluator, source-parse metrics, hard-fail overlap, smoke must prove updates, honest partial verdicts.
- `rules/data.md`: preserve source data, bounded chunking, regeneration command, atomic/mmap-safe writes, no user-data deletion.
- `rules/evidence.md`: exact relative evidence paths, source artifact authority, no remembered numbers, stable IDs.
- `rules/context.md`: targeted reads, `UNKNOWN` on conflicts, no historical prose as current state, no auto-edit of live rules.
- Component rules contain only entry points, scoped validation, hazards, and evidence sources; they do not duplicate the root policy.

### Step 5: Write `OPERATIONS.md` and `SCHEMA.md`

Document launch gates, read-back, smoke validity, safe resume, GPU rules, status vocabularies, and record fields.

### Step 6: Run tests and linter

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
```

Expected: rule/schema tests PASS; repository-wide linter may still report legacy task/experiment issues.

### Step 7: Commit persona/rules files

Run:

```powershell
git add SOUL.md rules/README.md rules/index.yaml rules/gpu.md rules/experiments.md rules/data.md rules/evidence.md rules/context.md rules/components/core.md rules/components/diacritizer.md rules/components/laya.md rules/components/ui.md rules/components/mex.md rules/components/langid.md rules/components/flow.md rules/components/viz.md rules/components/webui.md docs/agent/OPERATIONS.md docs/agent/SCHEMA.md tests/agent_context/test_check_agent_context.py
git commit -m "docs: add agent persona and conditional rules"
```

---

## Task 4: Rewrite the root adapters and add scoped component instructions

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Create: `diacritizer/AGENTS.md`
- Create: `laya/AGENTS.md`
- Create: `ui/AGENTS.md`
- Create: `mex/AGENTS.md`
- Create: `langid/AGENTS.md`
- Create: `flow/AGENTS.md`
- Create: `viz/AGENTS.md`
- Create: `webui/AGENTS.md`
- Rename: `data/snapshot_row13_row16_wip/docs/AGENTS.md` to `data/snapshot_row13_row16_wip/docs/AGENTS.archive.md`

### Step 1: Add a root-adapter regression test

Assert that `AGENTS.md`:

- contains no current PID or live-job claim;
- routes all active component directories;
- links `SOUL.md`, `CURRENT_STATE.md`, `rules/README.md`, and `docs/agent/OPERATIONS.md`;
- uses one validation vocabulary;
- says direct user instructions outrank repository boilerplate.

Assert that `CLAUDE.md` is a thin adapter and does not duplicate the full operating manual.

### Step 2: Run the regression test and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -v
```

Expected: FAIL against the current root files.

### Step 3: Rewrite `CLAUDE.md`

Keep only:

- read `AGENTS.md` first;
- direct-user priority;
- `CURRENT_STATE.md` and nearest scoped instructions;
- dirty-worktree preservation;
- no GPU co-run;
- no destructive operations;
- concise completion/reporting requirements.

### Step 4: Rewrite `AGENTS.md`

Keep:

- project portfolio map;
- stable repository safety;
- task-routed read matrix;
- component routing;
- validation and evidence rules;
- update triggers;
- links to `SOUL.md`, `rules/`, `OPERATIONS.md`, and `CURRENT_STATE.md`.

Remove:

- live machine/job claims;
- hard-coded test counts;
- historical outage policy;
- generic tool-pricing rules;
- duplicated generic engineering guidance.

### Step 5: Add component instructions

Each component file must contain:

- scope;
- source-of-truth state path;
- entry points;
- smallest validation commands;
- data boundaries;
- known hazards;
- required evidence.

Do not copy root safety policy into every component.

### Step 6: Archive the stale nested instruction file

Before moving, verify the source exists and the destination does not. Move only that exact file; do not alter archived data or user artifacts.

Run:

```powershell
Test-Path data\snapshot_row13_row16_wip\docs\AGENTS.md
Test-Path data\snapshot_row13_row16_wip\docs\AGENTS.archive.md
```

Expected: `True` then `False`. Perform the move, then run the linter to confirm no active archived `AGENTS.md` remains.

### Step 7: Run tests and inspect the diff

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
git diff --check
```

Expected: adapter tests PASS; no whitespace errors; remaining registry migration issues are listed separately.

### Step 8: Commit only adapter/component files

Run:

```powershell
git add AGENTS.md CLAUDE.md diacritizer/AGENTS.md laya/AGENTS.md ui/AGENTS.md mex/AGENTS.md langid/AGENTS.md flow/AGENTS.md viz/AGENTS.md webui/AGENTS.md data/snapshot_row13_row16_wip/docs/AGENTS.archive.md
git add -A -- data/snapshot_row13_row16_wip/docs/AGENTS.md data/snapshot_row13_row16_wip/docs/AGENTS.archive.md
git commit -m "docs: scope agent instructions by component"
```

If the archive file is untracked, stage only the destination and leave unrelated archive data untouched.

---

## Task 5: Add task-closeout reflections

**Files:**
- Create: `docs/agent/REFLECTION.md`
- Create: `docs/reflections/README.md`
- Create: `docs/reflections/INDEX.md`
- Create: `docs/reflections/_template.md`
- Create: `scripts/closeout_reflection.py`
- Create: `tests/agent_context/test_closeout_reflection.py`
- Create: `state/lessons_pending.json`

### Step 1: Write failing reflection tests

Test:

- default mode prints a draft and writes nothing;
- `--write` requires an explicit output path;
- required fields are checked;
- existing files are not overwritten without an explicit replacement flag;
- likely secret patterns are rejected or redacted;
- a reflection with a lesson emits a pending-lesson candidate but does not edit `SOUL.md` or `rules/`.

### Step 2: Run tests and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_closeout_reflection -v
```

Expected: FAIL because the reflection tool does not exist.

### Step 3: Implement the deterministic reflection generator

Provide this interface:

```python
def build_reflection(payload: dict) -> str:
    """Render a validated reflection from a JSON payload."""

def write_reflection(path: Path, content: str, replace: bool = False) -> None:
    """Write one new reflection without silently overwriting another."""

def stage_lesson_candidate(payload: dict, state_path: Path) -> dict:
    """Append a lesson candidate without activating a permanent rule."""

def main(argv: list[str] | None = None) -> int:
    """Default to dry-run; require --write for repository output."""
```

The tool may accept a JSON file containing the reflection fields. It must not call a model, launch jobs, inspect secrets, or automatically edit policy files.

### Step 4: Add the reflection policy and template

Document the hybrid persistence policy:

- every non-trivial task gets a short closeout reflection;
- durable files are created for failures, near-misses, reusable lessons, evidence corrections, or process changes;
- routine successful tasks do not create repository noise.

### Step 5: Run tests and a dry-run example

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_closeout_reflection -v
& .\.venv\Scripts\python.exe scripts/closeout_reflection.py --input tests/agent_context/fixtures/reflection_sample.json
```

Expected: test PASS; command prints a draft and creates no file.

### Step 6: Commit reflection files

Run:

```powershell
git add docs/agent/REFLECTION.md docs/reflections/README.md docs/reflections/INDEX.md docs/reflections/_template.md scripts/closeout_reflection.py state/lessons_pending.json tests/agent_context/test_closeout_reflection.py tests/agent_context/fixtures/reflection_sample.json
git commit -m "feat: add task closeout reflections"
```

---

## Task 6: Add gated self-improvement proposals

**Files:**
- Create: `docs/agent/SELF_IMPROVEMENT.md`
- Create: `scripts/validate_context_proposal.py`
- Create: `tests/agent_context/test_context_proposal.py`
- Modify: `state/lessons_pending.json`

### Step 1: Write failing proposal-gate tests

Test that the validator:

- compares baseline and candidate context;
- runs the static linter against both;
- reports regressions separately from improvements;
- refuses to write live `AGENTS.md`, `CLAUDE.md`, `SOUL.md`, or `rules/` files;
- writes only a proposal report and diff;
- never calls `skillopt-sleep` adoption automatically.

### Step 2: Run tests and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_proposal -v
```

Expected: FAIL because the proposal validator does not exist.

### Step 3: Implement the proposal validator

Provide the following types and functions:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ProposalReport:
    baseline_ok: bool
    candidate_ok: bool
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    diff: str

def validate_proposal(baseline: Path, candidate: Path, root: Path) -> ProposalReport:
    """Run static/context checks and return a non-adopting report."""

def render_diff(baseline: Path, candidate: Path) -> str:
    """Return a unified diff for user review."""

def main(argv: list[str] | None = None) -> int:
    """Write only a staged report; never adopt live policy."""
```

### Step 4: Document the SkillOpt-Sleep boundary

Document:

- deterministic mock first;
- bounded edit budget;
- baseline and candidate scores;
- held-out validation;
- staged output location;
- backup before adoption;
- explicit user adoption;
- no automatic `SOUL.md` evolution in the first phase.

### Step 5: Run tests and mock validation

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_proposal -v
& .\.venv\Scripts\python.exe scripts/validate_context_proposal.py --help
```

Expected: tests PASS; help output documents that the tool cannot adopt live policy.

### Step 6: Commit proposal tooling

Run:

```powershell
git add docs/agent/SELF_IMPROVEMENT.md scripts/validate_context_proposal.py state/lessons_pending.json tests/agent_context/test_context_proposal.py
git commit -m "feat: add gated context improvement proposals"
```

---

## Task 7: Repair task and experiment views through migration reports

**Files:**
- Create: `docs/agent/reports/context-migration-2026-09-24.md`
- Modify: `TASKS.md`
- Modify: `research/EXPERIMENTS.md`
- Modify: `state/tasks.yaml`
- Modify: `research/experiments.yaml`
- Modify: `MEMORY.md` only for explicit supersession metadata

### Step 1: Generate a migration report

The report must list, without editing the source ledgers:

- duplicate task IDs;
- duplicate experiment IDs;
- malformed rows;
- stale status claims;
- conflicting environment claims;
- old IDs and proposed namespaced aliases;
- evidence path existence.

### Step 2: Review the report manually

Check each proposed change against source artifacts. Do not infer a current status from the newest-looking prose when a machine artifact disagrees.

### Step 3: Migrate active records

For each active task:

- assign one namespaced ID;
- preserve the old ID in `aliases`;
- choose exactly one status;
- set one next action;
- attach exact evidence;
- remove historical narrative from the active view.

For each experiment:

- assign one namespaced ID;
- preserve old IDs as aliases;
- distinguish `closed` from `running`;
- separate verdict from evidence;
- keep narrative reports intact.

### Step 4: Regenerate views and run the linter

Run:

```powershell
& .\.venv\Scripts\python.exe scripts/render_current_state.py --write
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
```

Expected: duplicate IDs, malformed rows, and missing evidence paths are resolved or explicitly reported as historical exceptions.

### Step 5: Commit migration outputs

Run:

```powershell
git add docs/agent/reports/context-migration-2026-09-24.md TASKS.md research/EXPERIMENTS.md state/tasks.yaml research/experiments.yaml
git commit -m "docs: migrate task and experiment registries"
```

If `MEMORY.md` has a safe, user-approved supersession change, stage that exact file in a separate reviewed commit; otherwise leave it untouched.

---

## Task 8: Add documentation indexes and update human-facing maps

**Files:**
- Create: `docs/plans/README.md`
- Create: `research/README.md`
- Modify: `README.md`
- Modify: `ENVIRONMENT.md`
- Modify: `PLAN.md`

### Step 1: Write index tests

Test that every plan/research index entry has:

- path;
- scope;
- status;
- owner or `UNKNOWN`;
- last verified date or `UNKNOWN`;
- supersedes field.

Test that `README.md` links to all active component entry points and no longer presents the original S/P/T pipeline as the whole project.

### Step 2: Run tests and verify failure

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
```

Expected: FAIL until the indexes and documentation updates exist.

### Step 3: Write the indexes

`docs/plans/README.md` must classify plans as `active`, `completed`, `blocked`, or `historical`. `research/README.md` must classify reports, experiment ledgers, templates, and raw payloads without reading raw payloads into the index.

### Step 4: Update `README.md` and `ENVIRONMENT.md`

- README becomes a human-facing portfolio overview with links to scoped instructions.
- ENVIRONMENT retains stable policy and machine profiles; volatile current values move to `state/environment.json` and `CURRENT_STATE.md`.
- Bare Python and package-manager contradictions are removed.
- Missing dependency ownership is stated explicitly.

### Step 5: Mark `PLAN.md` correctly

Add a historical/current banner and turn it into a portfolio index. Do not delete the original specification; preserve it as an approved historical baseline with a clear status.

### Step 6: Run validation and commit

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
& .\.venv\Scripts\python.exe scripts/render_current_state.py --check
git diff --check
```

Expected: tests PASS, renderer check PASS, and the linter reports only explicitly accepted historical exceptions.

Commit:

```powershell
git add docs/plans/README.md research/README.md README.md ENVIRONMENT.md PLAN.md tests/agent_context/test_check_agent_context.py
git commit -m "docs: index project context and research"
```

---

## Task 9: Run the complete read-only validation gate

**Files:**
- No implementation source changes.
- Create only if needed: `docs/agent/reports/context-validation-2026-09-24.md`

### Step 1: Verify the worktree baseline

Run:

```powershell
git status --short
git diff --check
```

Record the pre-existing dirty paths in the validation report. Do not alter them.

### Step 2: Run all context tests

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .
```

Expected: PASS.

### Step 3: Run static and generated-state checks

Run:

```powershell
& .\.venv\Scripts\python.exe scripts/check_agent_context.py
& .\.venv\Scripts\python.exe scripts/render_current_state.py --check
```

Expected: PASS.

### Step 4: Run a deterministic SkillOpt-Sleep mock only

Invoke the installed SkillOpt-Sleep runner in `mock` mode, scoped to `E:\python_projects\nano_SLMs`, with a small edit budget. Do not use a real model backend and do not run `adopt`. If the runner is not installed, record this check as `SKIPPED: SkillOpt-Sleep runner unavailable` rather than substituting a live run.

Expected: staged proposal only; baseline and candidate results are recorded; live policy files remain unchanged.

### Step 5: Review the final diff and report

Run:

```powershell
git diff --check
git status --short
git log -5 --oneline
```

Confirm that:

- no GPU/network job was launched;
- no foreign process was stopped;
- no pre-existing path was removed;
- all generated files have a source;
- all live rules have evidence and an owner;
- all lessons awaiting promotion are clearly staged.

### Step 6: Commit the validation report only

Run:

```powershell
git add docs/agent/reports/context-validation-2026-09-24.md
git commit -m "test: validate agent context architecture"
```

---

## Final acceptance checklist

- [ ] `SOUL.md` contains stable identity only.
- [ ] `AGENTS.md` is a compact, accurate repository router.
- [ ] `CLAUDE.md` is a thin adapter.
- [ ] `CURRENT_STATE.md` is generated and timestamped.
- [ ] Task and experiment IDs are unique and namespaced.
- [ ] Active and completed work are separated.
- [ ] Conditional rules are routed, not globally loaded.
- [ ] GPU, data, evidence, and resume rules are explicit.
- [ ] Task reflections are deterministic and privacy-safe.
- [ ] Lesson candidates cannot directly edit live policy.
- [ ] Proposal validation produces a diff and gate report.
- [ ] SkillOpt-Sleep remains staged and opt-in.
- [ ] Context linter and regression tests pass.
- [ ] Existing dirty worktree paths remain intact.
