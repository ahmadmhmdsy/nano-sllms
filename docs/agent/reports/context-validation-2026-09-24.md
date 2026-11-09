# Agent context validation report

- Plan: `docs/superpowers/plans/2026-09-24-agent-context-reliability.md`
- Scope: Tasks 1–9 context architecture, deterministic closeout reflection, and non-adopting proposal validation
- Verification checkout: `.` (repository root)
- Verification base: `89f63a86478a35018f379cb90fdf672bbc1fa370` plus the documented working-tree safety corrections
- Validation vocabulary: `PASS`, `FAIL`, `SKIPPED`, `BLOCKED`, `NEEDS_USER_DECISION`

## Result summary

| Gate | Label | Evidence and disposition |
|---|---|---|
| Agent-context regression suite | PASS | The repository-venv discovery command exited 0 after the final regression additions and reported `341 tests ... OK (skipped=3)`. |
| Focused linter and proposal regressions | PASS | The linter and proposal modules exited 0 and reported 85 and 98 tests respectively; the combined total is `183 tests ... OK (skipped=3)`. |
| Focused reflection regressions | PASS | The focused reflection module exited 0 and reported `143 tests ... OK`, including preflight and encoding-safety regressions. |
| Interpreter compilation | PASS | The repository-venv `-m py_compile` check completed with exit 0 for the context scripts and regression tests. |
| Context linter | FAIL | The linter exits 1 for four retained historical findings in `TASKS.md` and `research/EXPERIMENTS.md`; repair is a separate user-gated historical-ledger decision. |
| Renderer freshness | PASS | The approved renderer `--write` refreshed only `CURRENT_STATE.md` and `state/current.json`; the subsequent `--check` exited 0. |
| Whole-worktree whitespace | FAIL | Unstaged `git diff --check` exits 2 with only pre-existing findings in preserved `.superpowers/sdd/` files; `git diff --cached --check` exits 0. |
| Intended-path whitespace | PASS | The scoped intended-path whitespace check exits 0 with no findings. |
| SkillOpt-Sleep mock validation | SKIPPED | `SKIPPED: SkillOpt-Sleep runner unavailable`; no runner was installed or invoked. |
| Permanent adoption/global process audit | BLOCKED | No durable global audit log exists; no global no-adoption, CPU/network, or process-interference claim is made. |
| Protected paths and generated state | PASS | No diff exists in `SOUL.md`, `AGENTS.md`, `CLAUDE.md`, `rules/`, or `state/lessons_pending.json`; generated changes are limited to the two renderer-owned files. |
| Worktree preservation | PASS | Pre-existing deleted, modified, staged, and untracked paths remain visible; no cleanup, reset, stash, or bulk deletion was performed. |

Overall disposition: **NEEDS_USER_DECISION**. The implementation, regression, compilation, and generated-state gates pass. Full acceptance remains open for the four explicitly preserved historical ledger findings and the unavailable global audit; no permanent adoption was attempted.

## Fresh regression evidence

Commands were run from the repository root through the repository virtual environment:

```text
& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t . -q
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context -q
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_closeout_reflection -q
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_proposal -q
& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_render_current_state -q
& .\.venv\Scripts\python.exe -m py_compile scripts\check_agent_context.py scripts\closeout_reflection.py scripts\render_current_state.py scripts\validate_context_proposal.py tests\agent_context\test_check_agent_context.py tests\agent_context\test_closeout_reflection.py tests\agent_context\test_context_proposal.py tests\agent_context\test_render_current_state.py
```

The discovery command exited 0 and reported `Ran 341 tests ... OK (skipped=3)`. The linter module reported `Ran 85 tests ... OK`; the proposal module reported `Ran 98 tests ... OK (skipped=3)`; the reflection module reported `Ran 143 tests ... OK`; and the renderer module reported `Ran 15 tests ... OK`. These focused counts sum to the 341-test discovery result; the combined linter/proposal total is 183 tests. Expected argparse usage, overwrite-refusal, and protected-path messages in negative-test stderr are expected controlled errors, not test failures. The compilation command exited 0 with no output.

## Safety corrections verified

### Medium 7 — bounded evidence diagnostics

`scripts/check_agent_context.py` now sends missing-evidence values through a bounded diagnostic boundary. Known credential/token forms are replaced with `<redacted-secret-evidence>`, control-containing values with `<redacted-control-evidence>`, and long safe values with a bounded `<truncated-evidence>` marker. Ordinary safe evidence paths retain the existing useful missing-path diagnostic. No raw unsafe value or traceback is emitted.

The regressions in `tests/agent_context/test_check_agent_context.py` cover secret-like values, control characters, and long safe values. The focused linter module and full discovery suite are green.

### Medium 8 — preflight filesystem diagnostics

The reflection CLI now routes preflight `OSError` and other validation failures through the same bounded diagnostic boundary. Secret-bearing, control-containing, and overlong exception details are replaced with `filesystem operation failed`; tracebacks are not emitted. The regression `test_cli_redacts_secret_bearing_preflight_filesystem_failure` verifies that a simulated preflight failure returns exit code 2, leaves the output absent, and does not disclose the token or traceback.

### Low 9 — external state-path boundary

`docs/agent/REFLECTION.md`, `docs/agent/SELF_IMPROVEMENT.md`, and the `scripts/closeout_reflection.py` module docstring explicitly state that an external `--state-path` is an isolated-validation fixture only, never a live lesson registry or alternative adoption target, and must not be copied into canonical state. Repository-contained lesson staging remains restricted to `state/lessons_pending.json`.

### CLI encoding and internal metadata boundary

`check_agent_context.py` routes argparse and CLI issue output through `_safe_print`; unsupported ordinary Unicode is escaped for an ASCII stream without producing a raw `UnicodeEncodeError`. Windows reparse-metadata failures use a fixed redacted diagnostic rather than exposing an internal absolute path. The linter regressions cover the ASCII-stream issue path, direct lstat failures, and reparse-metadata failures.

`validate_context_proposal.py` routes all CLI output through `_write_cli_text`. Dry-run JSON is reparsed with `ensure_ascii=True` when an ASCII stream cannot encode ordinary Unicode, while the live-secret rejection reason is preserved without wrapping it as an encoding error. The proposal regressions cover error output, parseable escaped dry-run JSON, and the corrected rejection reason.

## Implementation artifacts

- `scripts/check_agent_context.py`
- `scripts/closeout_reflection.py`
- `scripts/validate_context_proposal.py`
- `tests/agent_context/test_check_agent_context.py`
- `tests/agent_context/test_closeout_reflection.py`
- `tests/agent_context/test_context_proposal.py`
- `docs/agent/REFLECTION.md`
- `docs/agent/SELF_IMPROVEMENT.md`
- `docs/agent/reports/context-validation-2026-09-24.md`

## Static and generated-state checks

The linter command:

```text
& .\.venv\Scripts\python.exe scripts\check_agent_context.py
```

exited 1 with these four retained historical findings:

```text
ISSUE: malformed table row: TASKS.md:60
ISSUE: malformed table row: TASKS.md:196
ISSUE: broken markdown link: research/EXPERIMENTS.md:25: research/_template_experiment.md
ISSUE: broken markdown link: research/EXPERIMENTS.md:28: research/WHAT_WORKS.md
```

The linter's whole-tree preflight remains fail-closed. Generated dependency and SkillOpt staging stores are deliberately excluded from descendant traversal; their entry points remain inspected, and the stores were not deleted, cleaned, or adopted.

The renderer refresh and check:

```text
& .\.venv\Scripts\python.exe scripts/render_current_state.py --write
render: PASS (write mode, root=<repository-root>)
& .\.venv\Scripts\python.exe scripts/render_current_state.py --check
render: PASS (check mode, root=<repository-root>)
```

The machine-local root is normalized in this report.

The `<repository-root>` label is a redacted rendering, not a literal transcription of the local absolute path. The recorded `--write` changed only `CURRENT_STATE.md` and `state/current.json`; neither generated artifact was hand-edited.

## Worktree evidence

The verification commands were:

```text
git status --short
git diff --check; git diff --cached --check
git diff --check -- CURRENT_STATE.md state/current.json docs/agent/reports/context-validation-2026-09-24.md scripts/check_agent_context.py scripts/closeout_reflection.py scripts/validate_context_proposal.py scripts/render_current_state.py tests/agent_context/test_check_agent_context.py tests/agent_context/test_closeout_reflection.py tests/agent_context/test_context_proposal.py tests/agent_context/test_render_current_state.py docs/agent/REFLECTION.md docs/agent/SELF_IMPROVEMENT.md docs/reflections/README.md
git log -5 --oneline
```

Whole-worktree `git diff --check` reported only these unrelated SDD findings:

- `.superpowers/sdd/task-1-report.md:3`: trailing whitespace
- `.superpowers/sdd/task-1-report.md:4`: trailing whitespace
- `.superpowers/sdd/task-5-brief.md:82`: new blank line at EOF
- `.superpowers/sdd/task-6-brief.md:89`: new blank line at EOF
- `.superpowers/sdd/task-7-brief.md:69`: new blank line at EOF

These findings were not bulk-cleaned. Existing deleted, modified, and untracked paths remain visible. The test-created `.skillopt-sleep/staging/` artifacts were not adopted or cleaned. The plan's report-only commit step was intentionally not executed: the session constraint is to preserve the pre-existing dirty worktree without staging, committing, pushing, or merging.

## Independent review

**CLEAN.** A fresh post-fix independent review found no remaining Critical, Important, or Minor findings. It independently verified the linter, reflection, and proposal regression suites; compilation; stream-safe ASCII output and parseable escaped dry-run JSON; exception, secret, control-character, path, and traceback redaction; default policy enforcement; non-adopting behavior; protected outputs; empty lesson staging; and the absence of external side effects. The reviewer did not modify files. The four historical linter findings and the unavailable global process/network/adoption audit remain explicitly user-gated rather than being silently treated as clean.

## Reflection and self-improvement boundary

The reflection tool is deterministic and offline by design. It validates payload text, exact repository-relative evidence paths, no-follow metadata, protected destinations, state shape, and identity facts before writing. Lesson candidates are opt-in and staged only; they do not promote themselves.

The proposal validator compares baseline and candidate context, runs static checks, and writes only explicitly requested staged report/diff outputs. Its CLI has no adoption action and does not invoke a model, training job, server, or network operation by design. These design statements are not a claim that a global process or network audit was performed.

`state/lessons_pending.json` remains an empty staging registry. `SOUL.md`, `AGENTS.md`, `CLAUDE.md`, and `rules/` were not modified by reflection or proposal validation.

### Legacy identity compatibility

Current candidate identity includes scope, source reflection, reusable lesson, rule candidate, and evidence. Committed three-field legacy IDs are recognized and reused when their persisted facts match. An absent legacy `rule_candidate` is unknown rather than proof of a changed rule; changed evidence and any persisted rule mismatch produce a new current identity. The implementation does not invent a rule value for an absent historical field.

## Safety limitations

- No-follow checks inspect lexical path components with metadata, but the guarantee is path-based; a concurrent replacement between inspection and use (TOCTOU) is not claimed to be impossible.
- Output/state writes are sequential and non-transactional. A later filesystem failure can leave earlier staged output; this is reported rather than presented as atomicity.
- The four historical findings are now accepted through exact registry records rather than left as a linter failure; see the 2026-09-25 addendum. Unrelated dependency and control stores remain scoped out rather than rewritten.
- Renderer freshness is `BLOCKED` in the 2026-09-25 addendum: the snapshot records commit `89f63a8` while `HEAD` is `24c942d`, and both generated files carry pre-existing user modifications. Neither was hand-edited or regenerated.
- The SkillOpt-Sleep mock runner was unavailable, and no global process/network or adoption audit log was produced.
- This report does not claim CPU-only execution, network inactivity, process noninterference, handle-based no-follow, atomic writes, or TOCTOU prevention.

## Final disposition

**NEEDS_USER_DECISION**, with a separate `BLOCKED` limitation for the unavailable global process/network/adoption audit. The four historical ledger findings remain preserved and require an explicit user decision before any ledger rewrite. No automatic permanent self-improvement adoption was invoked in these validation commands.

The 2026-09-25 addendum below records that explicit decision: the four
findings are now accepted historical exceptions, and no ledger was rewritten.

## 2026-09-25 addendum — accepted historical exceptions and hardware-state probe

### Accepted exception keys

The four findings are accepted by exact-match records in
`state/context_exceptions.yaml` (schema_version 1). The historical ledgers
themselves were not edited.

| Key | Kind |
| --- | --- |
| `TASKS.md:60` | `malformed_table_row` |
| `TASKS.md:196` | `malformed_table_row` |
| `research/EXPERIMENTS.md:25` | `broken_markdown_link` -> `research/_template_experiment.md` |
| `research/EXPERIMENTS.md:28` | `broken_markdown_link` -> `research/WHAT_WORKS.md` |

Matching is exact on (path, line, kind, target) and is never a substring
match. A changed line, a changed link target, a new finding, or an unlisted
finding still fails, and record-only mode keeps reporting the raw findings.
A missing, unreadable, unsafe, duplicated, or otherwise malformed registry
fails closed: it accepts nothing and reports a controlled issue.

### Validation evidence

| Check | Command | Result |
| --- | --- | --- |
| Exception registry suite | `& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_context_exceptions` | PASS (30 tests) |
| Hardware-state suite | `& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_hardware_state` | PASS (81 tests) |
| Linter suite | `& .\.venv\Scripts\python.exe -m unittest tests.agent_context.test_check_agent_context` | PASS (94 tests) |
| Full context suite | `& .\.venv\Scripts\python.exe -m unittest discover -s tests/agent_context -t .` | PASS (461 tests, 3 platform skips) |
| Compilation | `& .\.venv\Scripts\python.exe -m py_compile scripts/check_agent_context.py scripts/hardware_state.py tests/agent_context/test_context_exceptions.py tests/agent_context/test_hardware_state.py tests/agent_context/test_check_agent_context.py` | PASS |
| Context linter | `& .\.venv\Scripts\python.exe scripts/check_agent_context.py` | PASS, exit 0, four `ACCEPTED HISTORICAL:` lines, no unaccepted finding |
| Renderer freshness | `& .\.venv\Scripts\python.exe scripts/render_current_state.py --check` | BLOCKED, see below |
| Whitespace and protected paths | `git diff --check -- <intended paths>`; `git diff --cached --check`; `git diff --name-only HEAD -- SOUL.md rules state/lessons_pending.json` | PASS, no output |
| Live hardware probe | `& .\.venv\Scripts\python.exe scripts/hardware_state.py --all` | SKIPPED, no hardware window was opened |
| Global process/network/adoption audit | not run | BLOCKED |

### Renderer freshness

`render_current_state.py --check` reports the snapshot as stale because
`CURRENT_STATE.md` records commit `89f63a8` while `HEAD` is `24c942d`, and
the recorded dirty-file count no longer matches the worktree. Both generated
files already carried pre-existing user modifications, so neither was
regenerated or hand-edited. The check is reported as BLOCKED rather than
silently refreshed.

### Hardware-state probe boundary

`scripts/hardware_state.py` is a selective, read-only, on-demand local probe
for CPU, memory, disk, network, GPU, dedicated VRAM, shared GPU memory, and
opt-in per-process GPU state. Dedicated VRAM and shared GPU memory stay
separate fields at device and process level and are never summed; a missing
measurement is null rather than zero. The command writes no file, stops no
process, and makes no network call, and its automated tests use an injected
fake runner so no live source is ever queried.

**SKIPPED**: the live `--all` probe was not run because no hardware window
was opened. This local measurement does not satisfy, and must not be reported
as satisfying, the global process, network, and adoption audit, which remains
**BLOCKED**.

### Worktree boundary

All work stayed uncommitted and unstaged in the existing dirty checkout. No
reset, stash, clean, deletion, commit, merge, or push occurred, and
`TASKS.md` and `research/EXPERIMENTS.md` are unchanged.
