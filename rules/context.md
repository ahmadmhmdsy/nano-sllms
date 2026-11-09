---
id: policy/context
status: active
scope: How context files are read, trusted, and edited by any agent in this repository.
applies_when: Every agent session; before any user-facing answer, task work, or write to SOUL.md, AGENTS.md, rules/, registries, or generated state.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-10-06"
supersedes: []
evidence:
  - AGENTS.md
  - SOUL.md
  - CURRENT_STATE.md
  - MEMORY.md
  - TASKS.md
  - HANDOFF.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
  - state/current.json
---

# Context policy

## Mandatory session bootstrap — read no matter what

This is a hard requirement, not a recommendation. It binds every agent in
this repository — the resident agent, subagents, and cloud agents — at
every session start and after every context compaction. (User-adopted
instruction, explicit, 2026-09-30.)

**Bootstrap set.** Read all of the following, in this order, before anything
else in the session:

1. [SOUL.md](../SOUL.md) — stable identity.
2. [CURRENT_STATE.md](../CURRENT_STATE.md) — generated current truth.
3. The nearest scoped [AGENTS.md](../AGENTS.md) (root or component).
4. [rules/README.md](README.md) plus the conditional rules for the task
   class — including this file.
5. [MEMORY.md](../MEMORY.md) — durable decisions and lessons.
6. [TASKS.md](../TASKS.md) — the what-done task ledger.
7. [HANDOFF.md](../HANDOFF.md) — the what-done narrative (latest entries;
   the file is newest-first).

**When the bootstrap is required:**

- **Session start.** The first action of any new session is the bootstrap.
  Never answer the user, search the project, or begin task work before it.
  Answers about the project come from these files and the project's own
  artifacts — never from parametric knowledge, a prior session's memory, or
  a guess about where something lives.
- **After context compaction.** If auto-compaction or any other context
  compression has dropped the bootstrap files from context, re-read them
  immediately. Not recalling where a file went or what it contained is the
  signal to re-read — never answer from a compressed or partial memory of
  the context.
- **Mid-session doubt.** Whenever the agent is unsure where it is, what was
  done, or what the rules are, re-read the relevant bootstrap file before
  continuing.

**Enforcement and penalty.** The user penalizes any session in which the
bootstrap was skipped, done from memory, or done only after the fact; a
missed bootstrap makes the session's results untrustworthy until the
bootstrap is completed and the gap is disclosed. The first reply of every
session must open with a one-line bootstrap confirmation naming the files
actually read (for example: "Bootstrap: SOUL.md, CURRENT_STATE.md,
AGENTS.md, rules/README.md, rules/context.md, MEMORY.md, TASKS.md,
HANDOFF.md."). A first reply without that line is treated as a missed
bootstrap.

## Targeted reads only
Read the slice the task needs: the root map, the nearest scoped guidance, the
relevant rules/ files, and the generated current state. Loading a large list
of documents is not a substitute for the exact record that matters.

## UNKNOWN on conflicts
Where two sources disagree (registry vs ledger vs prose), the rendered value
is UNKNOWN; it never silently inherits an older value. Resolve the conflict
against the source artifact; when it cannot be resolved this session, leave
the field UNKNOWN explicitly and note the conflict.

## Inspect first, record machine facts in ENVIRONMENT.md
Before substantial changes, inspect the environment with safe read-only
commands (OS, venv, versions, disk, git status). Do not install, delete,
migrate, reset, or upgrade anything during inspection; do not re-derive a
fact that [ENVIRONMENT.md](../ENVIRONMENT.md) already records — read it,
discover only what is stale or missing. For non-trivial tasks, add newly
detected durable facts (tools, versions, capabilities, limitations) back to
ENVIRONMENT.md; never encode live values (PIDs, disk numbers) there — those
belong to generated state.

## Ask the user when uncertain
Use active blocking rather than risky assumptions when a decision is
ambiguous, destructive, costly, or externally consequential. Raise it with
this exact format:

    QUESTION:
    [One precise question]

    CONTEXT:
    [What is unclear]

    OPTIONS:
    A. [Option]
    B. [Option]

    RECOMMENDATION:
    [Your recommendation and why]

Ask one focused question — never a checklist of trivia. Materially
uncertain triggers include: multiple valid interpretations; data could be
deleted or overwritten; security-relevant behavior; cost; production
impact; a needed real credential; conflicting conventions; a breaking API
or schema change; no safe implementation path; an irreversible next action.
Do not ask about trivial implementation choices, and record a blocked item
with an explicit NEEDS_USER_DECISION label when no in-session resolution
exists.

## Historical prose is not current state
Handoff narratives, old experiment reports, and archived plans are
historical evidence, not current truth. Use their dates and citations to
build the current state; never quote them as "how things are now". Anything
asserted live (a running job, an active PID, free disk) comes from the live
registry or a fresh measurement, not narrative history.

## No auto-edit of live rules
SOUL.md, AGENTS.md, and rules/ files are review-gated policy. Lesson
candidates, reflection upgrades, and skill adaptations stage proposed diffs;
they never silently rewrite an active rule. Any policy change carries its
own validation record.

## A file's self-declared authority governs whether it may be edited
Read a file's own banner before editing it. A ledger, report, or generated
artifact may declare itself frozen, historical, or deliberately redundant;
duplicate identifiers and malformed rows in such a file can be intentional,
and preserving them is the correct behaviour. When a historical record is
wrong, correct it through the designated current registry or an explicit
erratum, never by tidying the historical file itself. Generated files are
regenerated by their recorded command, never hand-edited.
