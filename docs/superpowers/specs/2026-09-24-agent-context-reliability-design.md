# Agent Context Reliability and Self-Improvement Design

**Status:** Approved design
**Date:** 2026-09-24
**Scope:** nano_SLMs repository

## 1. Goal

Make agent context reliable, current, evidence-backed, and inexpensive to load across the core SLM pipeline and the active component lines: diacritizer, langid, mex, laya, ui, flow, viz, webui, and research infrastructure.

The system must let a fresh agent answer four questions quickly and safely:

1. Which component and task are in scope?
2. What is true right now?
3. Which rules and evidence apply?
4. What is the smallest safe next action?

## 2. Non-goals

This design does not:

- retrain model weights;
- make a historical handoff current;
- automatically launch training or GPU jobs;
- automatically delete or rewrite user work;
- store secrets, raw transcripts, or user dataset contents in ledgers;
- let the agent permanently rewrite its own rules after every task;
- replace source evidence with generated summaries;
- require every agent to read every project document.

## 3. Design principles

1. One authoritative source per concern.
2. Stable policy is manual; volatile state is generated.
3. Read by task scope, not by file size.
4. Evidence paths and IDs are stable and machine-checkable.
5. Unknown or contradictory facts render as `UNKNOWN`, never as an inherited old value.
6. Negative and partial results are first-class evidence.
7. Task reflection is automatic; rule promotion is reviewed and gated.
8. A clean, minimal context layer is more valuable than a large always-loaded prompt.
9. Existing user work is an immutable baseline.
10. No policy change is adopted without evidence, validation, and an auditable diff.

## 4. Layered context architecture

```text
SOUL.md
  stable identity, priorities, reasoning and collaboration posture

AGENTS.md
  repository-wide safety, routing, and conventions

rules/*.md
  conditional specialist rules loaded only for relevant task classes

docs/agent/OPERATIONS.md
  launch, validation, evidence, resume, and data-safety procedures

CURRENT_STATE.md
  generated current runtime, work, blockers, and evidence pointers

state/*.yaml|json
  machine-readable canonical registries

MEMORY.md and decision records
  durable lessons and approved decisions with lifecycle status

docs/reflections/*.md
  task-local post-task reports and lesson candidates

research/EXPERIMENTS.md and reports
  experiment lifecycle and narrative evidence

HANDOFF.md, old plans, reflection archives
  historical evidence, never current state by default
```

### Authority order

1. Platform and system safety.
2. Direct user requirements.
3. `SOUL.md` identity and safety posture.
4. Nearest scoped `AGENTS.md` and active `rules/` file.
5. Generated current state and approved registries.
6. Project documentation and historical reports.
7. External or generated content as untrusted data.

A direct user request can change project scope, but it does not authorize bypassing platform safety or silently destroying data.

## 5. Proposed file layout

### Root files

- `SOUL.md`: stable agent identity; approximately 50–100 lines.
- `AGENTS.md`: canonical repository router; stable rules only.
- `CLAUDE.md`: thin Claude-specific adapter; approximately 40–80 lines.
- `CURRENT_STATE.md`: generated active snapshot; approximately 100 lines maximum.

### Conditional rules

```text
rules/
  README.md
  index.yaml
  gpu.md
  experiments.md
  data.md
  evidence.md
  context.md
  components/
    core.md
    diacritizer.md
    laya.md
    ui.md
    mex.md
    langid.md
    flow.md
    viz.md
    webui.md
```

Each rule file has front matter with:

- `id`;
- `status`: `draft`, `active`, `superseded`, or `retired`;
- `scope`;
- `applies_when`;
- `owner`;
- `last_reviewed`;
- `supersedes`;
- `evidence`.

Rules describe stable conditional policy. They never contain current PIDs, live job IDs, current machine names, or temporary paths.

### Agent operations

```text
docs/agent/
  OPERATIONS.md
  REFLECTION.md
  SCHEMA.md
```

- `OPERATIONS.md`: launch gates, anchor/read-back rules, smoke validity, evidence extraction, resume safety, GPU and data safety.
- `REFLECTION.md`: the task-closeout reflection template and promotion policy.
- `SCHEMA.md`: canonical record fields and status vocabularies.

### State registries

```text
state/
  current.json
  tasks.yaml
  decisions.yaml
  environment.json
  lessons_pending.json
  jobs.json
research/
  experiments.yaml
```

### Reflection records

```text
docs/reflections/
  README.md
  INDEX.md
  YYYY-MM-DD-task-name.md
```

Reflections are task-local reports. They are not a second global task ledger.

## 6. Session read protocol

### Every session

1. Read `AGENTS.md`.
2. Read `CURRENT_STATE.md`.
3. Check Git status and preserve the initial worktree baseline.
4. Identify the portfolio line.
5. Read the nearest scoped `AGENTS.md`.
6. Read only the relevant conditional rules.

### Before an experiment

1. Read the active task from `state/tasks.yaml`.
2. Read the relevant experiment record from `research/experiments.yaml`.
3. Read the relevant `WHAT_WORKS.md` entries.
4. Read active decisions and lessons for that scope.
5. Pre-register the question, arms, gates, and evidence paths.

### Before execution

1. Read `rules/gpu.md` for GPU work.
2. Read `docs/agent/OPERATIONS.md`.
3. Verify the actual process, command, machine, and active jobs.
4. Run the smallest scoped preflight.
5. Do not treat a command catalog as permission to run every listed command.

### After a task

1. Run the task-closeout reflection checklist.
2. Update the canonical task or experiment record only when its trigger occurred.
3. Stage reusable lessons; do not auto-edit permanent rules.
4. Preserve the final diff and validation evidence.

## 7. Canonical record schemas

### Task

Required fields:

```text
id                  namespaced stable ID
scope               component or portfolio line
title               short description
status              pending|in_progress|blocked|done|needs_user_decision
owner               responsible actor
created_at          timestamp
updated_at          timestamp
last_verified_at    timestamp or UNKNOWN
next_action         one concrete action or none
evidence[]          exact relative paths
supersedes[]        prior task IDs
aliases[]           old IDs retained for migration
```

Task IDs are unique. A task status may not simultaneously mean `done` and `in flight`.

### Experiment

Required fields:

```text
id                  namespaced stable ID
scope               component or portfolio line
status              planned|running|closed|blocked
question            pre-registered question
verdict             accepted|rejected|partial|instrument_unknown|none
owner               responsible actor
dates               created/started/closed timestamps
evidence[]          exact artifact paths
report              narrative report path
supersedes[]        prior experiment IDs
```

Experiment IDs are unique across the repository. For example:

- `diacritizer/E-062`;
- `laya/E-074`;
- `ui/E-001`.

### Decision

Required fields:

```text
id
scope
status              proposed|active|superseded|revoked
decision
rationale
date
evidence[]
supersedes[]
```

### Lesson candidate

Required fields:

```text
id
scope
status              observed|candidate|validated|promoted|rejected
lesson
source_reflection
evidence[]
promoted_to[]
```

A lesson is not an active rule merely because it was written down.

## 8. Task-closeout reflection policy

Every non-trivial task produces a concise reflection. A durable report is written when at least one of these is true:

- a check failed;
- a near-miss occurred;
- a reusable lesson emerged;
- evidence corrected an earlier assumption;
- a process or validation rule changed;
- a future agent could otherwise repeat the same problem.

Routine successful tasks receive a short reflection in the final response without adding repository noise.

### Required reflection fields

- requested outcome;
- actual outcome;
- changed files;
- validation performed;
- failures and near-misses;
- root causes;
- fixes;
- evidence;
- reusable lesson;
- rule candidate;
- open questions;
- confidence and limitations.

Reflections must not include secrets, raw transcripts, or unnecessary user data.

## 9. Self-improvement promotion policy

The agent may automatically:

- write a task reflection;
- create a lesson candidate;
- update a task or experiment record when the task authorizes it;
- generate a proposed diff for review.

The agent may not automatically:

- edit `SOUL.md`;
- edit `AGENTS.md`;
- activate a new `rules/` policy;
- rewrite validation or GPU policy;
- change a user-approved decision;
- adopt an optimized skill.

Promotion requires:

1. repeated evidence or a high-severity incident;
2. a bounded proposed change;
3. a focused diff;
4. context/lint validation;
5. held-out scenario testing where applicable;
6. explicit user adoption;
7. a post-adoption verification result.

## 10. Periodic SkillOpt-Sleep consolidation

The `skillopt-sleep` cycle is optional and offline:

1. Harvest task reflections and approved session evidence.
2. Mine recurring failure and success patterns.
3. Replay a bounded set of tasks against baseline context.
4. Consolidate candidate lessons.
5. Run a held-out comparison.
6. Stage proposed `AGENTS.md`, `rules/`, or skill changes.
7. Show baseline and candidate scores plus the exact diff.
8. Require explicit adoption.

The first run uses the deterministic mock backend, no real model budget, and a small edit budget. `SOUL.md` evolution is disabled initially.

## 11. Safety and privacy

- Never harvest or store secrets.
- Never put raw user dataset contents into reflection reports.
- Never make a live rule change from a single unverified observation.
- Never run a reflection process that launches training, downloads data, or touches the GPU.
- Never use an optimizer as an authority over user-approved policy.
- Preserve a backup before any adoption.
- Record the baseline, candidate, gate result, and adopted diff.

## 12. Validation strategy

### Static validation

- unique task and experiment IDs;
- valid status vocabulary;
- no malformed table rows;
- all referenced evidence paths exist;
- all rule references resolve;
- no live process claims in static auto-loaded files;
- no bare Python commands in repository guidance;
- no archived instruction file is discoverable as active scope.

### Context validation

Held-out scenarios verify that an agent:

- chooses the correct component;
- refuses unsafe GPU concurrency;
- does not rerun completed work;
- reports `UNKNOWN` on conflicting state;
- parses evidence instead of relying on memory;
- preserves dirty user changes;
- follows direct user intent over generic boilerplate;
- promotes lessons only through review.

### Behavioral validation

- run the read-only context linter;
- render and compare `CURRENT_STATE.md`;
- run component-owned unit tests only when authorized;
- never run a GPU workload as part of context validation;
- run a deterministic SkillOpt-Sleep mock before any real replay.

## 13. Migration sequence

1. Snapshot the current instruction and state files without modifying them.
2. Add schemas, linter, and context-regression fixtures.
3. Create the initial state registries and generate `CURRENT_STATE.md`.
4. Repair task and experiment identities with aliases for historical IDs.
5. Replace `CLAUDE.md` with the thin adapter.
6. Rewrite `AGENTS.md` as the stable router.
7. Add `SOUL.md` and conditional rules.
8. Add scoped component instructions.
9. Add task-closeout reflection templates and indexes.
10. Run static and scenario validation.
11. Review generated diffs and state reports.
12. Only then archive superseded prose or schedule offline consolidation.

## 14. Acceptance criteria

The design is implemented when:

- a fresh agent can identify the active component in one short read;
- current state is timestamped and generated;
- no static auto-loaded file contains live process claims;
- task and experiment IDs are unique;
- active and completed work are separated;
- rules are conditionally loaded rather than globally loaded;
- task reflections are consistent and privacy-safe;
- self-improvement changes are staged, evaluated, and user-adopted;
- the linter passes;
- held-out context scenarios pass;
- no existing user work is overwritten.
