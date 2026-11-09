---
id: components/ui
status: active
scope: ui
applies_when: "The task touches the U-line chain surface (ui/): corpus build, pico training, validity gates, or UI data tooling."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - ui
  - webui
  - WEBUI_PRD.md
  - research/EXPERIMENTS.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# ui — U-line component rule

## Scope and entry points
- The U-line lives under `ui/` (scripts and data) together with the
  Gradio surface in `webui/`; the product spec is `WEBUI_PRD.md`.
- Records are namespaced `ui/E-xxx` for experiments and `ui/task-...`
  for tasks.
- Validity gates are the pre-registered minimum validity metric on a
  pre-registered held-out task set, evaluated by the run's own eval script.

## Smallest validation
Before any U-line claim, parse the registered artifact (metrics JSON, per
set validity report) directly; never quote a console-only number.

## Boundaries and hazards
- The corpus under `ui/data/` is large and partly untracked: deletion or
  regeneration needs the `rules/data.md` boundary and is never done as a
  drive-by cleanup.
- The U-line is the chain-surface training line; generic Gradio shell work
  has its own component rule (`components/webui.md`).

## Evidence sources
- Validity rows in `research/EXPERIMENTS.md`; registered run
  directories; the corpus build README under `ui/data/` as applicable.
