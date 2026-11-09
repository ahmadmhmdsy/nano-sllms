---
id: components/webui
status: active
scope: webui
applies_when: "The task touches the Gradio web UI shell: webui/, its milestone scope, server lifecycle, or Model-tab behaviour."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - WEBUI_PRD.md
  - webui/app.py
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# webui — Gradio shell component rule

## Scope and entry points
- `webui/` is the Gradio shell governed by `WEBUI_PRD.md`
  (milestones U1-U5); entry points are the documented module launch command
  and only those background jobs this repo itself started.

## Smallest validation
UI changes end with the Gradio app booting cleanly plus a real browser
interaction drill of the touched tab; console-only claims are not verdicts.
Drill evidence lives beside the run records referenced in TASKS.md.

## Boundaries and hazards
- Never claim a milestone complete from code review alone; browser-drill
  evidence is required for interaction claims.
- Gradio validation errors surface at interaction time, not boot: a
  dropdown or dataset change must be exercised once in the drill, not
  assumed from source.
- A UI server holding a stale CUDA context after model unload is a real
  GPU hazard: clean up only what this session started, per
  `rules/gpu.md`.

## Evidence sources
- The `WEBUI_PRD.md` milestone section, drill evidence files, and the
  recorded console/log artifacts from this repo's own jobs.
