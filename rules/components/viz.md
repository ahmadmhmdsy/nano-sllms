---
id: components/viz
status: active
scope: viz
applies_when: "The task touches the React model-architecture explorer: viz/, its config-driven views, or its build and tests."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - viz/README.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# viz — architecture explorer component rule

## Scope and entry points
- `viz/` is the config-driven React model-architecture explorer, built
  with pnpm only; docs in `viz/README.md`, plan under `docs/plans/`.

## Smallest validation
Every viz change ends with the pnpm build plus its own component test
suite; a console-only render is not a verdict.

## Boundaries and hazards
- Config-driven by design: per-model views come from configs, never from a
  hard-coded model dump in source.
- Never download model weights for rendering; the explorer works from
  configs and recorded artifacts.

## Evidence sources
- `viz/README.md`, the explorer's config inputs, and the plan documents
  under `docs/plans/`.
