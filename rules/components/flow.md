---
id: components/flow
status: active
scope: flow
applies_when: "The task touches the visual flow editor: its React frontend, FastAPI backend, or example flows."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - flow/README.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# flow — visual flow editor component rule

## Scope and entry points
- Frontend build: `cd flow; pnpm build` (bundles into `flow/dist`).
- Backend service: run the flow server through the venv with its documented
  module entry; the port is an explicit argument of the launch, never a
  guess.
- Tests: vitest for the frontend; unittest under `flow/tests` for the
  server (run with the repo venv and the documented scope).

## Smallest validation
Frontend changes: build plus the vitest suite. Backend changes: the unittest
suite plus one HTTP round-trip probe of the touched endpoint.

## Boundaries and hazards
- Node work uses pnpm only; never mix package managers and never delete a
  lockfile.
- Never start a second flow server on a busy port; only servers this
  session started get stopped by this session.

## Evidence sources
- `flow/README.md`, the flow design and plan documents under
  `docs/plans/`, and the test suites themselves.
