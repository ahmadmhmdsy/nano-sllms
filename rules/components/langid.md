---
id: components/langid
status: active
scope: langid
applies_when: "The task touches the CPU-only language-identification line: its scripts, gates, or reports."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - langid
  - research/desert_ant_recreation/DESIGN.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# langid — Tongue-analogue D/A-line component rule

## Scope and entry points
- `langid/` is text language identification under the
  desert-ant-recreation design: design at
  `research/desert_ant_recreation/DESIGN.md`, plan under `docs/plans/`,
  CPU-only scripts under `langid/`.
- This line is CPU-only by design and never claims GPU time.

## Smallest validation
Validation is the pre-registered evaluation gates (registered as experiment
rows); no gate is claimed PASS without the registered artifact behind it.

## Boundaries and hazards
- CPU-only by design: never "help" it onto the GPU.
- Gates pre-register before any run per `rules/experiments.md`.

## Evidence sources
- The design's pre-registered gate table and its registered evaluation
  artifacts (Tongue-analogue eval outputs), not console memory.
