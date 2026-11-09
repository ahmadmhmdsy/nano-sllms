---
id: components/mex
status: active
scope: mex
applies_when: "The task touches the micro-expert (mu) sandbox line: its trainers, configs, replay data, or adapters."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - mex
  - configs
  - research/EXPERIMENTS.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# mex — mu-line component rule

## Scope and entry points
- `mex/` holds the mu-line micro-expert sandbox (trainers plus the
  registered replay configs under `configs/`).
- Runs are governed by their own registered experiment rows in
  `research/EXPERIMENTS.md`.

## Smallest validation
Before any mu-line launch decision, run the registered config's own
preflight (config parse, script smoke); never launch from an unvalidated
config.

## Boundaries and hazards
- GPU decisions follow `rules/gpu.md`; data and adapter artifacts follow
  `rules/data.md` (a replay corpus regeneration carries its recorded
  command, never a silent overwrite).

## Evidence sources
- Verdict rows in `research/EXPERIMENTS.md`; adapter/checkpoint
  artifacts under the registered run directory.
