---
id: components/diacritizer
status: active
scope: diacritizer
applies_when: "The task touches the Arabic diacritization D-line: its scripts, data, training, evaluation, or reports."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - diacritizer
  - research/EXPERIMENTS.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# diacritizer — D-line component rule

## Scope and entry points
- Everything under `diacritizer/` (the D-line submodule with its own
  `README.md` and `diacritizer/scripts/` for training and serving).
- Experiment records are namespaced `diacritizer/E-xxx` in
  `research/EXPERIMENTS.md` and the canonical registry.

## Smallest validation
Before any D-line launch, run the line's own preflight and confirm the
registered config parses; never launch from an unregistered config.

## Boundaries and hazards
- D-line training is a live GPU workload: any launch decision first reads
  `rules/gpu.md` (one protected training job; no foreign-process kills)
  and `docs/agent/OPERATIONS.md` for launch gates, read-back, and safe
  resume.
- Arms are pre-registered under `rules/experiments.md` (registration,
  anchor evaluator, overlap hard-fail).
- The deployed gold checkpoint is the anchor until a registered verdict
  changes it.

## Evidence sources
- `diacritizer/README.md`; the D-line verdict rows in
  `research/EXPERIMENTS.md`; registered gate-bench artifacts.
