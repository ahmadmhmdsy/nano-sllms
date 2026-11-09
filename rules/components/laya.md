---
id: components/laya
status: active
scope: laya
applies_when: The task runs or extends the laya line, including its unattended launch gates and pre-registered param checks.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - laya/scripts
  - research/EXPERIMENTS.md
  - AGENTS.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# laya — L-line component rule

## Scope and entry points
- `laya/scripts/preflight_check.py`: the required launch gate before any
  unattended long launch (config validation script: duplicate keys,
  pyflakes, --help smoke, data state, GPU smoke).
- `laya/scripts/param_count.py`: the pre-registration param check used
  before any launch.
- Experiment records: namespaced `laya/E-xxx`.

## Smallest validation
No launch, however small, skips the preflight gate; no param claim lands in
a report without the param-count artifact as its evidence.

## Boundaries and hazards
- Unattended-launch gates and resume/read-back procedure live in
  `docs/agent/OPERATIONS.md`; the single-job GPU rule lives in
  `rules/gpu.md`.
- Every laya arm is pre-registered under `rules/experiments.md`.

## Evidence sources
- Preflight and param-count artifacts saved to the run's own log location;
  the verdict cells in `research/EXPERIMENTS.md`.
