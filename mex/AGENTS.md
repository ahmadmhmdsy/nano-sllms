# mex/AGENTS.md - mu-line scoped instructions

## Scope
The micro-expert sandbox under `mex/` (mu0/mu1/mu2/mu3 and dia2-style
replay experts): trainers, eval and gate scripts, packing, adapters, and
their established configs under `configs/`. Dia-replay configs train the
shared core `src/model.py` under the mex line.

## State and records
- source-of-truth state path: `research/experiments.yaml` + the
  experiment rows for the active line recordings; `state/tasks.yaml`
  (scope `mex`).
- Reports and rows under `research/micro_experts/` and
  `research/EXPERIMENTS.md`.

## Entry points
- `mex/scripts/pack.py` and `pack_*.py` - replay corpus packing
- `mex/scripts/train_*.py` trainers run with a registered config from
  `configs/` (for example the dia2f replay config family)
- `mex/scripts/eval_*.py`, `bench_*.py`, `mark_metrics.py` - gates
- `mex/tests/` - unit tests for params, tasks, vocab, metrics

## Smallest venv validation

    & .\.venv\Scripts\python.exe -m unittest discover -s mex/tests -t mex

Launches additionally satisfy the documented preflight for their trainer
(config parse plus script smoke); every gate claim reads back the
registered metrics artifact.

## Data boundaries
- `data/mex/` is gitignored and regenerable by the recorded pack
  commands; `runs/mex/` holds checkpoints, adapters, and metrics; the
  global local-weights policy applies.

## Hazards
- Adapter-mount and widening edits are registered changes, not drive-bys.
- Replay corpora are deterministic and never silently recomposed; a pack
  change is a new registered experiment, never an overwrite.

## Evidence
Registered metrics artifacts under the run directory (metrics JSON, eval
outputs), selected experiment rows, and the pack commands printed by the
builder scripts.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
