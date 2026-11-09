# langid/AGENTS.md - Tongue-analogue scoped instructions

## Scope
Text language-identification line under `langid/` (Tongue-analogue,
desert-ant-recreation DA-line): feature and model training, eval tasks
(langid, topic, ner, emo, schemer, rank), and exports.

## State and records
- source-of-truth state path: `research/experiments.yaml` +
  `state/tasks.yaml` (scope `langid`).
- Gate rows pre-registered in the D/A-plan
  (see `docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md`)
  and in the experiment registry.

## Entry points
- `langid/scripts/train.py` and the train_*.py trainers (CPU-only)
- `langid/scripts/eval_*.py` - next-gate validation scripts
- `langid/scripts/prepare_data.py`, `fleurs_download.py` (network step)
- `langid/src/`, `langid/data/`

## Smallest venv validation

    & .\.venv\Scripts\python.exe -m unittest discover -s langid/tests -t langid

Any gate claim additionally parses the registered eval artifact for that
run; a claim is the artifact read-back, never console-only.

## Data boundaries
- Downloaded eval corpora (for example FLEURS, TaoBench) land only under
  `langid/data/` or their recorded cache location; downloads are network
  steps and need the launch-gate procedure when not registered.

## Hazards
- CPU-only by design: never trade it onto the GPU.
- Gates pre-register before any run; renaming or changing an eval script
  without a registration is not valid.

## Evidence
The registered eval artifacts produced this run (exact relative path),
the pre-registered gate rows of the design plan, and the verdict rows in
`research/EXPERIMENTS.md`.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
