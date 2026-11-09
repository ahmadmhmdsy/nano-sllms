# diacritizer/AGENTS.md - D-line scoped instructions

## Scope
The Arabic diacritization D-line under `diacritizer/`: char-level
bidirectional encoder with a 15-class diacritic head, plus its data
pipeline, training, eval, postprocessing, and gate benches.

## State and records
- source-of-truth state path: `research/experiments.yaml`
  (experiments named `diacritizer/E-xxx`) + `state/tasks.yaml`
  rows scoped to `diacritizer`.
- Narrative verdict rows live in `research/EXPERIMENTS.md`; the line
  concept and guarantees live in `diacritizer/README.md`.

## Entry points
- `diacritizer/scripts/selftest.py` - parser/labels/quarantine gates
- `diacritizer/scripts/train.py`, `train_lm.py` - registered training
- `diacritizer/scripts/prepare_data.py`, `tokenize_data.py`, `eval.py`,
  `postproc.py`, `dedup_check.py`, `bench.py` - data and evaluation
- `diacritizer/src/` - tokenizer, labels, passthrough codec

## Smallest venv validation

    & .\.venv\Scripts\python.exe diacritizer\scripts\selftest.py --phase all

Any launch additionally satisfies the launch-gate procedure; eval changes
read back the registered gate artifacts, not console output.

## Data boundaries
- `data/diac/` holds this line's data; raw subsets regenerate only by
  their recorded commands and are never deleted or rewritten as cleanup.
- Eval gates and caches rebuild on demand, never silently overwrite.

## Hazards
- D-line trainers are GPU workloads: launch decisions must follow the GPU
  rule and the launch-gate procedure first.
- The deployed anchor checkpoint stays the gold model until a registered
  verdict changes it; new-gold claims need registered artifact evidence.
- Postprocessing edits can silently change eval semantics: re-run the
  registered gate on the unchanged split after any output rewrite.

## Evidence
Registered gate-bench and metrics artifacts under the run's own directory,
cited by exact relative path; verdict rows in `research/EXPERIMENTS.md`.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
