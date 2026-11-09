# laya/AGENTS.md - L-line scoped instructions

## Scope
The Laya decision-model line (L1/L2/L3) under `laya/`: typed-decision
scoring heads, data builders, trainers, eval panels, preflight and
param-count gates, and corpus scaling (`laya/scripts/build_corpus_l3.py`).

## State and records
- source-of-truth state path: `research/experiments.yaml`
  (`laya/E-xxx`) + `state/tasks.yaml` (scope `laya`).
- Verdict rows in `research/EXPERIMENTS.md`; study reports and recorded
  rungs under `research/` (`2026-09-2*_laya_*`).

## Entry points
- `laya/scripts/preflight_check.py --script <trainer>.py --config <cfg>.yaml`
  (required gate before ANY unattended long launch)
- `laya/scripts/param_count.py` (pre-registration param artifact)
- `laya/scripts/train_l1.py`, `train_l2.py`, `train_l3.py`,
  `train_l3_sym.py` trainers; corpus builders (`build_data.py`,
  `build_corpus_l3.py`)
- `laya/scripts/eval_l*.py`, `eval_phish_sym.py`, `eval_public_choice_sym.py`
  and panel evaluators

## Smallest venv validation

    & .\.venv\Scripts\python.exe laya\scripts\param_count.py --help

followed, for any launch decision, by the full preflight gate above plus
the rung's registered eval script; a param or report claim cites the
param-count or eval-report artifact, never console memory.

## Data boundaries
- Built corpora and packed datasets under `data/` regenerate only by
  their recorded builder command; `runs/laya/` holds checkpoints and
  logs; weights stay local per repo policy.

## Hazards
- Every long launch is user-gated and must pass preflight first; resume
  is the exact re-run command with zero flags.
- Sym/phish/registered-panel gates are pre-registered; a new claim
  without a pre-registration is not a result.

## Evidence
Eval-track artifacts (`runs/laya/*/eval_report.json` and console logs
read back against registered gate values), preflight outputs, param-count
lines, and the verdict rows in `research/EXPERIMENTS.md`.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
