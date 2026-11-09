# webui/AGENTS.md - Gradio shell scoped instructions

## Scope
The Gradio web UI shell under `webui/` (spec `WEBUI_PRD.md`,
milestone U-lines through U5): dashboard, monitor, chat, training
launcher, and the Model tab (explorer plus training simulator).

## State and records
- source-of-truth state path: `research/experiments.yaml` +
  `state/tasks.yaml` (scope `webui`); milestone evidence per
  `TASKS.md` and drill evidence files.
- Product spec: `WEBUI_PRD.md` (user-gated changes only).

## Entry points
- `webui/app.py` launch with explicit port and --no-browser for headless
  checks; --lan/--auth/--webhook options are documented in
  `README.md`.
- `webui/artifacts.py`, `explorer.py`, `model_tab.py`,
  `simulator.py` - read-only introspection and replay modules

## Smallest venv validation

    & .\.venv\Scripts\python.exe webui\app.py --port 7861 --no-browser

Run it headless until a clean boot, then close it; interaction or
Model-tab claims additionally run a real browser drill of the touched tab
and record the drill evidence path, never just a boot log.

## Data boundaries
- The shell owns no training data; checkpoint/config access is read-only
  through its helper modules; run records stay in the registries.

## Hazards
- Gradio validation errors surface only at interaction time: exercise
  every touched dropdown or tab change once for real; a clean boot alone
  proves nothing about interaction behavior.
- A webui process can hold a stale CUDA context after model unload; only
  clean up server processes this session started, following the GPU rule.

## Evidence
Browser-drill evidence files saved alongside the run records, TASKS.md
milestone rows with their validation labels, and the app boot log.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
