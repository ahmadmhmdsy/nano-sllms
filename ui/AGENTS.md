# ui/AGENTS.md - U-line scoped instructions

## Scope
The U-line chain surface under `ui/`: prompt-to-UI-chain training data,
corpus building, pico-scale training, validity gates, and dataset tooling.
The Gradio shell itself is webui (see `webui/AGENTS.md`).

## State and records
- source-of-truth state path: `research/experiments.yaml`
  (`ui/E-xxx`) + `state/tasks.yaml` (scope `ui`).
- Verdict rows in `research/EXPERIMENTS.md`; line details in
  `ui/README.md`; dataset provenance notes in `ui/data/*/` READMEs.

## Entry points
- `ui/src/`: `catalog.py`, `chain.py`, `validate.py`, `generator.py`
  (round-trip asserted equal by the test suite)
- `ui/scripts/build_corpus.py`, `prepare_u1.py`, `eval_u1_gate.py`
- `ui/scripts/build_space_bunny_50k.py` (side-data builder)

## Smallest venv validation

    & .\.venv\Scripts\python.exe -m unittest discover -s ui/tests -t ui

Gate or corpus claims additionally parse the registered metrics artifacts
under the run directory (`runs/u1_full` et al), never console lines.

## Data boundaries
- `ui/data/` is the corpus/store, partly untracked and large;
  regeneration only by recorded builder commands with the registered
  seed and per-archetype caps; never deleted as cleanup.

## Hazards
- Validity/gate claims are hard-fail: parse the registered gate artifact
  right after the run; chain-equivalence claims need the round-trip test.
- Do not mix U-line scope with Gradio shell work
  (`rules/components/webui.md` owns that boundary).

## Evidence
Pre-registered gate metrics and validity reports under the registered run
directory, cited by exact relative path; verdict rows in
`research/EXPERIMENTS.md`; corpus provenance notes under `ui/data/`.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
