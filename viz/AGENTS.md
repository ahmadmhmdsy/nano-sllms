# viz/AGENTS.md - architecture explorer scoped instructions

## Scope
The config-driven React model-architecture explorer under `viz/`:
multi-model decoder visualizer, param/depth/simulation engine, the
architecture-kind registry, and model configs under `models/`.

## State and records
- source-of-truth state path: `research/experiments.yaml` +
  `state/tasks.yaml` (scope `viz`).
- Line details and honesty rules: `viz/README.md`; plan under
  `docs/plans/`.

## Entry points
- `viz/src/` (engine, schema, registry, kinds) - all views derive from
  YAML model configs under `models/`; adding a model is one YAML file
  with verified `honesty.sources`.

## Smallest venv validation (no venv needed: pnpm only)

    cd viz
    pnpm test

plus `pnpm build` for any shipped change plus `pnpm doctor` for the
schema/arithmetic/source-citation gate; a console render alone is not a
verdict.

## Data boundaries
- Inputs are the YAML model configs plus recorded artifacts; weights are
  never downloaded for rendering and tensor payloads are never loaded.

## Hazards
- Param numbers are engine-computed, never hand-written in UI text; a new
  architecture kind lands in the declared engine branch points only.
- Never special-case a model outside those branch points; broaden the
  schema only with the heal-the-schema entry in its own review.

## Evidence
Passing `pnpm test` with its anchors, a clean `pnpm build`, and the
cited config sources under `models/*.yaml` (honesty.sources).

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
