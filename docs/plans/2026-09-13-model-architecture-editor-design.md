# F2 (increment A) — Model-architecture editor: design

Date: 2026-09-13 · Status: user-approved (A/A/A scoping: flow-app tab · editor+engine first · store split first)

## Problem

The user's original wishlist: drag-and-drop *model architectures*, not just pipelines.
This is the "B/F2 model editor" selected from the retro menu
(`.superpowers/sdd/flow/retro-2026-09-13-flow-mvp.md`), based on the
flow-editor design doc §8 F1 sketch and repo `viz/` explorer engine.

## Decisions (confirmed by user)

1. Lives as a **second mode/tab inside the flow app** (shared canvas machinery, one build).
2. First increment = **editor + param/shape engine** (no export).
3. **store.ts split happens as step 0** (retro open issue #6 prerequisite).

## Scope

### In

- **Step 0 — store split**: `flow/src/store.ts` (685 lines) → composable zustand stores
  under `flow/src/stores/`: `graphStore` (domain nodes/edges + doc persistence),
  `registryStore` (snapshot + fallback), `uiStore` (tabs/selection/inspector),
  `runStore`. `store.ts` becomes a thin re-export; zero behavior change; all
  existing vitest stay green before any further change.
- **Model mode**: second ReactFlow canvas; separate document kind
  `.modelgraph.json` (format `model/0.1`) saved via new `POST /api/models`
  reusing the atomic-save helper from `flows.py` (`flows.save()`).
- **Layer-node registry** (same pattern as the pipeline registry, T3 interlock):
  `flow/src/model/` modules per kind — `input`, `embedding`, `rmsnorm`, `rope`,
  `gqaAttention`, `swigluFfn`, `layerStack` (repeat-N), `residual`, `lmHead` —
  each declaring tensor-contract ports, form props, and behavior:
  `inferShape(inputShapes, props)` + `paramCount(props, shape)`.
- **Shape inference**: topological DAG walk; shape errors render as per-node red
  badges (honesty surface, same style as pipeline gate reasons).
- **Param math**: dense functions from `viz/src/engine/params.ts` ported verbatim
  into `flow/src/model/paramMath.ts`, anchors pinned in tests:
  nano 226,526,208 · SmolLM2 134,515,008. Norms counted exactly once ((2L+1)·d).
- **Dense GQA kinds only** — exactly what the repo decoders need.
- **Acceptance**: recreate smoke / pilot / target decoders from scratch as
  `.modelgraph.json`; editor-computed totals must match each config's computed
  param counts exactly. Drill: knob edit → live counter update; broken wiring →
  honest shape error.

### Out (non-goals)

- No `model.py` / YAML / ONNX export (F2.next).
- No execution, no composites/subgraphs, no MoE/hybrid kinds.
- No Python engine; no changes to train.py; no composites in pipeline editor.

## Constraints honored

- Zero kind-`===` conditionals in render/inspector path (T3 interlock).
- Flow format `flow/0.1`, API, and pipeline nodes untouched.
- Single source of param math: shared workspace-package extraction deferred —
  recorded, not silently dropped.

## Test/verification delta

~15 new vitest (layer registry, shape walk, paramMath anchors); existing 92
vitest + 133 unittest stay green. One browser drill. Docs sweep (flow README,
HANDOFF, MEMORY, TASKS) closes the phase.

## Risks

- LayerStack repeat-N must keep param/shape math identical to `(2L+1)·d`
  accounting — the anchors test guards this.
- Multi-input ports (attention: q/kv) need a port-multiplicity rule the pipeline
  PortSpec lacks — model registry gets its own port spec, frontend-only for now.
- store split touches every consumer — re-export shim + full test suite gates it.
