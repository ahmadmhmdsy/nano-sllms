# F2 Model-Architecture Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** A second "Model" tab in the flow editor: drag layer nodes (dense GQA library), wire a non-linear tensor graph, live param count + shape inference — acceptance: recreate smoke/pilot/target decoders with exact param totals.

**Architecture:** Reuses the pipeline editor's proven pattern (registry-driven, zero kind-conditionals). Frontend-only layer registry with typed tensor ports; param math ported verbatim from `viz/src/engine/params.ts` with pinned anchors; god-store split as step 0; backend gains only a `.modelgraph.json` atomic-save endpoint.

**Tech Stack:** React + ReactFlow (@xyflow/react) + zustand (existing); FastAPI (existing); Vitest; no new runtime deps.

**Design doc:** `docs/plans/2026-09-13-model-architecture-editor-design.md` (user-approved).

**Acceptance anchors (dense GQA formula `pTotalDense`; pre-computed):**
- smoke (L=4 d=256 h=4 kv=2 ffn=1024 vocab=32768, tied): **12,323,072**
- pilot (L=12 d=768 h=12 kv=4 ffn=2048 vocab=32768, tied): **100,682,496**
- target (L=16 d=1024 h=16 kv=4 ffn=3072 vocab=32768, tied): **226,526,208** (equals viz nano anchor)
- SmolLM2-135M (L=30 d=576 h=9 kv=3 ffn=1536 vocab=49152, tied): **134,515,008** (viz anchor, port sanity)

**Environment rules (AGENTS.md):** venv python = `& .\.venv\Scripts\python.exe`; frontend `cd flow; pnpm ...`; no GPU contact.

---

## File structure

- Create `flow/src/model/paramMath.ts` (+`paramMath.test.ts`) — dense param math ported verbatim from `viz/src/engine/params.ts`.
- Create `flow/src/stores/{graphStore,registryStore,uiStore,runStore}.ts`; `flow/src/store.ts` becomes a re-export shim only.
- Create `flow/src/model/registry.ts` — `LayerSpec` (tensor ports with multiplicity, PropSpec, `inferShape`, `paramCount`), `LAYER_REGISTRY` — plus `flow/src/model/layers/*.ts`: input, embedding, rmsnorm, rope, gqaAttention, swigluFfn, layerStack, residual, lmHead.
- Create `flow/src/model/graphWalk.ts` — Kahn topological shape inference + param rollup + errors.
- Create `flow/src/model/{ModelCanvas,ModelNode,ModelInspector,ModelPalette}.tsx`; modify `App.tsx` (mode switch), `api.ts` (model APIs).
- Modify `flow/server/flows.py` (extract `atomic_write(path, text)`), `flow/server/app.py` (`GET/POST /api/models`).
- Create `flow/tests/server/test_models_api.py`; create `flow/models/{smoke,pilot,target}.modelgraph.json` + committed totals test.
- Modify docs: `flow/README.md`, `HANDOFF.md`, `MEMORY.md`, `TASKS.md`.

**.modelgraph.json format (model/0.1):** `{"format":"model/0.1","name":str,"nodes":[{id,kind,props,position}],"edges":[{id,from,to,fromPort,toPort}],"meta":{}}` — backward-open like flow/0.1.

**layerStack decision (locked):** layerStack counts ONLY the outer norms `pNorms((2N+1)*d)`; nested body-graph machinery is an explicit non-goal this increment — the smoke/pilot/target graphs express each layer explicitly (N stack nodes) instead.

---

### Task 0: param math port (pure, zero deps)

**Files:** Create `flow/src/model/paramMath.ts`, `flow/src/model/paramMath.test.ts`.

- [x] Failing tests first: the 4 anchor totals above equal `pTotalDense` with each model's dims (exports: `headDim, kvDim, pEmbed, pAttentionDense, pFFNDense, pNorms, pTotalDense`, local `DenseShape {layers,d,heads,kv,ffn,vocab}`).
- [x] Implement: port the dense section of `viz/src/engine/params.ts` verbatim (formulas + comments), no viz imports.
- [x] `cd flow; pnpm test` — new green, existing 92 green. Commit: `feat(model): port dense param math from viz engine with anchor tests`.

### Task 1: store.ts split (behavior-preserving)

**Files:** Create `flow/src/stores/{graphStore,registryStore,uiStore,runStore}.ts`; rewrite `store.ts` as shim.

- [x] Read store.ts fully; partition state+actions into 4 zustand stores; mappers into graphStore or `stores/mappers.ts`. Compose cross-store wiring via subscriptions.
- [x] `store.ts` = pure re-export, all previous export names unchanged; consumers untouched.
- [x] `pnpm test` 92/92 (never fix a failing test — fix the split); `pnpm build` green. Commit: `refactor: split flow god-store into composable stores (behavior-preserving)`.

### Task 2: layer-node registry + kind modules

**Files:** Create `flow/src/model/registry.ts`, `flow/src/model/layers/*.ts`, `flow/src/model/registry.test.ts`.

Spec shape (locked):
```ts
interface TensorPort { name: string; label: string }
interface PropSpec { name: string; kind: "int"|"float"|"bool"|"enum"; options?: string[]; default?: number|boolean; min?: number; max?: number }
interface Shape { b?: number; s?: number; d: number }          // example: token-ids shape d means "vocab index"
interface LayerSpec {
  kind: string; label: string; summary: string;
  inputs: TensorPort[];  // 0 (input) .. 2 (attention q+kv, residual stream+bypass)
  outputs: TensorPort[]; // exactly 1 everywhere
  props: PropSpec[];
  inferShape(props: Record<string, unknown>, inputs: Shape[]): Shape;
  paramCount(props: Record<string, unknown>, out: Shape): number;
}
```

Kind behaviors (locked):
- `input`: 0 inputs; props vocab_size(int), ctx(int); shape {s: ctx, d: vocab_size-as-index}; params 0.
- `embedding`: 1 in; props vocab_size, d; params vocab*d. (Concrete case: input -> embedding with the same vocab_size.)
- `rmsnorm`: 1 in/1 out (d passthrough); params d.
- `rope`: 1 in/1 out; params 0; propagate {s,d}.
- `gqaAttention`: 2 in (q-stream, kv-stream); props heads, kv_heads (d from input shape); params pAttentionDense; out d = q-input d.
- `swigluFfn`: 1 in/1 out; prop ffn; params pFFNDense.
- `residual`: 2 in (stream, bypass); requires d equality on bypass==stream or error; params 0.
- `lmHead`: 1 in; props vocab_size, tied(bool); params tied ? 0 : vocab*d.
- Example smoke body per layer inside the graph: rmsnorm -> rope(q,kv) split -> gqaAttention(q=rope-out-with-heads-prop, kv=rope-kv-out) -> residual(stream, attn-out) is EXPRESSED WITH TWO rope calls if the port model needs it, else attention takes 3 inputs (q, k, v). LOCKED: gqaAttention has THREE inputs (q, k, v) to keep wiring honest; all carry the pre-attn d.

- [x] Failing registry tests (uniqueness, props sanity, pure paramCount) + per-layer counts cross-check the Task-0 anchors for repo dims. Implement. `pnpm test` green. Commit: `feat(model): layer-node registry with dense GQA kinds`.

### Task 3: topological shape/param walk

**Files:** Create `flow/src/model/graphWalk.ts`, `graphWalk.test.ts`.

- [x] Failing tests: (a) linear body chain shapes correct; (b) cycle → one honest error; (c) wrong-port-count → per-node error; (d) a full mini-graph totals exactly the smoke anchor digest (can use reduced subset); (e) residual d mismatch error.
- [x] Implement `walkModelGraph(graph, registry) => {shapes: Map<nodeId,Shape>, params: Record<nodeId,number>, total, errors: {nodeId, message}[]}` — Kahn walk, per-node errors never abort the walk.
- [x] Green. Commit: `feat(model): topological shape inference + param rollup`.

### Task 4: backend persistence

**Files:** Modify `flow/server/flows.py` (extract `atomic_write`), `flow/server/app.py` (`GET /api/models`, `POST /api/models/<name>`), Create `flow/tests/server/test_models_api.py`, `flow/models/.gitkeep`.

- [x] Refactor flows.py to use the helper; existing atomic-save test must stay green unedited.
- [x] TDD API tests: save OK (atomic, no .tmp residue — mirror test_flows.py), list OK, bad format/unsanitized name → 400 honest message.
- [x] Server suite: `& .\.venv\Scripts\python.exe -m unittest discover -s flow/tests -t flow` green (133 + ~4). Commit: `feat(server): model-graph save/list endpoints with atomic writes`.

### Task 5: Model mode UI

**Files:** Create `flow/src/model/{ModelCanvas,ModelNode,ModelInspector,ModelPalette}.tsx`; modify `App.tsx` (segmented "Pipeline | Model" mode), `api.ts`; NO kind conditionals in render logic (registry-driven like PipelineNode).

- [x] Palette click-to-add from registry; ModelNode with multi-port handles + param badge + red error badge; Inspector props form from spec with live shape/param readout; header strip with total + error list; save/open via /api/models mirroring flows; empty-state hint pointing at committed examples.
- [x] `pnpm build` + `pnpm test` green. Commit: `feat(model): Model tab UI canvas/inspector/palette`.

### Task 6: acceptance graphs (hard gate)

**Files:** Create `flow/models/{smoke,pilot,target}.modelgraph.json`; Create committed totals test (`flow/tests/test_model_graphs.py` via a python-side regex-free JSON load + a vitest walker test is SECTION 3's job — python test asserts totals using a minimal mirroring formula ONLY IF the walker can't run in Python; prefer: totals test in vitest where the TS engine lives, exact-integer equality, e.g. `expect(total).toBe(12323072)`).

- [x] Smoke graph first: input->embedding->per-layer chain (L=4x [rmsnorm, rope, gqa q, gqa k, gqa v, swigluFfn, residual]) ->rmsnorm? per src/model.py topology — READ `src/model.py` wiring before authoring; no invented architecture.
- [x] Vitest totals assertions for all three graphs FAIL if wiring is off — fix wiring, never anchors.
- [x] Browser drill (agent-browser, ONE chained pwsh call): Model tab → open smoke graph → total shows 12,323,072 → set ffn 1024→2048 → expected total 13,109,504 → set back to 1024. Evidence logged.
- [x] Commit: `feat(model): committed smoke/pilot/target decoder graphs (param totals locked)`.

### Task 7: verification + docs sweep

- [x] All gates green: vitest 92+Δ, server unittest 133+Δ, `pnpm build`.
- [x] Update `flow/README.md` (Model mode section), `HANDOFF.md` (newest-first), `MEMORY.md`, `TASKS.md`, and this plan's checkboxes. Commit: `docs: F2 model editor implementation notes`.

## Deferred (explicit, recorded)

- Shared pnpm workspace package for param math (touches viz imports — after both suites green).
- MoE/hybrid kinds, model.py/YAML export, nested layerStack bodies, remote execution, composite nodes.
