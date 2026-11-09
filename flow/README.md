# flow — Visual Flow Editor (MVP)

Draw your training pipeline (Dataset -> Prepare -> Tokenize -> Train -> Eval -> Infer)
on a node canvas; validate it; run it through the repo's existing scripts.
Spec: docs/plans/2026-09-13-flow-editor-design.md · Plan: docs/plans/2026-09-13-flow-editor-mvp-plan.md

## Run

```powershell
# Backend (FastAPI, serves the built frontend in prod)
& .\.venv\Scripts\python.exe -m flow.server.app --port 3010        # prod (static flow/dist)
& .\.venv\Scripts\python.exe -m flow.server.app --dev --port 3010   # dev: CORS for Vite

# Frontend
cd flow
pnpm install          # first time: also "pnpm approve-builds esbuild" (pnpm 12 gate)
pnpm dev              # Vite on http://localhost:5174, proxies /api to 3010
pnpm build            # tsc --noEmit + vite build -> flow/dist (prod static)
pnpm test             # vitest
```

Backend unit tests (stdlib unittest, no pytest in venv):

```powershell
& .\.venv\Scripts\python.exe -W error::SyntaxWarning -m unittest discover -s flow/tests/server -t . -v
```

Port: default 3010 (--port flag wins, then FLOW_PORT env). Never collides with the DSH GUI (3080), webui (7860), viz (5173/5174 dev).

## What is real vs placeholder

- Run/Validate/Stop/Save/Open ARE wired to the backend (REST on 3010).
- Run launches the EXISTING pipeline via scripts/run_custom.py (UI wraps, never reimplements).
- Single-GPU job slot; NO kill button — Stop writes the checkpoint-aligned STOP flag (src/stop.py contract); auto-resume contract (PLAN 5.3) untouched.
- Validate runs graph_schema + registry props + a config_gen dry-run in a tempfile (never writes repo files).
- Preview tab shows the last-validated graph summary (honest label; not the compiled YAML).
- Infer node is render-only: hands the checkpoint to the webui Chat tab (http://127.0.0.1:7860), manual instructions always shown.

## Editing the graph (M1.1 / M1.3 / M2.3 / M5.5)

### Undo / redo (M1.1)

The toolbar has **Undo** / **Redo** buttons (disabled when there is nothing to
undo/redo) and the canvas honors **Ctrl+Z** (undo) and **Ctrl+Shift+Z** /
**Ctrl+Y** (redo). The history is a command-pattern stack
(`flow/src/stores/historyStore.ts`): every commit action (add node, connect,
delete, props/label edit, setGraph) pushes the PRE-mutation graph before
applying the change, so undo restores exactly that snapshot. Transient
changes (position drags, selection) never push history — otherwise every
mouse-move during a drag would create an undo step. A new action clears the
redo stack (standard editor semantics: no redo after a new action), and the
stack is capped at 100 entries. Undo/redo also clears the canvas projection
and the last validation result, since the graph they described no longer
exists.

### Auto-save draft (localStorage)

The graph is auto-saved to the browser's localStorage (key `flow:draft`)
2 seconds after every change (debounced; skipped while a modal is open). On
the next load a recovered draft toasts "Unsaved draft recovered — Save to
keep it (or Clear draft in Save modal)"; the Save modal's **Clear draft**
button drops it. This is a LOCAL draft only — the server copy still needs
**Save** (PUT /api/flows/{name}); the draft survives reloads but never
leaves the browser.

### Export / import (M2.3)

- **Export** serializes the current graph to a pretty-printed `flow/0.1`
  document and downloads it as `<slug>.flow.json` (`untitled.flow.json`
  when the graph was never saved). The bytes are exactly what Save/PUT
  would send — the pure `serializeFlow` reuses the store's document
  builder, so the on-disk document is byte-identical.
- **Import** opens a file picker (`.flow.json`), schema-validates the text
  client-side (`parseFlowDocument`: JSON parse, `schema: "flow/0.1"` tag,
  meta/graph shape, per-node/per-edge field checks) and loads a valid
  document through setGraph (which pushes undo history). A malformed or
  wrong-schema file toasts the exact parser error and loads nothing.

Both live in `flow/src/stores/flowDocument.ts` as pure, unit-tested
functions; only the Blob download and the file input are browser-only.

### Keyboard shortcuts

| keys | action |
|---|---|
| Ctrl+Z | undo |
| Ctrl+Shift+Z / Ctrl+Y | redo |
| Delete / Backspace | delete selected nodes/edges (pushes undo history; inert while typing in a form field) |
| Ctrl+K | toggle the node-search modal |
| Arrow keys | move the node selection to the nearest node in that direction (M5.5) |
| Tab / Shift+Tab | cycle focus through the node connection handles (M5.5) |
| Escape | close the add menu / search modal / dialogs |

Undo/redo and Delete are ignored while focus is in an input/textarea/select.

## Node contract (MVP palette: 6 kinds)

| node | in -> out | gate |
|---|---|---|
| dataset | - -> raw-dir | none |
| prepare | raw-dir -> cleaned-dir | none |
| tokenize | cleaned-dir -> shard-dir | none |
| train | shard-dir -> ckpt-dir | gpu |
| eval | ckpt-dir -> report | gpu |
| infer | ckpt-dir -> (render-only) | gpu/cpu |

Only linear chains map to YAML (branching graphs are future F5 execution).
Graphs persist as flow/flows/<slug>.flow.json (git-tracked) — saves are
atomic (temp file + fsync + os.replace): a reader never sees a torn file.

### Node definitions (2026-09-13 promotion refactor)

Each built-in kind is a behavior-carrying `NodeDefinition` subclass
(`flow/server/nodes/builtin/*.py`), not a dict entry. One node owns every
fact about itself:

- `spec` — ports, `PropSpec`s (type/required/widget/help), gate,
  `label_semantic` (dataset: `hf-dataset-name`), `features`
  (infer: `webui-chat-button`) — this is what `/api/nodes` snapshots and
  what the frontend renders; the UI has ZERO `kind ===` conditionals.
- `required_upstream` — structural dependency (`train` needs `shard-dir`
  from `tokenize`, …). The linear-chain order in `config_gen` is DERIVED
  topologically from these declarations; there is no hardcoded CHAIN list.
- `validate_semantic(node)` — per-node knob errors (steps missing, dataset
  name shape, rows cap MAX_ROWS, preset ctx choices) with the historic
  message wording preserved.
- `build_section(ctx)` — the config keys the node contributes
  (prepare owns `data` details, tokenize owns `tokenizer`, train owns
  `model/train/eval`; PRESETS/LR_PRESETS live in `train.py`,
  TOKENIZER_NAME/SHARD_TOKENS in `tokenize.py`, MAX_ROWS in `prepare.py`).

Registering a new kind = one module under `builtin/` exporting a
`NodeDefinition`; the registry (`flow/server/nodes/__init__.py`) derives
everything else. Plugin discovery for user-defined nodes is a possible next
step (planned, not implemented). Golden files under `flow/tests/goldens/`
lock config_gen's output so refactors cannot silently change behavior.

## Example workflows (Open picker)

- `example-linear-smoke` — a fully-knobbed linear chain (dataset label,
  prepare/tokenize knobs, train steps + presets); Validate returns the OK
  toast and the Preview tab shows the compiled config summary.
- `example-linear-untuned` — same topology with NO knobs; Validate walks
  you through each missing knob honestly. Use as a teaching/start flow.
- `example-infer-handoff` — the train chain plus an eval branch AND an
  infer node; Validate honestly refuses (branching = future F5) while the
  infer node's handoff dialog works (manual webui launch instructions).

## Model mode (F2, 2026-09-13) — model-architecture editor

The top-level **Pipeline | Model** switch opens a second editor: drag dense-GQA
layer nodes (Input, Embedding, RMSNorm, RoPE, GQA-Attention, SwiGLU-FFN,
Residual, LM-Head, LayerStack), wire tensor edges between named ports (multi-input
honest for attention q/k/v), and read live shape inference + param counts from
the shared param-math package (`packages/param-math`, re-exported by
`flow/src/model/paramMath.ts`; anchors pinned in tests:
nano 226,526,208 · SmolLM2 134,515,008).

- Everything is registry-driven (`flow/src/model/registry.ts`): kinds, ports,
  props, defaults, param math — **zero kind conditionals** in render/inspector,
  same interlock rule as the pipeline side.
- Shape/param rollup walks the DAG topologically; wiring mistakes (cycle,
  missing/duplicate input, port-name mismatch, shape mismatch) show as per-node
  red badges with the exact reason — never a silent number.
- Graphs persist as `flow/models/<slug>.modelgraph.json` (format `model/0.1`,
  git-tracked) through `GET/POST /api/models[/{name}]` — same atomic writer as
  flow saves (`flows.atomic_write`).
- Examples (Open picker): `smoke`, `pilot`, `target` — the repo's actual
  decoders rebuilt as graphs; their param totals are locked to exact integers
  by `flow/src/model/acceptance.test.ts` (smoke 12,323,072 · pilot 100,682,496 ·
  target 226,526,208 = pTotalDense).
- LayerStack counts ONLY the outer norms ((2N+1)·d); inner layer params come
  from the explicitly wired body nodes — no hidden nested-graph magic.
- NOT in this increment: model.py/YAML export, MoE/hybrid kinds, execution,
  nested layerStack bodies.

## Future phases (not in MVP)

Portable architecture export (generated model.py / trust_remote_code / ONNX; F2.next),
composite/subgraph nodes, branching pipeline execution. Selections and layout choices
were deliberately left open — see the design doc's future-phase section. (The shared
param-math package with `viz/` shipped early — see `packages/param-math`.)
