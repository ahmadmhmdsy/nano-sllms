# flow/ — Visual Flow Editor (design)

Date: 2026-09-13. Status: DRAFT for user review — nothing built yet.
Companion mockups (throwaway design sketches, not source): `flow_design_mockup.html`, `flow_layout_candidates.html` (P1 selected).

## 0) Decision log (user-locked)

| Decision | Choice | Date |
|---|---|---|
| Scope order | **B→A**: pipeline/training-workflow graph first, model-architecture graph second | 2026-09-13 |
| App home | **Option A**: new sibling `flow/` app + minimal FastAPI backend; `viz/` and `webui/` untouched | 2026-09-13 |
| MVP layout | **P1**: palette-left (collapsible) / canvas / tabbed right inspector (Properties · Run log · Preview) | 2026-09-13 |
| Layout evolution path | **Deferred P4 elements** (floating windows, right-click radial add-menu as primary add path, full-canvas experience) promoted to future-phase target — the P1 shell must not hardcode anything that blocks that evolution | 2026-09-13 |
| Composites | **None in MVP** — flat 6-node palette only; composite/boxed subgraph nodes arrive with the model phase | 2026-09-13 |
| Final MVP question round (2026-09-13) | Q1 SFT nodes→future phase · Q2 flows/*.flow.json→tracked in git · Q3 Monitor node→deferred, expand later · Q4 Infer node→render-only shortcut handing checkpoint to webui chat · Q5 port→default 3010, user-overridable via --port / FLOW_PORT |
| Spec policy | MVP spec written exactly as decided; a "future phases" section enumerates the rest so no MVP choice forecloses later options | 2026-09-13 |
| Add-node UX | Palette drag-drop AND right-click-anywhere (canvas) from MVP | 2026-09-13 |
| Hard rule | Inherited from WEBUI_PRD.md: **the UI wraps, never reimplements** — run always resolves to the exact existing `run_custom.py` chain; auto-resume contract (PLAN §5.3) untouched; no kill button (checkpoint-aligned stop flag only, same U11 exception) | inherited |

## 1) Problem & goal

Today, configuring a training run means YAML + shell. The user wants a visual,
node-based way to author and run workflows (data processing → training → eval →
inference), Langflow/ComfyUI style, that later also lets them **author model
architectures layer-by-layer with arbitrary (non-linear) wiring**, save them as
portable artifacts, and train from them. MVP = the pipeline editor, on the
Claude/GUI-approved scope of this repo, local machine, localhost.

## 2) MVP architecture

```
flow/                      # new sibling app (Option A)
├─ package.json  pnpm      # React + TypeScript + @xyflow/react (React Flow) + fastapi client
├─ src/
│  ├─ Canvas.tsx            # React Flow canvas, edges/ports, right-click menu
│  ├─ Palette.tsx           # drag-source node library (6 pipeline nodes), collapsible
│  ├─ Inspector.tsx         # tabbed: Properties | Run log | Data preview
│  ├─ nodes/                # one component per pipeline node (typed ports)
│  ├─ store.ts              # graph + selection + run state (zustand or Context)
│  └─ api.ts                # typed fetch client
├─ server/                  # fastapi (uvicorn), launched via repo .venv python
│  ├─ app.py                # static-serves built frontend in prod; dev via pnpm dev + CORS
│  ├─ nodes.py              # node registry: 6 pipeline nodes, each = existing script contract
│  ├─ graph_schema.py       # validates .flow.json (see §4); never executes user JSON
│  ├─ runner.py             # subprocess wrapper: builds config → run_custom.py; stdout tail to API
│  └─ lock.py               # single-GPU lock policy (reuses webui's rules)
└─ flows/                   # saved graphs: marshall .flow.json (git-ignored or tracked: user call)
```

Stack rationale: a node canvas is not achievable in Gradio; `viz/` already
established React+pnpm conventions. Backend is ~150–250 lines thin glue.

## 3) Node contract (MVP palette — flat, 6 nodes)

Each node = a thin adapter over one EXISTING script; ports declare data kinds:

| Node | Source of truth | Ports (out→in) |
|---|---|---|
| Dataset | prepare_data.py / webui data path | out: raw-dir |
| Prepare | prepare_data.py | in: raw-dir; out: cleaned-dir |
| Tokenize | tokenize_data.py | in: cleaned-dir; out: shard-dir (*.bin) |
| Train | train.py via run_custom.py | in: shard-dir + config-yaml; out: checkpoint-dir; **GPU-gated** |
| Eval | eval.py | in: checkpoint-dir; out: eval_report |
| Infer | infer.py / webui chat policy | in: checkpoint-dir; **GPU-gated or CPU-fallback** |

Port type-checking (raw-dir ≠ shard-dir) is the MVP-level graph validator; full
shape/type inference belongs to the model phase. Node properties map 1:1 to the
webui's ~5 visible knobs preset/dataset/rows/epochs/LR; everything else behind an
"advanced" disclosure that edits generated YAML.

## 4) Graph file format (.flow.json)

```json
{
  "schema": "flow/0.1",
  "meta":   { "name": "...", "created_at": "...", "app_version": "..." },
  "graph": {
    "nodes": [
      { "id": "n1", "kind": "dataset|prepare|tokenize|train|eval|infer",
        "props": { "...node-specific, validated against nodes.py registry..." },
        "position": {"x": 0, "y": 0} }
    ],
    "edges": [ { "id": "e1", "from": "n1", "port": "o0", "to": "n2", "port": "i0" } ]
  }
}
```

Rules: server-side schema validation on every save/load/run; unknown kinds and
pseudo-nodes ("Monitor/Chat", dashed read-only monitor links — see mockup) are
render-only and never execute; JSON structure stable across refactors (forward-
migration hook in graph_schema.py).

## 5) Backend API (fastapi, localhost, no auth)

| Endpoint |
|---|
| GET /api/nodes — registry |
| GET /api/flows · GET/PUT /api/flows/{name} — save/load |
| POST /api/validate — graph precheck (ports, props, disk, GPU-lock) before Run |
| POST /api/run — builds configs/flow_<name>.yaml then spawn run_custom.py |
| GET /api/run/status — live tail, phase, exit; GET /api/stop — checkpoint-aligned stop per webui exception |

Single-GPU lock, no-kill policy, session-local (one run at a time) as in webui

## 6) Validation gates

- G1. Editor shell PASS: canvas, palette, inspector, save/open, zero GPU/console launched and noop.
- G2. Validate mode PASS: every bad graph (port-mismatch, missing shard input to Train before dataset) rejected with human-readable error.
- G3. Run PASS: a smoke phase graph (Dataset→Prepare→Tokenize→Train) trains till completion; restart-resume with zero flag; status shows in Run log tab.
- G4. Export PASS: graph → configs/flow_<name>.yaml equals what webui generates for the same knob settings.

## 7) Explicit non-goals (MVP)

No composites; no model-graph editing; no auth/multi-user; no remote/cloud; no
kill button; no editing of train.py/resume behavior; no React build-chain inside
webui/ or viz/.

## 7b) Final question round — outcomes applied (2026-09-13)

- SFT nodes (SFT data / SFT train as palette nodes): **future phase**; MVP keeps the 6 nodes (Train covers SFT only via preset, honestly labeled).
- flows/.flow.json files: **tracked in git** — graphs are reviewable, agent-consumable artifacts.
- Monitor node: **deferred**; the Run log inspector tab is the MVP's only feedback surface; the Monitor node (render-only readouts over runs/) returns later.
- Infer node: **render-only** — hands the selected checkpoint to the webui Chat tab (a link/launch action, no execution in flow).
- Server port: **default 3010**; user override via --port CLI flag or FLOW_PORT env.

## 8) Future phases (analytical scope) — NOT included in MVP but its choices do not foreclose these

- **F1 — Model-architecture graph editor (phase A)**: layer node library (Input, Embedding, RMSNorm, RoPE, GQA-Attention, SwiGLU-FFN, LayerStack, Residual, Head/Output), multi-input/non-linear wiring, form-based inspector per layer, shape inference + param count computed with viz/'s engine; must recreate smoke/pilot/target decoders from scratch as acceptance.
- **F2 — Portable export**: graph → configs YAML (when pattern matches repo decoders) + generated pytorch model.py for arbitrary DAGs; later HF trust_remote_code, ONNX (inference-only).
- **F3 — Composites/subgraphs**: boxed collapsed nodes that expand to sub-graphs — must be introduced into both pipeline- and model-graph catalogues predicated on schema v0.2 forward-compatible extension (graph schema already leaves kind-registry open).
- **F4 — Editor evolution toward P4**: promote P1 shell to full-canvas "floating everything" (P4): palette → right-click radial + Ctrl+K as the primary add path, floating inspector/log windows. MVP shall implement palette/inspector/log as components that can detach.
- **F5 — Execution/backend**: direct-protocol execution engine replacing subprocess- per node (Spark-like topological execution of the pipeline graph), remote agents consuming .flow.json alongside configs: other frameworks / agents get the architecture "config file" export (model file or whatever fits, as the user requested).
- **F6 — Data-level composability**: nodes can also represent generic processing cells to chain with the model phase to generalize beyond the repo's narrow 6 pipeline stages.

## 9) Risks / open questions for later

- Form-based props ↔ generated YAML fidelity must be honestly reported in the inspector's "advanced" section; run log is the single source of truth in live runs (tfevents), like webui decisions.
- pnpm/fastapi dependency budgets: 2 runtime deps deliberate and justified.
- Status endpoint is polled and coarse-grained (1–2 s tail window; fine for runners, not a streaming log).

---

## Review notes (for the user)

This doc is the conversation contract in writing. Review and say "approved" (or amend). After approval the next steps are, in order: implementation plan via the writing-plans skill, then code — never before.