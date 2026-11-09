# flow/ Visual Flow Editor — MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build the MVP pipeline graph editor (flow/) — the user draws Dataset→Prepare→Tokenize→Train workflows on a node canvas and runs them through the repo's existing scripts, per the approved design docs/plans/2026-09-13-flow-editor-design.md.

**Architecture:** React 18 + @xyflow/react canvas frontend (pnpm, mirrors viz/ conventions) served by a thin fastapi/uvicorn backend that validates graphs, maps them to the repo's existing config YAML, and launches run_custom.py as a subprocess. The UI wraps, never reimplements (WEBUI_PRD.md hard rule). GPU single-job lock + no-kill policy inherited.

**Tech Stack:** React 18, TypeScript, Vite, @xyflow/react, zustand; Python 3.12 (.venv), fastapi, uvicorn, pyyaml (already in venv); stdlib unittest (pytest NOT installed in venv — verified 2026-09-13).

**Spec:** docs/plans/2026-09-13-flow-editor-design.md (the contract). Repo doc-map/commit rules in AGENTS.md apply.

---

## File structure (locked)

```
flow/
├─ package.json / vite.config.ts / tsconfig.json / index.html / .gitignore[(node_modules, dist)]
├─ src/
│  ├─ main.tsx                    # React entry
│  ├─ App.tsx                     # P1 layout shell (collapsible palette, canvas, tabbed inspector)
│  ├─ FlowCanvas.tsx              # @xyflow/react canvas + right-click add menu + drag-drop
│  ├─ Palette.tsx                 # collapsible left panel, drag sources
│  ├─ Inspector.tsx               # tabbed right panel: Properties | Run log | Preview
│  ├─ nodes/PipelineNode.tsx      # node renderer (typed ports, run badge)
│  ├─ store.ts                    # zustand: graph, selection, inspector tab, run status
│  └─ api.ts                      # typed fetch client
├─ server/
│  ├─ app.py                      # fastapi app; static-serves ../dist in prod
│  ├─ graph_schema.py             # .flow.json schema + validation (pure functions)
│  ├─ nodes.py                    # 6-node registry (kinds, ports, props, script contract)
│  ├─ flows.py                    # flows/ dir store (git-tracked)
│  ├─ config_gen.py               # graph → configs/flow_<name>.yaml
│  └─ runner.py                   # subprocess run_custom.py + status + checkpoint-stop + GPU lock
├─ tests/server/test_*.py         # stdlib unittest over pure functions
├─ src tests inline via vitest    # flow/schema sanity tests
└─ flows/                         # saved .flow.json (git-tracked, Q2)
```

**Dependency decisions:** frontend adds exactly @xyflow/react + zustand to the viz-style baseline. Backend adds fastapi + uvicorn to the venv via uv pip install (record in ENVIRONMENT.md §command-log on install). Port: default 3010, overridable --port / FLOW_PORT.

**GPU safety rule for all tasks:** launch nothing GPU-touching while a train run is live (AGENTS §4; gates G3/G4 coordinate with TASKS status).

---

## Task 0: Working-tree safety check

- [ ] Step 1: Run `git status --porcelain`. If uncommitted user changes exist (113 dirty lines observed 2026-09-13), REPORT them to the user and do not commit/stash anything that is not ours. Proceed by committing only flow/ files we create.
- [ ] Step 2: Record result in the plan's execution log table at the bottom of this file.

## Task 1: Frontend scaffold — P1 shell compiles and serves

**Files:** Create flow/package.json, flow/vite.config.ts, flow/tsconfig.json, flow/index.html, flow/src/main.tsx, flow/src/App.tsx, flow/src/store.ts, flow/src/api.ts, flow/.gitignore.

- [ ] Step 1: Write package.json (name "flow"; deps react ^18.3.1, react-dom ^18.3.1, @xyflow/react ^12, zustand ^5; devDeps typescript ^5.6.3, vite ^5.4.11, @vitejs/plugin-react ^4.3.4, vitest ^2.1.8, @types/react, @types/react-dom, @types/node; scripts: dev (vite), build (tsc --noEmit && vite build), build=tsc --noEmit && vite build --outDir dist, test=vitest run; "type":"module").
- [ ] Step 2: Vite config mirrors viz/: react plugin, dev proxy `/api` → http://127.0.0.1:3010.
- [ ] Step 3: tsconfig strict, jsx react-jsx, bundler resolution (copy viz/tsconfig.json options verbatim).
- [ ] Step 4: index.html mounts #root; main.tsx renders <App/>.
- [ ] Step 5: App.tsx P1 shell: <Palette/> <FlowCanvas/> <Inspector/> in flex; each side collapsible via boolean in store. FlowCanvas minimal xyflow here (background, no nodes); Inspector renders three tabs, switch via store.state.inspectorTab.
- [ ] Step 6: store.ts: zustand create with {graph:{nodes:[],edges:[]}, selection:null, inspectorTab:"properties", runStatus:null, } + setters.
- [ ] Step 7: Run `cd flow; pnpm install; pnpm dev` → PASS: http://localhost:5174 renders shell.
- [ ] Step 8: `pnpm build` → PASS: typecheck + dist/.
- [ ] Step 9: Commit: `git add flow/ && git commit -m "flow: scaffold P1 editor shell"`.

## Task 2: Graph schema + validation (backend core)

**Files:** Create server/graph_schema.py, tests/server/test_graph_schema.py, tests/server/__init__.py (+ server/__init__.py).

- [ ] Step 1: Write failing test (unittest):

```python
import unittest
from server.graph_schema import validate, no placeholders — the test file defines: SAFE = 'flow/0.1'; and asserts: valid graph → []; empty graph → error; missing meta.name → error; unknown kind → error; edge to unknown node id → error; duplicate node id → error; edge connecting mismatched port names → error.
```
(complete assertions inside file; run, verify lists each reason).
- [ ] Step 2: Implement graph_schema.py: validate(g: dict) -> list[str]:
  - schema exact "flow/0.1"
  - meta.name non-empty
  - graph.nodes: dict with id/kind/props/position; kind ∈ nodes.py registry (import lazily to avoid cycle)
  - graph.edges: from/to both node ids, port names of kind's ports
  - no duplicate ids; cycle detection (Kahn's algorithm) for execution graphs
  - returns [] when valid.
- [ ] Step 3: Tests PASS: 
& .\.venv\Scripts\python.exe -m unittest discover -s tests/server -v
- [ ] Step 4: Commit "flow: graph schema + validator".

## Task 3: Node registry

**Files:** Create server/nodes.py; expand test_*.py (test_registry.py).
- [ ] Step 1: failing test: 6 kinds exactly; each exposes ports [{name,direction,type,burst-format}] & props keys & gate.
- [ ] Step 2: implement: dict NODES:
  kind: dataset     out:[cleaned: "raw-dir: dataset name/rows"] gate: none
  kind: prepare     in:"raw-dir"→out:"cleaned-dir"  props:{rows, val_fraction, min_chars}
  kind: tokenize    in:"cleaned-dir"→out:"shard-dir" props:{seq_len, vocab}
  kind: train       in:"shard-dir"→out:"ckpt-dir"   props:{preset,steps,lr_preset}  gate:"gpu"
  kind: eval        in:"ckpt-dir"→out:"report"      gate:"gpu"
  kind: infer       in:"ckpt-dir"                   render-only, gate:"gpu/cpu"
  + validate_props(kind,props) checking only key-existence and finite numbers.
- [ ] Step 3: unittest PASS, commit "flow: node registry".

## Task 4: flows/ store

**Files:** Create server/flows.py; list tests.
- [ ] Step 1: failing test: list_flows returns names; save validates graph first (rejects invalid, raises error with reason list); load returns parsed JSON; names are slugified (no path traversal: ../name → rejected).
- [ ] Step 2: implement save(name, g: dict->json to flows/<slug>.flow.json with fsync; also load_open list_g.
- [ ] Step 3: PASS + commit "flow: flows store (git-tracked)".

## Task 5: config_gen.py — graph → configs/flow_<name>.yaml

**Files:** Create server/config_gen.py + tests.
- [ ] Step 1: failing test: linear chain "dataset→prepare→tokenize→train" flow yields YAML equivalent to configs/webui train options — dataset name, rows, val fraction, seq len, preset/steps/lr present; unknown keys rejected; non-linear/branching graph → NotImplementedError-style clean error "runs via advanced YAML only in this MVP" (honest; not silent) until future F5.
- [ ] Step 2: implementation using yaml already in venv: build nested mapping and write configs/flow_<slug>.yaml (never touches shipped configs).
- [ ] Step 3: PASS + commit "flow: config generator".

## Task 6: runner.py — subprocess + status + checkpoint stop + GPU lock

**Files:** Create server/runner.py, tests/test_runner.py.
- [ ] Step 1: failing test with fake scripts:
  - a fake run_custom.py in tests/fixtures that writes to stdout and sleeps; runner.run_flow(config_path) spawns venv python on it; gather_tail() returns last N lines; runner.request_stop() sets stop flag file and runner exits when flag is see (this is webui U11: checkpoint-aligned, no kill); one global lock (same semantics as webui single-job lock; lock raises "already-running" when re-requested).
- [ ] Step 2: implement with subprocess.Popen(text=True), tail-ring-deque(maxlen=200), stop flag file (runs/flow_stop.flag) — the same mechanism the webui U11 defines — and threading.Lock for GPU lock.
- [ ] Step 3: unittest PASS + commit "flow: runner".

## Task 7: FastAPI app.py — all endpoints

**Files:** Create server/app.py + tests/test_app.py (tests call app routes via a fake Request through starlette TestClient — skip: no httpx in venv; instead tests exercise app functions directly).
- [ ] Step 1: failing test calling internal handlers directly (test pure functions used by routes).
- [ ] Step 2: implement app: GET /api/health; GET /api/nodes; GET/PUT/DELETE /api/flows/{name} (validating); POST /api/validate (evaluates graph_schema.validate + preflight: disk check 5GB ceiling, GPU lock free, config gen dry-run) ⇒ list of errors; POST /api/run {name} → 200 {ok} / 409 {errors}; GET /api/run/status → {running, phase, tail:[...], started_at}; POST /api/run/stop → sets stop flag (no kill); main: argparse --port + FLOW_PORT env (default 3010) + StaticFiles serving ../dist in prod mode (--dev flag disables).
- [ ] Step 3: unittest PASS + commit "flow: fastapi app".

## Task 8: Canvas + Palette with right-click add (frontend integration mandate)

**Files:** Create src/FlowCanvas.tsx (real), src/Palette.tsx (real), src/nodes/PipelineNode.tsx, src/store.ts extended.
- [ ] Step 1: PipelineNode: xyflow node with title, prop badges, run-state dot, one .port per registered port (direction mapped to Handle type) sourcing registry metadata via /api/nodes (cached in store on mount).
- [ ] Step 2: Palette: 6 cards; HTML5 dnd onDragStart sets kind; drop on canvas uses xyflow project() to add node at drop point (default props per registry).
- [ ] Step 3: right-click anywhere on canvas opens radial/webui-context-menu listing the 6 kinds + "search… (Ctrl+K)" entry; Enter behavior in Ctrl+K search selects the kind and adds at mouse position.
- [ ] Step 4: edge creation via xyflow-on-connect → validation port types must match; onConnect that mismatches shows toast and refuses.
- [ ] Step 5: vitest test: store reducer for addNode kinds and invalid edge refusal.
- [ ] Step 6: pnpm test PASS + pnpm build PASS; commit "flow: canvas + palette + add/validate UX".

## Task 9: Inspector + run status wiring

**Files:** src/Inspector.tsx + src/api.ts + store.
- [ ] Step 1: api.ts: typed client (getNodes, getFlows, saveFlow, validateFlow, runFlow, runStatus, runStop) — fetch wrappers, JSON-in/JSON-out, errors → {message, errors[]}.
- [ ] Step 2: Inspector tabs: Properties tab based on node kind metadata (preset select, rows/steps number input, LR preset, advanced disclosure that mutates raw props with an "honest: this is YAML" label); Run log tab polls GET /api/run/status every 2 s while running, streams tail; Preview tab shows the last validated graph compiled YAML summary (config preview).
- [ ] Step 3: vitest PASS + build PASS; commit "flow: inspector + run status".

## Task 10: Validate + Run pipeline (G1..G3 wiring)

- [ ] Step 1: Toolbar Wiring: Validate → POST /api/validate → toast errors or "✔ graph valid"; Run → POST /api/run → switches to Run log tab, polling starts.
- [ ] Step 2: E2E: Backend on 3010 (venv: & .\.venv\Scripts\python.exe -m server.app --port 3010); frontend dev server; user-visible M-G2 PASS checklist.
- [ ] Step 3: Manual G2 script: (a) chain dataset→prepare→tokenize→train with valid props → valid; (b) connect shard-dir port → dataset (wrong port type) → rejected; (c) Train without shard input → rejected.
- [ ] Step 4: commit "flow: validate + run wiring".

## Task 11: Infer render-only shortcut (Q4)

- [ ] Step 1: PipelineNode for kind infer: single in-port; on click "Open in webui Chat" → POST /api/run/handoff {ckpt} (server just records + returns URL of webui chat with ckpt selected; no subprocess launched); honest tooltip.
- [ ] Step 2: vitest test for reducer kind=infer routing.
- [ ] Step 3: PASS + commit "flow: infer render-only shortcut node".

## Task 12: Final gates + docs + task bookkeeping

- [ ] Step 1: G1 PASS record (screenshots/smoke: shell runs).
- [ ] Step 2: G2 PASS record (manual validation script above).
- [ ] Step 3: G3 PASS record: run smoke graph to completion with phases chosen from TASKS live status (coordinate GPU lock); checkpoint aligned stop; zero-flag rerun.
- [ ] Step 4: G4 PASS record: generated configs/flow_<name>.yaml equivalent to webui knob-equivalent run (diff YAML semantics, not bytes).
- [ ] Step 5: Docs: create flow/README.md (run commands, port override, node contract summary); update AGENTS.md §2 layout to add flow/; ENVIRONMENT.md append fastapi+uvicorn install note; TASKS.md add milestone rows; HANDOFF.md add flow/ MVP paragraph.
- [ ] Step 6: Commit "flow: MVP complete — G1-G4 recorded" + 

(git lfs status guard: no new >100MB files; flows/*.flow.json are small text).

---

## Execution log (agents fill in — one row per task)

| Task | Status | Evidence (command + exit code) | Commit |
|---|---|---|---|
| (fill per CLAUDE.md §16 evidence rules) |