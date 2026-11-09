# GUI Hardening & Scalability Milestones

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement these milestones task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Fix the identified mistakes, missing implementations, and weaknesses across all three GUI components (flow/, viz/, webui/) in a reusable, scalable way — not one-time patches.

**Scope:** All three GUI components share cross-cutting concerns (param math duplication, no E2E tests, no shared UI library, no error boundaries). This plan addresses both per-component fixes and cross-cutting infrastructure.

**Principle:** Every fix must be reusable — no hardcoded one-offs, no special-casing, no "fix it for this one model/node." If a fix works for one component, it must work for all.

---

## Milestone M1: Safety & Data Protection (flow/)

**Why:** A visual graph editor without undo/redo or auto-save is a data-loss machine. Users *will* make mistakes.

### Task M1.1: Undo/Redo System for flow/ Pipeline Canvas

**Files:** `flow/src/stores/graphStore.ts`, `flow/src/store.ts`, `flow/src/FlowCanvas.tsx`, `flow/src/App.tsx`

- [ ] Step 1: Design a command-pattern history stack in the graph store. Each mutating action (`addNode`, `connect`, `applyNodesChanges` with remove, `applyEdgesChanges` with remove, `updateNodeProps`, `updateNodeLabel`, `setGraph`) pushes a `{undo: () => Graph, redo: () => Graph}` pair onto a past-stack and clears the future-stack.
- [ ] Step 2: Expose `undo()` and `redo()` actions on the store. `undo()` pops past-stack, pushes current to future-stack, sets graph. `redo()` is the inverse. Both are no-ops when their stack is empty.
- [ ] Step 3: Add `canUndo` and `canRedo` boolean selectors to the store for UI button states.
- [ ] Step 4: Wire `Ctrl+Z` (undo) and `Ctrl+Shift+Z` / `Ctrl+Y` (redo) in `FlowCanvas.tsx` via a `keydown` listener. Ignore when focus is in an input/textarea/select.
- [ ] Step 5: Add Undo/Redo buttons to the Toolbar with `disabled={!canUndo}` / `disabled={!canRedo}` and tooltips.
- [ ] Step 6: Write unit tests: undo after addNode restores empty graph; redo re-adds; undo after connect removes edge; multiple undos/redos in sequence; undo after props write-back restores old props; future-stack cleared on new action after undo.
- [ ] Step 7: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 8: Commit: `git add flow/ && git commit -m "flow: undo/redo history stack for pipeline canvas"`

### Task M1.2: Auto-Save & Draft Recovery

**Files:** `flow/src/stores/graphStore.ts`, `flow/src/App.tsx`, `flow/src/FlowCanvas.tsx`

- [ ] Step 1: Add a `useEffect` in `App.tsx` that serializes `store.graph` to `localStorage` under key `flow:draft` on every graph change (debounced 2s).
- [ ] Step 2: On mount, check `localStorage.getItem('flow:draft')`. If present and different from the empty graph, show a non-blocking toast: "Unsaved draft recovered — Save to keep it" with a "Discard" action.
- [ ] Step 3: Clear the draft from localStorage on successful `saveFlow` and on `openFlow` (the server is now the source of truth).
- [ ] Step 4: Add a "Clear draft" button to the Save modal for explicit user control.
- [ ] Step 5: Write unit tests: draft is written to localStorage on graph change; draft is cleared on save; draft is cleared on open; corrupted localStorage JSON is handled gracefully (no crash).
- [ ] Step 6: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 7: Commit: `git add flow/ && git commit -m "flow: auto-save draft to localStorage with recovery"`

### Task M1.3: Delete Key Handler for Selected Nodes/Edges

**Files:** `flow/src/FlowCanvas.tsx`

- [ ] Step 1: Add a `keydown` listener for `Delete` and `Backspace` keys. When a node is selected, dispatch a `remove` NodeChange for that node id. When an edge is selected, dispatch a `remove` EdgeChange for that edge id.
- [ ] Step 2: Ignore the handler when focus is in an input/textarea/select (check `document.activeElement.tagName`).
- [ ] Step 3: Write unit tests: Delete key removes selected node; Delete key removes selected edge; Delete key is inert when typing in an input.
- [ ] Step 4: Run `pnpm test` → PASS.
- [ ] Step 5: Commit: `git add flow/ && git commit -m "flow: Delete key removes selected nodes/edges"`

### Task M1.4: Error Boundary for flow/ Frontend

**Files:** `flow/src/App.tsx`, `flow/src/ErrorBoundary.tsx` (new)

- [ ] Step 1: Create `ErrorBoundary.tsx`: a React class component with `componentDidCatch` that renders a fallback UI with the error message and a "Reload" button (`window.location.reload()`).
- [ ] Step 2: Wrap the `<Palette/>`, `<FlowCanvas/>`, and `<Inspector/>` components in `<ErrorBoundary>` so a crash in one panel doesn't white-screen the entire app.
- [ ] Step 3: Wrap `<ModelCanvas/>` in its own `<ErrorBoundary>`.
- [ ] Step 4: Write a unit test: a component that throws renders the fallback UI, not a white screen.
- [ ] Step 5: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ && git commit -m "flow: error boundary prevents white-screen crashes"`

---

## Milestone M2: Editor UX & Navigation (flow/)

**Why:** A graph editor without minimap, zoom controls, or node search is unusable on complex graphs.

### Task M2.1: Minimap & Zoom Controls

**Files:** `flow/src/FlowCanvas.tsx`

- [ ] Step 1: Import `MiniMap` and `Controls` from `@xyflow/react`.
- [ ] Step 2: Add `<Controls />` inside the `<ReactFlow>` component (position bottom-left).
- [ ] Step 3: Add `<MiniMap />` inside the `<ReactFlow>` component (position bottom-right). Style with `className="minimap"` and a dark background matching the theme.
- [ ] Step 4: Run `pnpm build` → PASS.
- [ ] Step 5: Commit: `git add flow/ && git commit -m "flow: minimap and zoom controls"`

### Task M2.2: Node Search & Highlight on Canvas

**Files:** `flow/src/FlowCanvas.tsx`, `flow/src/store.ts`

- [ ] Step 1: Add a `searchQuery: string` field to the UI store slice.
- [ ] Step 2: Add a search input to the Toolbar (magnifying glass icon, placeholder "find node…").
- [ ] Step 3: When `searchQuery` is non-empty, highlight matching nodes (by id, kind, or label) with a CSS class `node-highlight` (e.g., a glowing border). Non-matching nodes get `node-dim` (reduced opacity).
- [ ] Step 4: Pressing Enter cycles through matches (select + center on the next match).
- [ ] Step 5: Write unit tests: search query filters nodes by kind; search query filters by label; Enter cycles matches.
- [ ] Step 6: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 7: Commit: `git add flow/ && git commit -m "flow: node search and highlight on canvas"`

### Task M2.3: Export/Import Flow as JSON

**Files:** `flow/src/App.tsx`, `flow/src/api.ts`

- [ ] Step 1: Add an "Export" button to the Toolbar that serializes the current graph to a `.flow.json` file and triggers a browser download.
- [ ] Step 2: Add an "Import" button to the Toolbar that opens a file picker, reads a `.flow.json` file, validates it client-side (schema check), and loads it into the store via `setGraph`.
- [ ] Step 3: Show a toast on import success/failure.
- [ ] Step 4: Write unit tests: export produces valid flow/0.1 JSON; import parses and loads; import rejects invalid JSON with a toast.
- [ ] Step 5: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ && git commit -m "flow: export/import flow as JSON"`

### Task M2.4: Confirmation Dialog for Destructive Actions

**Files:** `flow/src/App.tsx`, `flow/src/ConfirmDialog.tsx` (new)

- [ ] Step 1: Create a reusable `ConfirmDialog` component: props `{open, title, message, onConfirm, onDismiss}`. Renders a modal with "Confirm" and "Cancel" buttons.
- [ ] Step 2: Wire it to the "Clear canvas" action (add a "Clear" button to the Toolbar that opens the dialog before calling `setGraph({nodes:[],edges:[]})`).
- [ ] Step 3: Wire it to the "Delete flow" action (add a "Delete" button next to "Open" that calls `DELETE /api/flows/{name}` after confirmation).
- [ ] Step 4: Write unit tests: dialog shows on Clear click; dialog confirm clears canvas; dialog cancel does nothing.
- [ ] Step 5: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ && git commit -m "flow: confirmation dialog for destructive actions"`

### Task M2.5: Loading States & Tooltips

**Files:** `flow/src/App.tsx`, `flow/src/FlowCanvas.tsx`, `flow/src/Inspector.tsx`, `flow/src/App.css`

- [ ] Step 1: Add a `Spinner` component (CSS animation, `role="status"`, `aria-label="Loading"`).
- [ ] Step 2: Show the Spinner in the Toolbar while `saveFlow`, `validateFlow`, or `runGraph` is in-flight (the `saving` state already exists; extend to validate/run).
- [ ] Step 3: Show the Spinner in the Inspector while the Run Log is polling and the tail is empty.
- [ ] Step 4: Add `title` tooltips to all Toolbar buttons (Open, Save, Validate, Run, Export, Import, Clear, Delete).
- [ ] Step 5: Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ && git commit -m "flow: loading spinners and toolbar tooltips"`

---

## Milestone M3: Model Canvas Hardening (flow/)

**Why:** The model canvas computes errors but doesn't show them visually on the canvas itself.

### Task M3.1: Visual Error Indication on Model Nodes

**Files:** `flow/src/model/ModelNode.tsx`, `flow/src/model/ModelCanvas.css`

- [ ] Step 1: When `data.error` is non-null, add a CSS class `model-node-error` to the node wrapper (red border, error icon).
- [ ] Step 2: When `data.shape` is null (no shape inferred), add a CSS class `model-node-uncomputed` (dashed border, "?" icon).
- [ ] Step 3: Show a tooltip on hover with the full error message.
- [ ] Step 4: Write unit tests: node with error gets error class; node without shape gets uncomputed class; node with shape and no error gets neither.
- [ ] Step 5: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ && git commit -m "flow: visual error indication on model canvas nodes"`

### Task M3.2: Fix Incomplete TokenFlowDemo Implementation (viz/)

**Files:** `viz/src/viz/TokenFlowDemo.tsx`, `viz/src/viz/Diagram.tsx`

- [ ] Step 1: Remove the three `void` statements (`void onLiveLayer`, `void at`, `void liveAttn`) and wire the `onLiveLayer` callback to actually highlight the current layer in the parent `Diagram`.
- [ ] Step 2: Pass `onLiveLayer` from `TokenFlowDemo` up to the parent `Stacks` component, which passes it to `Diagram` to set a `liveLayer` state that highlights the corresponding block.
- [ ] Step 3: Remove the dead `FfnBars` code path (`if (layer == null || [...].includes("none")) void 0;`).
- [ ] Step 4: Remove the dead `badges` function in `viz/src/App.tsx`.
- [ ] Step 5: Run `pnpm test` → PASS. Run `pnpm build` → PASS.
- [ ] Step 6: Commit: `git add flow/ viz/ && git commit -m "viz: wire live-layer highlight, remove dead code"`

---

## Milestone M4: Webui Hardening

**Why:** The webui has a race condition on `_MODEL`, no input validation, and no logging.

### Task M4.1: Fix `_MODEL` Race Condition

**Files:** `webui/app.py`

- [ ] Step 1: Add a `threading.Lock` (`_MODEL_LOCK`) around all reads and writes of the `_MODEL` global.
- [ ] Step 2: In `chat_stream`, acquire the lock before reading `_MODEL`, copy the needed references (`tok`, `model`, `device`, `ctx`) to local variables, release the lock, then use the locals. This prevents a concurrent `unload_model` from setting `_MODEL = None` mid-generation.
- [ ] Step 3: In `unload_model`, acquire the lock before checking/unloading `_MODEL`.
- [ ] Step 4: In `_model_status_md`, acquire the lock before reading `_MODEL`.
- [ ] Step 5: Write a unit test: concurrent load + unload doesn't crash; chat_stream with unload mid-generation completes gracefully.
- [ ] Step 6: Commit: `git add webui/ && git commit -m "webui: fix _MODEL race condition with lock"`

### Task M4.2: Input Validation on Chat Parameters

**Files:** `webui/app.py`

- [ ] Step 1: Clamp `max_new_tokens` to a reasonable range (1–4096). Reject values outside this range with a clear error message.
- [ ] Step 2: Clamp `temperature` (0.01–2.0), `top_p` (0.01–1.0), `top_k` (1–200).
- [ ] Step 3: Validate that `message` is a non-empty string after stripping.
- [ ] Step 4: Write unit tests: max_new_tokens=0 is rejected; max_new_tokens=10000 is clamped; temperature=0 is rejected; empty message is rejected.
- [ ] Step 5: Commit: `git add webui/ && git commit -m "webui: input validation on chat parameters"`

### Task M4.3: Structured Logging

**Files:** `webui/app.py`, `webui/model_tab.py`, `webui/explorer.py`

- [ ] Step 1: Add a `logging.getLogger(__name__)` to each module.
- [ ] Step 2: Replace all `except Exception: pass` with `except Exception: logger.warning("...", exc_info=True)` — errors are recorded, not swallowed.
- [ ] Step 3: Add INFO-level logs for key events: model loaded/unloaded, run launched, checkpoint deleted, validation errors.
- [ ] Step 4: Commit: `git add webui/ && git commit -m "webui: structured logging replaces silent exception swallowing"`

### Task M4.4: Caching for Expensive Reads

**Files:** `webui/app.py`, `webui/model_tab.py`

- [ ] Step 1: Add a simple TTL cache (e.g., `functools.lru_cache` with a manual TTL wrapper) for `_full_curve` — cache the (steps, values) tuple keyed by (logs_dir, tag, file_mtime).
- [ ] Step 2: Cache `_ckpts()` results for 30 seconds (the checkpoint list doesn't change during a run).
- [ ] Step 3: Cache `_free_vram_mib()` for 5 seconds.
- [ ] Step 4: Write unit tests: cache returns same object on second call; cache invalidates on TTL expiry; cache invalidates on file mtime change.
- [ ] Step 5: Commit: `git add webui/ && git commit -m "webui: TTL caching for expensive reads"`

---

## Milestone M5: Cross-Cutting Infrastructure

**Why:** These issues affect all three components and must be fixed once, reusable everywhere.

### Task M5.1: Extract Shared Param Math Package

**Files:** `packages/param-math/` (new shared package), `viz/src/engine/params.ts`, `flow/src/model/paramMath.ts`, `webui/explorer.py`, `scripts/status.py`

- [ ] Step 1: Create `packages/param-math/` as a shared TypeScript package (for viz/flow) and a Python module (for webui/scripts).
- [ ] Step 2: Move the canonical param math functions (`pEmbed`, `pAttentionDense`, `pFFNDense`, `pNorms`, `pTotalDense`, `pTotalHybrid`, `pMixerLinear`, `pMixerFull`, `pMoeLayer`) into the shared package.
- [ ] Step 3: Make `viz/src/engine/params.ts` re-export from the shared package (backward-compatible).
- [ ] Step 4: Make `flow/src/model/paramMath.ts` re-export from the shared package (backward-compatible).
- [ ] Step 5: For Python: create `packages/param-math/param_math.py` with the same functions. Update `webui/explorer.py` and `scripts/status.py` to import from it.
- [ ] Step 6: Write unit tests in both TS and Python that verify the shared package produces the same anchors (nano 226,526,208 / SmolLM2 134,515,008).
- [ ] Step 7: Run all test suites → PASS.
- [ ] Step 8: Commit: `git add packages/ viz/ flow/ webui/ scripts/ && git commit -m "all: extract shared param-math package"`

### Task M5.2: Shared UI Component Library

**Files:** `packages/ui-components/` (new shared package)

- [ ] Step 1: Create `packages/ui-components/` with reusable React components: `Spinner`, `ConfirmDialog`, `Toast`, `ErrorBoundary`, `Tabs`, `SearchInput`.
- [ ] Step 2: Each component accepts a `className` prop for styling flexibility and uses CSS variables for theming.
- [ ] Step 3: Migrate `flow/src/App.tsx` to use the shared `ConfirmDialog`, `Spinner`, and `ErrorBoundary`.
- [ ] Step 4: Migrate `viz/src/App.tsx` to use the shared `Tabs` and `ErrorBoundary`.
- [ ] Step 5: Write unit tests for each shared component.
- [ ] Step 6: Run `pnpm test` in both flow/ and viz/ → PASS.
- [ ] Step 7: Commit: `git add packages/ui-components/ flow/ viz/ && git commit -m "all: shared UI component library"`

### Task M5.3: E2E Test Infrastructure

**Files:** `tests/e2e/` (new)

- [ ] Step 1: Set up Playwright (or Cypress) in the repo root with configs for flow/, viz/, and webui/.
- [ ] Step 2: Write E2E tests for flow/: add node → connect → validate → save → reload → open → verify graph restored.
- [ ] Step 3: Write E2E tests for flow/: undo/redo (M1.1), delete key (M1.3), export/import (M2.3).
- [ ] Step 4: Write E2E tests for viz/: switch tabs → verify content renders; click model card → verify inspector updates.
- [ ] Step 5: Write E2E tests for webui/: boot → verify dashboard renders; load model → verify chat works.
- [ ] Step 6: Add a `pnpm test:e2e` script that runs all E2E suites.
- [ ] Step 7: Commit: `git add tests/e2e/ && git commit -m "all: E2E test infrastructure"`

### Task M5.4: Error Boundary for viz/ and webui/

**Files:** `viz/src/App.tsx`, `viz/src/ErrorBoundary.tsx` (new), `webui/app.py`

- [ ] Step 1: Create `viz/src/ErrorBoundary.tsx` (same pattern as flow/ M1.4).
- [ ] Step 2: Wrap the viz/ tab content in `<ErrorBoundary>`.
- [ ] Step 3: For webui/, add a Gradio `gr.Error` handler or a try/except wrapper around the `launch()` function that shows a friendly error page instead of crashing.
- [ ] Step 4: Run `pnpm test` and `pnpm build` in viz/ → PASS.
- [ ] Step 5: Commit: `git add viz/ webui/ && git commit -m "viz/webui: error boundaries"`

### Task M5.5: Accessibility Audit & Fixes

**Files:** `flow/src/FlowCanvas.tsx`, `flow/src/Inspector.tsx`, `viz/src/App.tsx`, `viz/src/viz/Diagram.tsx`

- [ ] Step 1: Add ARIA labels to all icon-only buttons (toolbar, palette collapse, inspector collapse).
- [ ] Step 2: Add `role="tab"`, `aria-selected`, and `aria-controls` to the viz/ tab buttons (currently just `<button>` with no tab semantics).
- [ ] Step 3: Add keyboard navigation to the flow/ canvas: arrow keys to move between nodes, Tab to cycle through handles.
- [ ] Step 4: Add `aria-live="polite"` to the toast region (already partially done — verify all toasts have it).
- [ ] Step 5: Add `aria-label` to the ReactFlow canvas region.
- [ ] Step 6: Run `pnpm build` in both → PASS.
- [ ] Step 7: Commit: `git add flow/ viz/ && git commit -m "all: accessibility audit and fixes"`

---

## Milestone M6: Performance & Scalability

**Why:** The current implementations don't scale to large graphs or many models.

### Task M6.1: Virtualization for Large Graphs

**Files:** `flow/src/FlowCanvas.tsx`, `flow/src/model/ModelCanvas.tsx`

- [ ] Step 1: Profile rendering with 50+ nodes. If frame drops occur, implement node virtualization (only render nodes within the viewport + margin).
- [ ] Step 2: For the model canvas, memoize the `walkModelGraph` result with `useMemo` keyed on the graph object identity (already partially done — verify it works for large graphs).
- [ ] Step 3: Write a benchmark test: 100-node graph renders in < 100ms.
- [ ] Step 4: Run `pnpm test` → PASS.
- [ ] Step 5: Commit: `git add flow/ && git commit -m "flow: virtualization for large graphs"`

### Task M6.2: Code Splitting & Lazy Loading

**Files:** `flow/src/App.tsx`, `flow/src/model/ModelCanvas.tsx`, `viz/src/App.tsx`

- [ ] Step 1: Use `React.lazy` + `Suspense` to lazy-load the `ModelCanvas` component (it's only needed when the user switches to Model mode).
- [ ] Step 2: Lazy-load the viz/ `TokenFlowDemo` component (it's only needed in the Demo tab).
- [ ] Step 3: Add a `Suspense` fallback with the shared `Spinner` component.
- [ ] Step 4: Run `pnpm build` → PASS. Verify bundle size reduction.
- [ ] Step 5: Commit: `git add flow/ viz/ && git commit -m "all: code splitting and lazy loading"`

### Task M6.3: Responsive Layout for flow/

**Files:** `flow/src/App.css`, `flow/src/FlowCanvas.tsx`, `flow/src/Palette.tsx`, `flow/src/Inspector.tsx`

- [ ] Step 1: Add CSS media queries: below 1200px, the palette and inspector become overlays (toggleable) instead of side-by-side columns.
- [ ] Step 2: Below 768px, the toolbar wraps to two rows.
- [ ] Step 3: Test at 1920px, 1280px, 768px, and 375px widths.
- [ ] Step 4: Run `pnpm build` → PASS.
- [ ] Step 5: Commit: `git add flow/ && git commit -m "flow: responsive layout"`

---

## Milestone M7: Cleanup & Documentation

### Task M7.1: Remove Dead Code

**Files:** `viz/src/App.tsx`, `viz/src/viz/TokenFlowDemo.tsx`, `viz/src/engine/params.ts`

- [ ] Step 1: Remove the `badges` function from `viz/src/App.tsx` (dead code).
- [ ] Step 2: Remove the `void` statements and dead code paths from `TokenFlowDemo.tsx` (done in M3.2 — verify).
- [ ] Step 3: Fix the confusing comment in `pLayerDense` (`viz/src/engine/params.ts`): clarify that `2 * s.d` is the per-layer norm weight and `pNorms` counts all norms exactly once.
- [ ] Step 4: Run `pnpm test` → PASS.
- [ ] Step 5: Commit: `git add viz/ && git commit -m "viz: remove dead code, fix confusing comment"`

### Task M7.2: Update Documentation

**Files:** `flow/README.md`, `viz/README.md`, `webui/README.md`

- [ ] Step 1: Document the new undo/redo, auto-save, export/import, and keyboard shortcuts in `flow/README.md`.
- [ ] Step 2: Document the shared param-math package in `viz/README.md` and `flow/README.md`.
- [ ] Step 3: Document the E2E test setup in the root `README.md` or `tests/e2e/README.md`.
- [ ] Step 4: Commit: `git add flow/README.md viz/README.md webui/README.md && git commit -m "docs: update for new features"`

---

## Execution Order & Dependencies

```
M1 (Safety) ──→ M2 (UX) ──→ M3 (Model Canvas)
                    │
                    ├──→ M4 (Webui)
                    │
                    └──→ M5 (Cross-Cutting) ──→ M6 (Performance) ──→ M7 (Cleanup)
```

- **M1** is the highest priority (data loss prevention).
- **M2** and **M4** can proceed in parallel after M1.
- **M3** depends on M2 (shared components).
- **M5** should start after M2/M4 (the shared library needs to know what components are needed).
- **M6** depends on M5 (performance patterns need the shared infrastructure).
- **M7** is last (cleanup after all features are stable).

## Validation Labels

Per AGENTS.md, every task completion must declare one of:
- **PASS**: All unit tests pass, build is clean, feature works as specified.
- **FAIL**: Tests fail or feature doesn't work — describe the failure.
- **SKIPPED**: Task not attempted — describe why.
- **BLOCKED**: Task cannot proceed — describe the blocker.
- **NEEDS_USER_DECISION**: Task requires user input — describe the decision needed.

## Evidence Requirements

Every commit must reference:
- The test file(s) that verify the change.
- The build output (clean `pnpm build` / `pnpm test`).
- For webui changes: the headless boot log or browser drill evidence path.
