# E2E tests (Playwright)

Browser-level tests for the three GUI surfaces — flow/, viz/, webui/ —
driven through the real apps (no mocks). Config: `playwright.config.ts`
at the repo root; specs and helpers live in `tests/e2e/`.

## Run

```powershell
pnpm test:e2e            # all three projects (servers auto-started)
pnpm test:e2e:flow       # one project: flow | viz | webui
```

Playwright starts every server itself via the config's `webServer` array
and stops them afterwards; nothing needs to be running beforehand.

## Playwright config

- **testDir** `./tests/e2e`; per-test timeout 90 s, expect timeout 15 s.
- **webServers** (each health-checked before the suites run):
  - `flow-backend` — FastAPI (`flow/server/app.py --port 3010 --dev`) on
    http://127.0.0.1:3010/api/health. The flow Vite dev server proxies
    /api here, so the flow suite exercises the full browser → Vite →
    FastAPI stack.
  - `flow-frontend` — Vite dev server (flow/) on http://127.0.0.1:5174.
  - `viz-frontend` — Vite dev server (viz/) on http://127.0.0.1:5173.
  - `webui` — Gradio (`webui/app.py --port 7861 --no-browser`) on
    http://127.0.0.1:7861 (7861, not the 7860 default: 3010 is taken by
    the flow backend and 7860 is the Gradio default).
  - `reuseExistingServer: !CI` — locally an already-running server is
    reused; CI always starts fresh.
- **projects** — one per app, each matching its spec file:
  - `flow` → `flow.spec.ts`, baseURL :5174, viewport 1600×1000 (the
    suite places a 4-node chain and needs every node + port handle on
    screen for drag-connects).
  - `viz` → `viz.spec.ts`, baseURL :5173.
  - `webui` → `webui.spec.ts`, baseURL :7861.
- **Serial by design**: `workers: 1`, `fullyParallel: false` — the suites
  share long-lived dev servers and (for flow) server-side saved flows.
- **Windows notes**: the webServer python path uses the platform separator
  (cmd.exe rejects forward-slash relative paths), and the Vite commands
  pass `--host 127.0.0.1` because "localhost" resolves to ::1 first on
  this machine while the health checks use 127.0.0.1.
- **Resilient by construction**: every assertion goes through Playwright's
  auto-retrying `expect()` — no fixed sleeps. Selectors prefer ids, roles
  and data-* attributes the apps already ship (toolbar button ids,
  role="tab" / "menuitem" / "dialog", data-model / data-block in viz,
  react-flow__* classes).

## What each spec covers

### flow.spec.ts (shared helpers in flow-helpers.ts)

Drives the real editor end-to-end: right-click add menu, drag-connect
between port handles, inspector prop fields, the toast channel
(role="alert").

1. **add → connect → validate → save → reload → open** — builds the
   4-node smoke chain through the UI (knobs mirror the golden
   `flow/flows/example-linear-smoke.flow.json`), validates against the
   FastAPI backend, saves under a known slug, reloads the page and reopens
   the flow from the server: the full graph is restored.
2. **undo/redo (M1.1)** — toolbar buttons and Ctrl+Z / Ctrl+Shift+Z.
3. **delete key (M1.3)** — Backspace removes the selected node.
4. **export/import (M2.3)** — Export downloads `untitled.flow.json`
   (schema `flow/0.1`); a fixture document is loaded back through the
   hidden file input and appears on the canvas.

### viz.spec.ts

1. **tab switching** — Stacks (default) → Compare → Demo → About → back.
2. **block click → inspector** — clicking the embedding block and a decoder
   layer in the nano card updates the block inspector.

### webui.spec.ts

1. **dashboard boot** — title, headings, run table, GPU/disk line, all six
   tabs (Dashboard / Monitor / Train / Chat / Settings / Model).
2. **chat UI presence** — message input, checkpoint picker, Load/Unload.
   Actually loading a chat model is skipped with an explicit reason: it
   needs GPU checkpoint weights (and a live training job makes the webui
   refuse to co-run).
