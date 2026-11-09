/**
 * Playwright E2E config for the nano_SLMs GUI surfaces.
 *
 * Three projects (flow / viz / webui), each pointing at its own dev server.
 * All four servers are started by Playwright's webServer array before the
 * suites run and stopped afterwards:
 *
 *   - flow-backend : FastAPI (flow/server/app.py) on :3010. The flow Vite
 *                   dev server proxies /api here, so the flow suite exercises
 *                   the full browser -> Vite -> FastAPI stack.
 *   - flow-frontend: Vite dev server (flow/) on :5174 (flow/vite.config.ts).
 *   - viz-frontend : Vite dev server (viz/) on :5173 (Vite default; viz has
 *                   no server.port override).
 *   - webui        : Gradio (webui/app.py) on :7861 --no-browser (the
 *                   headless-check port from webui/AGENTS.md; 3010 is taken
 *                   by the flow backend and 7860 is the Gradio default).
 *
 * Two Windows-specific notes:
 *   - Playwright spawns webServer commands with shell: true, i.e. through
 *     cmd.exe. cmd.exe rejects relative program paths written with forward
 *     slashes (".venv/Scripts/python.exe" -> "'.venv' is not recognized"),
 *     so the interpreter path uses the platform separator.
 *   - Vite's default host is "localhost", which resolves to IPv6 ::1 first
 *     on this machine, so the dev servers would listen on ::1 while the
 *     health checks and the tests use 127.0.0.1 (ECONNREFUSED). The vite
 *     commands therefore pass --host 127.0.0.1 explicitly.
 *
 * The suites are resilient by construction: every assertion goes through
 * Playwright's auto-retrying expect() (visibility / text / counts), never
 * fixed sleeps. Selectors prefer ids, roles and data-* attributes that the
 * apps already ship (toolbar button ids, role="tab"/"menuitem"/"dialog",
 * data-model/data-block in viz, react-flow__* classes).
 */
import { defineConfig, devices } from "@playwright/test";
import { join } from "node:path";

const FLOW_BASE = "http://127.0.0.1:5174";
const VIZ_BASE = "http://127.0.0.1:5173";
const WEBUI_BASE = "http://127.0.0.1:7861";

// Playwright spawns webServer commands with shell: true, i.e. through
// cmd.exe on Windows. cmd.exe rejects relative program paths written with
// forward slashes (".venv/Scripts/python.exe" -> "'.venv' is not
// recognized"), so the interpreter path uses the platform separator.
const PY =
  process.platform === "win32"
    ? join(".venv", "Scripts", "python.exe")
    : join(".venv", "bin", "python");

// Vite dev server, pinned to IPv4: the default host "localhost" resolves
// to ::1 first here, which the 127.0.0.1 health checks cannot reach.
const VITE = "node node_modules/vite/bin/vite.js --host 127.0.0.1";

export default defineConfig({
  testDir: "./tests/e2e",
  // Generous per-test timeout: the flow suite builds a 4-node chain through
  // the UI and the webui boots a Python/Gradio stack.
  timeout: 90_000,
  expect: { timeout: 15_000 },
  // The suites share long-lived dev servers and (for flow) server-side saved
  // flows; serial execution keeps runs deterministic.
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  webServer: [
    {
      name: "flow-backend",
      command: PY + " -m flow.server.app --port 3010 --dev",
      url: "http://127.0.0.1:3010/api/health",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      name: "flow-frontend",
      command: VITE,
      cwd: "flow",
      url: FLOW_BASE,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      name: "viz-frontend",
      command: VITE,
      cwd: "viz",
      url: VIZ_BASE,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      name: "webui",
      command: PY + " " + join("webui", "app.py") + " --port 7861 --no-browser",
      url: WEBUI_BASE,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
  projects: [
    {
      name: "flow",
      testMatch: /flow\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        baseURL: FLOW_BASE,
        // Wide viewport: the flow suite places a 4-node chain on the canvas
        // and needs every node + port handle on screen for drag-connects.
        viewport: { width: 1600, height: 1000 },
      },
    },
    {
      name: "viz",
      testMatch: /viz\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], baseURL: VIZ_BASE },
    },
    {
      name: "webui",
      testMatch: /webui\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], baseURL: WEBUI_BASE },
    },
  ],
});
