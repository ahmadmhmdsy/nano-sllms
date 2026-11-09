# flow/AGENTS.md - visual flow editor scoped instructions

## Scope
The drag-and-drop pipeline graph editor under `flow/` (React frontend
plus FastAPI backend), its node registry, example flows, and the run/step
tooling for the shared pipeline scripts.

## State and records
- source-of-truth state path: `research/experiments.yaml` +
  `state/tasks.yaml` (scope `flow`).
- Line details and run contract: `flow/README.md`; plans under
  `docs/plans/`; golden registry/config outputs under
  `flow/tests/goldens/`.

## Entry points
- Frontend: `cd flow; pnpm build / pnpm dev / pnpm test` (pnpm only)
- Backend: `& .\.venv\Scripts\python.exe -m flow.server.app --port 3010`
  (--dev for Vite CORS; the port is an explicit launch argument)
- Server tests: venv unittest under `flow/tests/server`; frontend vitest

## Smallest venv validation

    & .\.venv\Scripts\python.exe -W error::SyntaxWarning -m unittest discover -s flow/tests/server -t .

plus `pnpm build` and `pnpm test` for any frontend change; endpoint
changes additionally run one HTTP round-trip probe of the touched
endpoint.

## Data boundaries
- `flow/flows/<slug>.flow.json` are tracked, atomically saved documents;
  example flows are golden inputs; saves cannot leave torn files.

## Hazards
- Registry golden output is a contract; regenerating goldens happens only
  inside the refactor's own review, never as a drive-by.
- Port hygiene: 3010 backend, 5174 dev; never occupy 3080, 7860, or 5173.
- Run wiring wraps the shared scripts; never reimplement pipeline logic
  and never add a kill path (STOP flag contract stays, per the run rules).

## Evidence
The unittest and vitest suite outputs, a clean production build, and the
HTTP gate evidence saved to the review package where applicable.

Root policy is never restated here: see the root AGENTS.md,
rules/gpu.md, rules/data.md, rules/experiments.md, rules/evidence.md,
rules/context.md, docs/agent/OPERATIONS.md, and docs/agent/SCHEMA.md. Live
truth is CURRENT_STATE.md plus state/ and research/experiments.yaml - never
this file.
