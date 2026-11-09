# viz — Model Architecture Explorer (React platform)

Config-driven multi-model decoder visualizer. **Adding a model = one YAML file.**

## Run

```powershell
cd viz
pnpm install
pnpm dev        # http://localhost:5173
pnpm build      # typecheck + production build (dist/)
pnpm preview    # serve dist
pnpm test       # unit tests (params / depth / sim)
pnpm doctor     # schema + arithmetic + source-citation gate
```

## Adding a model (G1)

1. Copy `models/_template.yaml` → `models/<id>.yaml`.
2. Fill dims **from a verified source** (paste the config.json URL into `honesty.sources`).
3. Save. The dev server hot-reloads; the model appears in ALL views — diagram column,
   inspector, compare tables, knowledge map, demo — with zero code changes.

Its param total, block params, shares, depth-band widths and heat curve are all
**computed** in `src/engine/params.ts` / `depth.ts` — never hand-written.

## Architecture kinds (D3)

- `denseGqaMoe` — Llama-style GQA + dense SwiGLU (nano_SLMs, SmolLM2-135M).
- `hybridLinearMoe` — GDN-linear/full-attention hybrid + sparse MoE (Qwen3.8-Flash-Next).

New arch shapes: add a field branch in `src/engine/params.ts` + a kind module under
`src/viz/kinds/`, then a heal-the-schema entry in `src/schema/spec.ts`. Nothing outside
those places special-cases a model.

## Shared param math (packages/param-math)

Every param number this app renders comes from ONE canonical implementation:
`packages/param-math` (TypeScript `src/index.ts`, plus a stdlib-only Python
mirror `param_math.py`). `src/engine/params.ts` is a one-line `export *`
shim — the formulas live only in the shared package, and `flow/` re-exports
the same module for its model editor (`flow/src/model/paramMath.ts`). The
Python mirror is consumed by `webui/explorer.py` and `scripts/status.py`
(they put the directory on sys.path and `import param_math`; the hyphen in
the directory name means it can never be a regular package). Dense anchors
are pinned to exact integers on both sides — nano_SLMs 226,526,208 ·
SmolLM2-135M 134,515,008 — by `packages/param-math/test/anchors.test.ts`,
`viz/tests/params.test.ts`, `flow/src/model/paramMath.test.ts` and
`webui/tests/test_param_math.py`. Run the package's own anchors through the
viz vitest install (the package has no node_modules of its own):

```powershell
cd viz
pnpm exec vitest run --root ../packages/param-math
```

## Escape hatch (G2)

For unusual details, add `models/<id>.overrides.ts` exporting `{ facts: { ... } }` —
string overrides only, no logic. It merges over generic defaults at registry load.

## Honesty rules (G3)

- Param numbers NEVER hand-written in UI text — always rendered from the engine.
- Every activation demo value is SIMULATED (seeded engine `src/engine/sim.ts`).
- Norms are counted exactly once (`(2L+1)·d`) — `pnpm test` anchors:
  nano 226,526,208 · SmolLM2 134,515,008 · Qwen text stack within ±1.5% of 124.6B.

Full design rationale: `docs/plans/2026-09-10-model-viz-react-platform.md`.
