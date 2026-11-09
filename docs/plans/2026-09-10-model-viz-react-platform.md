# MODEL-VIZ REACT PLATFORM — Full build plan
**Status:** PLAN WRITTEN (user asked for a full plan + fresh-context handoff). BUILD NOT STARTED.
**Gate:** user go required before Phase 1 (Navigator? one TASKS.md row owns status: this is TASKS.md row 48).
**Plan written by:** agent session 2026-09-10 (built the 3-model static explorer first, this is its replacement/generalization).

---

## 0) WHY THIS EXISTS (past experience, so you don't repeat it)

What already exists and works: `model_architecture_explorer.html` — a single-file offline HTML
viz comparing **nano_SLMs target (dense GQA, 226.5M)** · **SmolLM2-135M (dense GQA)** ·
**Qwen3.8-Flash-Next (hybrid linear/full attention + 512-expert MoE, ≈125B)**. It contains:
side-by-side SVG layer stacks, a clickable block inspector (Plain/Technical), a knowledge-heat
depth band, a simulated token-flow demo player, depth-role + param-share maps, and 3-model
spec/similarity/purpose tables. Everything data-driven except prose, and fully browser-verified
(2026-09-10; see TASKS.md row 47 + its HONEST UPDATE for the evidence trail).

LESSONS LEARNED (this build caused every bug below; do NOT repeat them):
1. **Data was embedded in code** (a 740-line JS blob). Adding the 3rd model = ~15 surgical
   string-edits, several of which broke the script silently. Therefore: **config-data-only
   inheritance is the core of this design.**
2. **Shared prose carried per-model facts.** "BOTH models tie" became ternaries inside
   paragraphs. Therefore: facts are schema fields; prose stays generic or is generated.
3. **Derivable values were hand-written once** (param totals double-counted norms; caught only
   by a browser drill). Therefore: param numbers are computed from shapes and unit-tested.
4. **A missing static container killed the whole page silently** (no #col-qwen38 div → header
   loop threw → every later section stayed empty). Therefore: registry-driven rendering NO
   hand-wired containers; and an error surface that is visible.
5. **Validation was improvised** (node --check on the extracted script, injected window.onerror,
   agent-browser probes). It worked and saved hours. Therefore: build these in as developer
   tooling from Phase 0.
6. Qwen "3.8 Next-Flash" was initially unverified folklore in TASKS row 28 — always verify
   externally-published specs against the primary config.json before rendering them.

## 1) GOALS

- G1 **Add-a-model = one config file.** An agent (or human) fills `models/<id>.yaml`; dev server
  hot-reloads; the model appears in ALL views (diagram column, inspector, comparison tables,
  knowledge map, demo chips) with zero code changes.
- G2 **Escape hatch for mutants.** An optional `<id>.overrides.ts` handles unusual archs
  without polluting the schema (git: only 'hybridMoe' and 'dense' and 'hybridAttention'
  kinds exist as built-in shapes).
- G3 **Honest by construction.** Every number showsagen where it came from ("computed from
  config shapes", "SIMULATED", citation links) — schema-enforced, not goodwill.
- G4 **Fresh-agent-proof** — this document + repo docs are the only context needed.
- G5 Scalable: adding many features later (filters, sharing, more visual kinds) must not
  require touching existing model files.

## 2) NON-GOALS (do not build)

- No backend, no DB, no auth, no translations/i18n, no mobile app, no CDN dependencies
  (offline-first, matches repo culture — see ENVIRONMENT.md network constraints).
- No coupling into the Gradio webui/ Model tab (webui/explorer.py U12 already exists there;
  the React app is standalone and read-only the same way).
- No "replace the static HTML" in V1 — it stays as-is until this reaches parity; then the
  static file becomes an EXPORT TARGET (see D7), not a maintenance surface.
- No per-model prose files except override hooks (that's the failure mode this plan exists
  to prevent).

## 2) TECH DECISIONS

| # | Decision | Why |
|---|---|---|
| D1 | Vite + React + TypeScript, **pnpm** (target node ≥20; machine has node 24) | single package.json; no runtime deps beyond zod + yaml; dev server + type-safety catch the #1 historical failure (silent script corruption) |
| D2 | Data in **YAML**, types in **TS** | user-editable + agent-friendly, validation with clear fix-hints |
| D3 | **archKind enum**: `denseGqa` · `hybridMoe` (GDN-linear/full-gated attention + MoE) · `denseSwiGLU` fallback? — NO. Keep exactly **two** built-in kinds for V1: `denseGqaSegment` and `hybridLinearMoe` matches every model we have; new kinds are opt-in later. |
| D4 | Params computed from shapes with **unit tests** (never hand-written in prose/data) |
| D5 | Simulated demo activations from ONE seeded engine with per-model knobs (depth shape, entropy curve, vocab bias), **clearly labeled SIMULATED** |
| D6 | Knowledge-location depth bands = heuristic defaults (1/3 thirds; overridable per model) with a "rule-of-thumb" badge — Geva 2021 / ROME 2022 citations stay in the UI |
| D7 | Static single-file export stays as a build target (works offline; the repo values offline artifacts) — implemented as a build script that reuses the same data, NOT a second hand-maintained artifact |
| D8 | Scope discipline: **only 4 features in V1** (diagram, inspector, knowledge map, demo) + compare tables. Anything else = new phase |
| D9 | All styling tokens/colors in one theme module; no CSS framework |

## 3) TARGET FILE TREE

```
viz/
  package.json · tsconfig.json · vite.config.ts · index.html
  src/
    main.tsx · App.tsx                         # tabs: Stacks | Compare | Demo | About
    theme.ts                                   # colors/icons/fonts (single place)
    registry/models.ts                         # Vite glob import of *.yaml → typed specs
    schema/spec.ts                             # Zod schemas (ModelSpec, Facts, Demo)
    engine/params.ts                           # param math from shapes (+ tests)
    engine/depth.ts                            # bands + heat ramp (+ tests)
    engine/sim.ts                              # seeded fake activations (+ tests)
    viz/Stacks.tsx   Diagram.tsx  BlockInspector.tsx
    viz/KnowledgeMap.tsx  CompareTables.tsx  TokenFlowDemo.tsx
    viz/kinds/denseGqaMoe.tsx  viz/kinds/hybridLinearMoe.tsx
  models/
    _template.yaml                             # copy-me starter (dense family)
    nano-target.yaml · smollm2-135m.yaml · qwen3.8-flash-next.yaml
    qwen3.8-flash-next.overrides.ts            # example escape hatch (kept tiny)
  tests/ (engine unit tests; vitest)
  export/standalone.ts                         # later (D7): builds the old-style single file
```

## 4) ModelSpec — the contract (full source of truth = viz/models/_template.yaml)

See **appendix A** for the three fully-filled YAMLs (nano / SmolLM2 / Qwen) — the fresh agent
can copy-paste these; the facts inside are already verified this session (do not re-research).
Core shape (TypeScript):

```ts
interface ModelSpec {
  id: string; name: string; fullName: string;
  accent: string;                          // hex, unique per model
  archKind: "denseGqaMoe" | "hybridLinearMoe";
  shape: { layers:number; d:number; heads:number; kv:number; ffn:number;
           vocab:number; ctx:string; theta:string; eps:string; tied:boolean;
           precision:string; attentionImpl?:string; };
  experts?: { total:number; active:number; shared:number; inter:number };
  layerPattern?: string;                   // e.g. "3 linear + 1 full" (hybrid only)
  bands?: { early:[number,number]; mid:[number,number]; late:[number,number] };
  facts: {                                 // overrides of shared component prose
    tokens?: string,  embed?: string,  mixer?: string,  ffn?: string,
    lnorm?: string,   head?: string,   band?: string,   …
  };
  demo?: { prompts?: {...}[]; entropyBias?: number; moePeaks?: boolean };
  honesty: { sources: string[]; simulatedNote: string; trainedOn: string; speaks: string };
}
```
**Defaults layer**: a global DEFAULTS object supplies common prose (residual-add, finalnorm,
logits, tokenizer boilerplate) so a dense model's YAML is ~15 lines. The `facts` block only
overrides where the model genuinely differs (tie mode, mixer kind, expert counts).

## 5) PHASES (build gates)

| Phase | Work | Done-gate (VALIDATION REQUIRED) |
|---|---|---|
| P1 scaffold + schema + registry | vite app, zod schema, 3 YAMLs migrated, `Stacks` shows 3 columns with real SVG blocks (port the drawing code as a React component — the geometry is proven; do NOT redesign it) | `pnpm build` clean; `pnpm doctor` PASS; browser drill: 126 blocks / 3 cards / inspector opens on click |
| P2 inspector + maps | BlockInspector (Plain/Technical), KnowledgeMap + purpose table from specs + facts branches; compare tables (spec/sim/diff) auto-generated from ALL loaded specs | clicks on every block of every model produce correct detail; `pnpm test` (params/bands) green |
| P3 demo engine | seeded activations, per-model knobs, GDN "no grid" note, top-5 bars, chips auto-populate | demo plays for all 3 models; no console errors (window error hook kept) |
| P4 hardening + handoff | vitest unit tests for params/bands/sim; doctor fix-hint messages; README-viz; this file + HANDOFF updated; static export (D7) only if trivially cheap | all gates green; TASKS row 48 `done` with evidence links |
Rough effort: P1 0.5–1 d · P2 0.5–1 d · P3 0.5 d · P4 0.5 d.
**Stop after P4. New features need a new plan entry.**

## 5) VALIDATION HABITS (from hard experience)

- `pnpm doctor` first after ANY config edit (schema + arithmetic + "container count" checks).
- `pnpm test` unit tests for: param sums (nano=226,526,208; smollm2=134,515,008; qwen≈124.6B
  text stack; norm accounting EXACT), band widths sum to 100%, heat curve peaks mid-band.
- Browser checks via the agent-browser skill (it already proved itself twice in this repo):
  assert SVG block count, click path: each block → inspector heading contains the right
  component name; demo next-layer lands on attention stages; SIMULATED badge present.
- Never hand-write a param number in UI text — always render from engine output.
- The error overlay hook (window error → red banner) ships in dev; keep it.

## 6) PITFALL CHECKLIST (each one actually shipped a bug this week)

[ ] Container divs created by `.map(registry)` — never hard-coded pairs/triples.
[ ] No module-scope references to per-model variables (types only; values inside functions/components).
[ ] Every config key optional-safe: components must render with facts missing (default prose).
[ ] Param formulas in ONE module, each with a unit test.
[ ] Attr selectors with numeric values must be quoted (`[data-layer="48"]`) — burned once in the browser drill.
[ ] Text escaping: Ġ/Ċ marks; regex-free JS preferred.
[ ] Never trust external model specs without the config.json link in the same PR.
[ ] On any change: build + doctor + quick browser smoke; THEN claim done.
[ ] Do not special-case Qwen logic into shared components — it must flow through archKind/facts.
[ ] Git: keep `models/*.yaml` in-repo (they are the product), no LFS binaries, weights never here.

## 7) FRESH-AGENT START CHECKLIST

1. Read CLAUDE.md, then AGENTS.md (doc map), then this file top to bottom, then the last
   20 lines of HANDOFF.md + TASKS rows ≥ 47 for the latest state.
2. Machine basics: pwsh; project root = E:\python_projects\nano_SLMs; node ≥ 20 present;
   install with pnpm (do not npm/pnpm-switch mid-build); never touch the Python venv for the viz work.
3. Phase 1 first; stop at every Gate; report PASS/FAIL plainly; update TASKS row 48 status as you go.
4. If user gates or design tension appears → STOP and ask (the plan's Open Questions section).
5. On any paused/incomplete state: write what you left into HANDOFF.md "Model-viz" dated entry
   (status + next action + evidence path) and mark TASKS row 48 accordingly.

## 8) OUT OF SCOPE / FUTURE
Per-model simulated probes; visual kinds beyond what exists; i18n; link-out pages; a11y audit pass.

---
### Appendix A — the three verified spec seeds (= P1 YAML content source)
nano-target: layers 16 d 1024 heads 16 kv 4 ffn 3072 vocab 32768 ctx "1,024 → 4,096 YaRN" theta 1e4 eps 1e-5 tied TRUE fp16 SDPA; params 226,526,208 (unit test anchor!); data CodeSearchNet Python + Evol-Instruct; accent #56c8d8.
smollm2-135m: layers 30 d 576 heads 9 kv 3 ffn 1536 vocab 49152 ctx 8,192 theta 1e5 eps 1e-5 tied TRUE bf16 HF; params 134,515,008; accent #e0a458; source HF config.json.
qwen3.8-flash-next: layers 48 d 2560 heads 24 kv 2 ffn=640(moe) vocab 248320 ctx 262,144 theta 1e7 partialRotary 0.25 eps 1e-6 archKind hybridLinearMoe layerPattern "3 linear + 1 full" experts 512/10/1 untied bf16; total ≈124.6B text-stack (announced ≈125B, ≈6B active); accent #e879a0; source HF config.json (released 2026-08-26).
Demo predictions (SIMULATED, hand-authored, already in the HTML): france/code/fox per model as in the current file — port them verbatim. For new models without authored preds, the demo falls back to a generic neutral list + big SIMULATED badge.

sources: repo configs/target.yaml + src/model.py; HF SmolLM2-135M config.json; Qwen3.8-Flash-Next config.json (2026-08-26); interpretation cites: Geva et al. 2021 (arXiv:2104.08696), Meng et al. 2022 ROME (arXiv:2202.05262).