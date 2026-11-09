# Research: training a small model to generate UI-as-JSON (json-render target line)
Date: 2026-09-24 (session report; user-requested research + design thinking, no training launched)
Status: research note — feeds future pre-registration on the UI-JSON line (name TBD, e.g. "UI line" / U-1)

## 0. TL;DR
- Task: user prompt -> json-render spec (flat `{root, elements, state}` JSON; 41-component standard catalog; $-expression bindings). This is a *catalog-constrained declarative-UI generation* task — exactly the regime validated for small models by the 2026 A2UI study (arXiv 2609.04184): 0.8B-4B SFT students recover ~97-98% of front-tier teacher quality; base 2B models get only 3% valid parses untutored.
- No published sub-100M result exists on this task family. Narrow catalog + shallow tree targets + heavy synthetic SFT is the field's supported route for tiny models. Our niche: 5-150M specialist on ONE catalog, strict validator as the primary gate (not held-out perplexity).
- Data: synthetic self-generated loop (WebSight pipeline shape + UICoder automated-feedback iteration) is the proven recipe; we reuse our proven judge/rerank lever (KT-2 = E-06/E-09) as the filter, and graft onto our pico retrosynthesis rewrite codegen for JSON, not prose.
- Decoder: keep autoregressive decoding over the flat spec, but consider the element-CHAIN linearization (one element record per line, children listed as already-defined keys) instead of raw JSONL patches — it moves structural validity from the model into a cheap linearizxer while encoding layout decisions the model actually must make (siblings, types, props). This CRUD-like shift is the "sequence design" strategy change the report targets.
- Compute plan: what the D-line and Laya-line taught us about depth-per-param, data mix, balanced replay, metric-pegged early stop, all apply directly.

## 1. Target format (json-render, vercel-labs/json-render; json-render.dev) — facts needed for training
- Apache-2.0, docs ship LLM-friendly markdown + an (`@json-render/core`) prompt builder. Official repo ships no "train a model" dataset; example specs (7, in examples/no-ai) + prompt templates are the only gold seeds.
- Wire format the model should learn: flat Spec `{root, elements: {key: {type, props, children[], visible?, on?, repeat?, watch?, slots?}}}`, optional `state` sibling. JSON Pointer paths everywhere (RFC 6901).
- Streaming variant = RFC 6902 JSONL patches (one patch/line, /root first, then /elements + /state interleaved). JSON-Patch and inline-tree (nestedToFlat) are alternate ingests; flat spec is THE LLM target.
- Grammar-critical rules: `children` REQUIRED (leaves emit []); all child keys must resolve to defined elements (missing_child is the dominant generated failure mode); `visible/`on`/`repeat`/`watch`/`slots` are element-level siblings, NOT inside props; nullable optional props (Zod .nullable()) -> explicit nulls are valid and common.
- Expression language: any prop can be { "$state": "/path" }, { "$bindState": "/path" } (2-way), { "$item"/"$bindItem": "field" } (repeat scope), { "$index": true }, conditional { "$cond", "$then", "$else" }, { "$template": "..." }, { "$computed": "fn", "args": {...} }, directives ($format, $math, $concat, $count, $truncate, $pluralize, $join, $t). State actions (on events / watch): setState / pushState (with "$id" auto-ids) / removeState / validateForm, all with params-as-expressions, optional confirm, onSuccess/onError handlers.
- Vocabulary: 36 shadcn components (Retrieval-ready definitions + props/events) in @json-render/shadcn; the official chat demo extends to ~41 with Metric/Bar/Line/Pie charts, Callout, Timeline + optional 3D. React runtime renders in ~1-2ms per element; action/event protocol via graph dedup'ed in registry.
- Validation contract available TODAY (free oracle): validateSpec + autoFixSpec check schema AND semantic (missing_child, catalog membership, bindings on value-bearing props, visible/on/repeat shape). This is the perfect 3-layer gate for training data filtering AND eval — .visible/on/repeat slot field placement is auto-fixable losslessly in many cases via autoFixSpec (report only lossless fixes as "repairable").

## 2. Prior-work survey (details in-session subagent report; URLs saved in-session)
| Work | Arch / size | Data | Strategy | Result | Limits for us |
|---|---|---|---|---|---|
| Pix2Code '17 | CNN+RNN 109M | fully SYNTHETIC GUI generator | supervised, template DSL | 77% token acc on toy DSLs | toy space, image-conditional (we are text-conditional) |
| WebSight-Sightseer '24 | VLM ~8B | 823k-2M synthetic pairs (Mistral idea->DeepSeek-33B HTML->Playwright shot) | DoRA FT | near-commercial on in-domain | Tailwind syntax thin in pretraining; 8B, screenshot-conditional |
| Design2Code '24 | GPT-4o + FT open models | 484 real pages benchmark | eval benchmark | GPT-4o best; FT 8-18B approach it in-domain | confirms FT >> prompting for open models |
| UICoder '24 | StarCoder-15B | ~1M self-generated SwiftUI, 0.4% survive-filter | iteration: compile+CLIP filter->SFT->pref-align | beats downloadable baselines | 15B; loop cost |
| Web2Code '24 | LLaMA-3-8B IT | 1.18M web instructions | SFT | WCGB 1.79->8.53 | 8B browser-judge eval |
| Pix2Struct '23 | 282M-1.3B | 80M web screenshots | masked screenshot->simple-HTML pretrain | SOTA 6/9 UI tasks | pretrain compute overwhelming |
| A2UI small-model study '26 | Qwen 0.8-4B, SmolLM3 | catalog-conditioned A2UI JSON, teacher(GT)+| Perturbed-catalog + Constrained-GT (SIMPLER trees) best | 4B SFT ~98% teacher quality; 0.8B viable in-distribution | closest prior art; still >=0.8B |
| Liquid-WebSight '25 | LFM2.5-VL 450M | WebSight v0.2 subset | LoRA | edge demo | screenshot-conditional |
| Constrained decoding | PICARD / Outlines / XGrammar / OpenAI SO | — | grammar-masked decode; JSONSchemaBench '25 | structure correctness ~100% feasible at decode time, +up to 4% quality | "Grammar-Aligned Decoding" (2405.21047): grammar forces low-likelihood strings -> degrades quality; mitigate via candidate resampling |
Field takeaways for sub-150M: (1) synthetic pipeline + auto-filter/iterate is standard and works; (2) nobody has a text->typed-JSON-UI model below ~450M — our lane; (3) evaluate on THREE validity layers (parse/schema/catalog/bind) not perplexity; (4) shallow-restricted GT trees beat free-depth GT at small scale (A2UI Constrained-GT).

## 3. What the task actually decomposes into (small-model epistemology)
A json-render spec is a combined decision chain, no free lunch:
1. STRUCTURE decision: how many elements (usually 5-40), parent/sibling tree shape, slots.
2. TYPE choice: element types under a 36-41 type catalog (a 36-way decision with local content context).
3. PROP fill: discrete enums (direction/gap/level/variant), free text (labels/text/title), booleans, arrays (items,options,tabs), DATA-BINDING objects vs literals.
4. BEHAVIOR: events (on.press etc.), actions (setState/pushState/removeState), watch, visible conditions, repeat + state layout.
5. (optional) STREAMING form: emitting JSONL patches incrementally rather than full spec (this is a change of SURFACE, not of decisions).
For a 5-150M specialist we want to TRAIN decisions 1-4 and make 0 (JSON syntax) machine-guaranteed.

## 4. Our recommendation: three candidate architectures (strictly ordered by compute/complexity)
### A1. Reuse the Nano-arch decoder, element-chain linearization (first pre-registration candidate)
- Killed idea-of record: train a GPT-style decoder on raw JSONL patch streams; the JSON bracket balance burden stays with the model, and our sub-150M scale makes that the main wall (prior field evidence: base 2B obtains ~3% without FT, ours would be worse).
- Instead: represent each spec as a canonical linear element-chain:
  TASK <prompt> -> EL <key> TYPE <TypeName> P <prop>:<v> ... K <childkey>,<childkey>
  ...
  with one line (= one BPE-atomic unit boundary) per element, children-before-or-after rules fixed (children listed AFTER the element that references them -> "resolve-forward" is gone: better, REQUIRE the chain to emit elements in a fixed valid topological order: parent-before-child, plus children[]: references only ALREADY-emitted keys, or forward-ref marker that our determinstic packer resolves).
- Deterministic decode/encode: chain -> flat spec is a trivial pass (we own it), spec -> canonical chain exists so data can be canonically serialized for training only.
- Nothing autoregressive is lost; syntax errors become IMPOSSIBLE if we gate decoding on type-token/prop-token segments with a tiny per-layer grammar or simply choose non-JSON surface tokens. This is NOT grammar-constrained regular-LLM decode, it is a proxy task surface s.t. an arbitrary decoder variant works.
- The model still does the REAL work: which types, which props, which values, structure. This is exactly catalog-conditioned generation.
- Trade: training consumes a custom surface; you need a chain->spec packer (cheap, deterministic CPU) and the eval validator (parse + catalog + binding checks) as the objective metric; no off-the-shelf renderer round-trip in training loop, but json-render validateSpec is our offline filter; the renderer itself runs only at eval/demo.
- Non-fully-autoregressive extensions (fits our six-type past experiments): probe (a) element-level decode order = BFS (layout-tree order is semantically loaded — use per-spec canonical order, not BFS), (b) two-pass draft element-TYPE skeleton then prop fill (curriculum for pass-2), (c) pointer/copy attention for binding paths (a $state path is a COPY of a state subtree path — pointer nets fit this) — separate pre-registrations, maybe U-2.

### A2. Element-by-element autoregressive with CFG-constrained decode (outlines/XGrammar)
- Standard GPT-style nano/P-scale model; output raw flat spec.
- Inference: grammar-masked decoding per-token tree CFG extracted from the json-render schema + catalog (cheap per cached schema; JSONSchemaBench reports near-free overhead), used only at INFERENCE; training = SFT on chain-linearized or raw specs.
- Note the Grammar-Aligned-Decoding distortion risk: a hard grammar can pin low-probability channels; mitigate with candidate resampling (top-5) + validator pick.
- This is the pragmatic fallback if A1's custom surface complicates inference-time tooling/UX.

### A3. Two-stage hybrid: nano-arch encoder-decoder CED-lite or GDN hybrid (per our E-08/E-40 results)
- Only if A1/A2 hit a quality wall at 5-150M; the encoder gets the prompt+catalog summary, the decoder emits chain. CED-lite halves KV. Not the first rung.

## 5. Data plan (CPU-heavy, per proven pipeline; the GPU is only stage-B/SFT)
Stage U-0 (build + gold):
- CATALOG FREEZE: pick the 41-component chat-example catalog (Card, Stack, Grid, Metric, Table, Line/Bar/Pie, Input/Select/Radio/Switch, Button, Tabs, Accordion, Text/Heading/Alert/Badge, Dialog, Progress...) + 6 actions. Catalog text = constant prompt prefix, learns as an effective system prompt. (A2UI paper: perturbed-catalog DATA beat fixed-catalog training; keep perturbation for robustness arm, not default.)
- GOLD SEEDS: the repo's 7 hand-made examples + prompt-templated (generatePrompt) exemplars; hand-extend to ~50 high-quality prompts->spec pairs by LLM-authorship (via our judge loop): task prompts (users requests, natural phrasing) -> canonical chain.
Stage U-1 (synthetic scale, CPU-only ~minutes-hours):
- Programmatic SPEC Generator: a Python generator with 8-15 structured archetypes (form+validation, dashboard w Metrics+charts+state, todo/list w repeat+setState, profile card, settings w $bindState, filter/search w watch(like cascading cities), landing w hero+CTA, detail view w tabs+table, checkout w conditional drawer, i18n w $template+ directives, data table w $computed+filters, pagination view). This mirrors Pix2Code's synthetic generator but at ~10-50x richer decision space, and it guarantees validity by CONSTRUCTION: every generated spec must pass validateSpec + our 3-layer gate.
- Richness levers: randomized tree-shape (depth 2-6, width 1-6, slot mix), prop subsetting (nulls where nullable), state-usage probability (some specs stateless -> easier, some with bindings/visible/watch/repeat mixed in), controlled null/enum noise, catalog-subset masks (perturbation idea), persona/template prompts.
- SIZING from field evidence: UICoder filtered ~0.4% of 1M -> 1M doesn't map to our budget; A2UI uses ~ tens of thousands; WebSight needed millions only for the harder visual pixel task. For text->chain at 50M params, target 50k-200k pairs first, measuring curve on held-out validity + tree similarity, NOT fixed at 200k blindly.
- Prompt DIVERSITY: paraphrase with templates + optional viewer/retrieval corpus from repo microscope? NO — keep synthetic (scope: codegen for JSON, not search across real corpora). Paste-style prompt variants (user asks for 'a todo widget that shows ...') -> spec; persona fields (dashboard, form, ...) -> mix.
Stage U-2 (judge/rerank + self-improvement, our proven lever KT-2):
- Sample k=8-16 candidates per prompt from the trained model, validate (3 layers) + score design-quality via our Laya-MiniLM decision-judge (E-70..E-73 artifacts are TEXT judges; NOT redesigned here) or via deterministic heuristic scorer (valid + archetype match + no degenerate repetition + render-size sanity), SFT again on survivors. 2-3 iterations like UICoder's, batched (our 5.6x batched-judge lesson).
- This directly reuses judge/rerank loop infra (E-06/E-09) WITHOUT new ML plumbing.

## 6. Training recipe (lever-by-lever from WHAT_WORKS.md)
| Recipe choice | Supporting experiment | Notes for this line |
|---|---|---|
| Warm-start via embedding transplant (KT-1) | E-05 | transplant from our larger GQA/GDN sibling tokenizer; reuse scripts/embed_transplant.py |
| Depth over width at fixed params | E-12/E-13 (D-line) | axis conversions: tree generation needs long-chain reliable recall; depth or GDN hybrid better than wide |
| GDN hybrid 3:1 layout | E-08 (S-scale) | candidate arch for the chain decoder; proven at S, un-gated at P |
| effective-batch elevation; 8-bit Adam | E-09, E-12 | defaults |
| 1-epoch SFT sweet spot | E-15 | watch forgetting; here forgetting-of-base LESS a concern (specialist), but 2-epoch AST-style collapse analog: watch chain-ability vs. base-ability tension; validation-gate not recipe |
| distill instead of pretrain from scratch | E-16 | IF a sibling larger model exists on this task; first rungs = scratch (fresh task), maybe stage-3 distill later |
| plain KD (not skew-KL), fp32 teacher logits | E-03 | if distilling |
| 520M-token scale Stage-A pretrain + Stage-B SFT | E-70..E-73 | our 2-stage = stage-A "format-grammar" LM pretraining on synthetic chains (60-80% of tokens), stage-B SFT on (prompt,chain) pairs |
| per-source stratified replay | E-71 | sources = archetypes; prevents archetype-coupling collapse seen in Laya typed/transfer coupling |
| metric-pegged early stop | E-73 WIN | stop stage-B when held-out VALIDITY + tree-similarity composite drops 0.02 below peak |
| avoid: MTP aux head, calibration fits, QA-mixing-as-cure | E-42, E-15 | same negatives expected here: a $-binding-validity aux head could be re-tried ONLY decoupled-LR, otherwise skip |
| grammar-aligned decoding caution | 2405.21047 | if A2 used, resample top-3 |

### Gating (pre-register before ANY run; per repo rule 5 / EXPERIMENTS.md process)
- VALID4U (valid@4, validity= 3-layer pass+catalog: parse+schema/catalog+binding/object): PRIMARY.
- TreeEditSimilarity/generation-similarity (softmax over tree-edit-distance per batch, or the axis converted to equal children list order canonicalized): continuous quality GATE secondary.
- F1 over (type, prop-path, value-eq) triples per generation (light CAT metric).
- Prompt-holdout split by archetype AND by prompt-template (Laya lesson: unattended splits rotate best when each seed index is disjoint).
- Budget note: 6 GB card, fp16 only; a 50M model with chain-ctx 1024-2048 fits entirely in stage-B (+8-bit Adam) as per E-40 probe: activations dominate only at ctx 2048; batch 32-64 feasible with CED-lite in-era.

## 7. Risks / open questions
1. LINEARIZATION inventiveness: chain surface is bespoke; risk = losing community tooling. Mitigation: keep chain <-> spec duals lossless + test round-trip extensively; JSONL-patch streaming remains a port available later.
2. $-expressions: binding paths need pointer-copy generalization; unknown how rare mutations of state subtree shape flow... mitigate with a "binding-path coverage analysis" dossier in corpus build.
3. Catalog drift: if we later serve against DIFFERENT catalogs (or users register their own 36-41 types), a frozen-vocabulary model must be re-perturbed-trained (Perturbed-catalog arm pre-registered as robustness rung, cheap to add to U-1).
4. Unlike A2UI (0.8B floor for good quality), sub-150M may cap at "fast, valid, medium semantic quality" — set expectations: pattern-fill widget-grade output, not full-app freeform; keep "app" expectations out of the first gate (no next.tanstack app spec in scope).
5. Streaming (JSONL patches) is a SEPARATE surface: don't mix it in v1; date it as a post-hoc wrapper (chain->patch emitter is deterministic code, not a model).

## 8. Recommended next concrete step
E-stage plan on the UI line (name suggestion: "U-line"), mirroring the D-line playbook:
- U-0: freeze catalog, write chain encoder/decoder + 3-layer validator + canonical template generator (~1-2 d CPU work).
- U-1: build 100k synthetic chain corpus (CPU, minutes-hours) + holdout by archetype AND template seed.
- U-1b: pre-register U-2 PRETRAIN (format-LM on chains) + U-3 SFT (prompt->chain) with the metric-pegged early stop of E-73 and balanced per-archetype replay of E-71 from the start; our longest proven traits (embedding transplant, 8-bit Adam, depth-over-width) as the base recipe levers.
- Decide after U-2/U-3 gates whether to also pre-register the A2 grammar-constrained decode arm as a robustness fallback.
Total: recipe inheritance ~high; new engineering = chain serialization, synthetic generator, validator gate; prior-art risk LOW (A2UI study proves regime at >=0.8B; our novel claim is the sub-150M element-chain simplification).

