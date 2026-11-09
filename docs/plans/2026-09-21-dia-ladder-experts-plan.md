# dia-Line Ladder + Expert-Composition Plan (E-60 chain) — 2026-09-21

Status: DRAFT for user approval. Written after the E-56→E-59 failure chain; codifies the lessons as hard rules so they are never repeated.

## 0. Mistakes ledger (never repeat — each becomes a rule)

| # | Mistake (what actually happened) | Rule that forbids it |
|---|---|---|
| M1 | Trunk born BIG (49.4M, 2-layer, hidden 1280) instead of small-then-grow | **R1 ladder**: every new task starts at a small trunk (~3–6M). Growth only by rung-gated additions after the current rung passes its gates |
| M2 | Marks never had a supervised target during pretrain — marks learned implicitly through copy-patterns, then extracted from frozen features by bolted-on heads | **R2 joint objective**: char-LM loss AND mark-class loss are co-optimized from step 0 of every rung. No head ever trained on frozen features again |
| M3 | 21% of params sat in frozen bridges transplanted from other experts and never co-trained | **R3 no decorative mounts**: an expert enters a rung only as (a) a co-TRAINING teacher (distillation loss) or (b) weights that get co-trained into the student. Frozen side-path mounts are banned |
| M4 | Depth-starved: 2 layers / hidden 1280 for a word-sequential task | **R4 deep-narrow**: growth favors adding layers over widening hidden. Per rung, params/layer small, count grows |
| M5 | Every fix was a patch on the same frozen cast (data mix, ctx, head, head-lr) | **R5 one variable per rung, chosen before launch; if a rung fails structurally twice, the recipe itself is retired — no third tuning pass on the same recipe |
| M6 | Experts never used as training-time signal | **R6 experts are teachers, not ornaments**: each growth rung distills from the relevant deployed expert(s), measured with a distill-vs-alone A/B arm (KD harness already exists: mex KD scripts + c12 kd.py pattern) |
| M7 | Batch-size/step surprises (OOM, misread eval loops) burned wall-time | **R7 probe before every new shape**: one VRAM/throughput probe rung-first, batch sized to measured peak, effective batch fixed across rungs for comparability |

Invariants kept (they were the right things): honest pre-registered gates, full metric profile per standing policy, no destructive merge (init-order/composition only), single-GPU discipline, venv-only, evidence-based verdicts.

## 1. Ladder design (Rung A → Rung B → Rung C, each user-gated)

### Rung A — E-60a: small joint trunk (the true "start small")
- Arch: 4 layers × hidden 320, ffn 1280, GQA 8q/2kv, ctx 96 → **~5.2M params** (measured at sanity; adjust if measurement differs).
- Vocab: the existing 97-char CharVocab (no new tokenizer).
- Task, co-trained end-to-end: (a) char-LM CE on the packed stream; (b) per-base mark classification over 13 classes (bare + 8 single + 4 tanwin heads… exact class list from diacritizer/src/labels.py) on positions aligned via TK.decode; class-weighted to correct the bare/fatha prior imbalance.
- Data: gold's v3q tokens (data/diac/v3q/tokens) as primary + g1/g2 replay at 50/50 only if A under-fits (E-57 lesson preserved, applied to the small trunk).
- Gates (pre-registered, same bar as the chain): external 4-gate mark-acc ≥ 0.65 mean, lift ≥ 0.20, in-domain markpos ≥ 0.70 at small scale + CE ≤ 3.0. Full metric profile artifacts per standing policy.
- Budget: probe first (R7); ~1–2 h GPU est (dataset identical to E-58's, trivially small model).
- **Kill criterion**: if markpos < 0.60 at small scale, the architecture class itself is questioned before any widening (report to user).

### Rung B — E-60b: grow by +4 layers (first real ladder step)
- Load Rung A trunk; add 4 fresh deep-narrow layers of the same width (deep-narrow rule R4); new layers train, old 4 layers optionally frozen-with-LoRA (growth-by-mount, never destructive).
- Expert use (R6): mark-pattern expert + word-shape expert serve as KD teachers during this rung — teacher forward each step, loss = CE + λ·KL(student||expert logit/hidden states); zero frozen bridges. A/B arm without distillation must be run (cheap at this scale) to prove the expert signal earns its cost.
- Gates: same bar + must beat Rung A on markpos by ≥ +2 points; distill arm must beat no-expert arm by ≥ 2% mark-acc or the expert term is dropped next rung.

### Rung C — E-60c: grow to target depth
- Repeat Rung B (+4 layers → 12) and/or modest widen (Net2Net wide 320→640, the proven mu2 mechanism — E-32) — whichever the rung B evidence favors.

### Composite closure
- Trained mark head + trunk compose into the existing composed bench pipeline (bench_dia2 with the E-60 trunk), full gates + profile, verdict rows per rung, HANDOFF/TASKS updated, commit per rung.

## 2. Process defaults carried forward
- Probe → config → pre-register → launch (background) → eval (SEQ=96) → 4 external gates + mark_metrics full profile → verdict → commit. One GPU job at a time.
- Effective batch constant across rungs (whatever the probe sets), lr per repo precedent (5e-5 settle / 4e-4 pretrain), adamw_bnb_8bit default (Milestone B decision).
- No bridge mounts anywhere in the chain; expert access is distillation-loss only.
