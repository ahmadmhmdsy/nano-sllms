# μ1 Composition Bake-off — Report (E-27/E-28/E-29, closed 2026-09-19)

Plan: docs/plans/2026-09-19-mu1-composition-plan.md · Gated launch: user approved
("benefit from the past result"; student budget 12K steps USER-APPROVED).
Experts material for ALL arms: the 12K finals (runs/mex/archive_12000) per μ0c's
min-eval finding (high-step checkpoints overfit).

## Headline verdict
At micro scale (≤800K params/task, one consumer GPU), the 120K-step DENSE
multi-task control remains the snapshot champion across 3 of 4 arms measured:

| arm | mechanism | verdict | headline number |
|---|---|---|---|
| A soup/TIES (E-27) | weight merge of 12K experts | **FAIL** — 0.0 exact-match on every task | basin divergence: embedding corr(x1,x2)=0.027 |
| C dispatch router (E-28) | tiny char-MLP router + whole-input dispatch | **routing PASS (1.00 [0.981,1.0])**, team premium marginal | routed team 0.68 vs control 0.64 on 200 mixed (+0.04, CIs overlap) |
| D committee distill (E-29) | dense student, 0.5KL+0.5CE, 12K steps | **FAIL to beat control** | mixed 0.50 [0.431,0.569] vs 0.64; below expert baseline on all 4 tasks |
| B MoE-merge | arch change (USER-GATED) | not run | — |

## Detailed results (exact match, Wilson 95% CI)
- (per-task table mirrored from runs/mex/*/mex_eval.json — 12K expert refs
  x1 0.1075 / x2 0.898 / x3 0.918 / x4 0.846; control cross-task
  x1 0.1085 / x2 0.848 / x3 0.918 / x4 0.802.)
- Student (12K): x1 0.0465 [0.038,0.057] · x2 0.580 [0.536,0.622] ·
  x3 0.888 [0.857,0.913] · x4 0.544 [0.500,0.587] — all below expert AND control.
- Routed team (arm C): x1 0.18 / x2 0.90 / x3 0.82 / x4 0.82.
- Router confusion: perfect diagonal; tasks are self-distinguishing by format
  (same structural finding as the x3 label-prior edge).

## What μ1 teaches (the "benefit from the past" angle)
1. Weight merging at micro scale = dead end (basins diverge from the shared
   seed because each task drives embeddings to near-orthogonal geometry).
   The honest fix is routing/distillation, NOT merging — recorded in MEMORY.
2. Dispatch is nearly free to implement and its router is PERFECT here — the
   interesting number it produces is the routed-team ceiling, which is still
   only ~+0.04 over the dense control. At this scale specialization buys less
   than compute: control > all composed arms except x4 specialist margin.
3. The 12K-budget student cannot match the 120K-step control — an
   equal-tokens control (12K-step dense) is the missing baseline for a fair
   read; recorded as a mu1-follow-up candidate in the E-29 row.
4. Self-distinguishing task formats make arm-B MoE-merge's learned router
   redundant at this scale (format-only routing ≈ dispatch).

## Verification trail
Row-first discipline held: pre-registered E-27/28/29 (commit 82aab6f) BEFORE
any results; results in separate commits (e67ad36 soup, 4b9eaca router,
E-29 pending commit with artifacts). Tests 30/30 (mex/tests). Eval artifacts:
runs/mex/soup/*/mex_eval.json, runs/mex/mex_eval_mixed.json,
runs/mex/distill_student/final/{mex_eval.json train_summary.json}.
