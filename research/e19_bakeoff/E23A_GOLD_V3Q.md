# E-23a — gold 30M + QCRI weak windows (S1) — RESULT (2026-09-17; verdict corrected 2026-10-06)

Config configs/diac_e23a_gold_v3q.yaml: EXACT stage2b2500 schedule (28L/272h/1088ffn/
8H/2KV, ctv 128, lr 2e-4, batch 32, 2500 steps, stage1lm warm-start, reset last 2)
except tokens_phase v3q (our v3 pool + 362,621 gate-deduped QCRI windows).
Gates every 500 steps on the standard pipeline. Job pwsh-51, commit 8569362.

## Gates (vs production gold stage2b2500, runs/diac/stage2b2500/gate_eval.csv @2500)

**CORRECTED 2026-10-06 (verdict audit).** The original table used a gold
wikinews2024 of 48.65 and a gold mean(4) of 44.14. Both are wrong: 48.65 is the
gold's *wikinews2014 + word-cache* row from E-20/E-21, not its wikinews2024
score. The deployed gold's own wikinews2024 DER is **57.58**, so its mean(4) is
**46.37**. Against the corrected baseline E-23a at step 2500 is a small win, and
at its best step 2000 it beats the gold on **all four gates**.

| gate | gold (v3 only) | E-23a @2500 | E-23a @2000 | best-step delta |
|---|---|---|---|---|
| fadel_test | 33.23 | 33.54 | **32.87** | -0.36 |
| sadeed25 | 45.68 | 45.33 | **43.36** | -2.32 |
| wikinews2024 | **57.58** | 55.71 | **54.65** | -2.93 |
| wikinews2014 | 48.99 | 47.78 | **44.37** | -4.62 |
| mean (4) | 46.37 | 45.59 | **43.81** | -2.56 |
| mean (3, GATE_EXTERNAL) | 42.64 | 42.22 | **40.20** | -2.44 |

## Trajectory (mean of the 4 gates)

54.84 (500) -> 46.76 (1000) -> 45.54 (1500) -> 43.81 (2000) -> 45.59 (2500);
best_val_loss 0.2257. Non-monotone: the run was still improving before a late
plateau. Checkpoint/best gate weights: runs/diac/e23a_gold_v3q/best.pt
(see gate_eval.csv for exact per-step rows).

## Verdict

**CORRECTED 2026-10-06: this is a WIN, not "NOT-ADOPTED".** QCRI/v3q improves
the 30M gold on every gate once the gold's wn2024 baseline is fixed. The
recorded "+7.06 worse on wikinews2024" compared E-23a's 55.71 against 48.65 -
the gold's wn2014+cache number - instead of the gold's true 57.58. Rescored, at
step 2000 E-23a is better on all four gates and on both means (mean3 40.20 vs
42.64, mean4 43.81 vs 46.37); at the matched step 2500 it is better on 3 of 4
(fadel +0.31 worse).

The surviving interpretation is narrower than the original: the 2.98M micro
still benefits far more (E-22 delta -2.76), so the *size gradient* claim is
intact, but "machine data HURTS the saturated gold" is refuted by E-23a itself.
What still hurts is the distillation variant E-23c (4-gate 51.22 vs gold
46.37). S1 (v3q) is therefore an ADOPTED-CANDIDATE for the production line,
gated on the one-cheap-check rule: it must also beat the current best tiny
checkpoint (E-065 F2, 3-gate 0.3974 @115000) before it displaces anything.

## Artifacts
- configs/diac_e23a_gold_v3q.yaml; runs/diac/e23a_gold_v3q/{gate_eval.csv,final}
- decision rule: adopt only if gold's wn2024/wn2014/fadel/sadeed improve ->
  at step 2000 all four DO improve, so the original "not all improve" call was
  a baseline error, not a measurement.
