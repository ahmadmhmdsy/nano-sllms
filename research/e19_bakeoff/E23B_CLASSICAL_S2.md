# E-23b — S2 classical tashkeela-family expansion (option A, user-approved 2026-09-18)

**Status: CLOSED — check arm NOT adopted. Arm C stays the micro champion.**

## What

Stage S2 of the E-23 plan, executed with the user-picked option A: expand the
classical pool with the ungated tashkeela-family mirrors (the earlier HF hunt
confirmed no open modern-MSA labeled source; classical-only was approved).

## Pipeline

- Sources: community-datasets/tashkeela (97 book-scale rows), asas-ai/Tashkeela
  (804k line rows; it filled the quota before arbml/tashkeela was reached).
- 10-word shingle barrier vs the 4 gates + fadel train (the already-owned
  classical pool): 3,276,760 windows DROPPED vs 1,200,000 kept — dedup did
  real work; raw Tashkeela overlaps our existing classical pool heavily.
- PT.parse quarantine: 2,879 windows only.
- Pack (e23_pack_zm-style single pass): data/diac/v3t = v3q (2,803,948) +
  1,200,000 new windows = 4,003,948 train rows; val copied IDENTICAL (160,630).
- configs/diac_e23b_t.yaml = arm C recipe EXACTLY (micro 12/128/512, 4H/2KV,
  2,976,128 params, batch 32, lr 4e-4, 2500 steps, fp16, no cache).

## Trajectory (DER @2500-style probes, fadel/sadeed/wn24/wn14)

| step | fadel | sadeed | wn24 | wn14 | mean(4) |
|---|---|---|---|---|---|
| 500 | 48.43 | 60.71 | 65.43 | 60.70 | 58.89 |
| 1000 | 41.55 | 54.00 | 61.95 | 55.30 | 53.20 |
| 1500 | 39.97 | 52.49 | 60.96 | 54.67 | 52.02 |
| 2000 | 38.35 | 50.23 | 59.90 | 52.44 | 50.23 |
| 2500 | 38.69 | 50.70 | 59.93 | 52.55 | 48.72 |

## Verdict vs arm C @2500 (frozen yardsticks)

| gate | arm C | arm T | delta |
|---|---|---|---|
| fadel_test | 36.44 | 38.69 | +2.25 worse |
| sadeed25 | 48.67 | 50.70 | +2.03 worse |
| wikinews2024 | 58.13 | 59.93 | +1.80 worse |
| wikinews2014 | 50.11 | 52.55 | +2.44 worse |
| **mean** | **48.42** | **48.72** | **+0.30 worse** |

Worse on EVERY gate at equal budget/steps/data-mix rule (only the data changed).
Interpretation: ~27% of the drawn classical windows were new after dedup, but
the new material is same-family classical — the classical mass roughly doubles
while the modern-MSA share stays fixed, and the four gates (modern-M-leaning
mix) all recede. This matches the lesson-67..69 family pattern.

**NOT adopted; v3q (arm C) stays. No user gate triggered** (arm C remains
champion; nothing shipped changed).

## Side notes

- The classical human-gold expansion lever is now measured at ~= zero on this
  line: the remaining headroom is the modern-MSA label gap, which no open
  labeled source fills (hunt + this check arm both confirm).
- v3t tokens kept per tokens-tracking policy; val identical across all micro arms.
- Run: runs/diac/e23b_t (kept: best.pt, final/, gate_eval.csv, best_gate.json).
- Scripts: diacritizer/scripts/s2_classical_build.py (stage 1),
  diacritizer/scripts/s2_pack.py (stage 2), scratch/s2_peek.py (dataset probe).
