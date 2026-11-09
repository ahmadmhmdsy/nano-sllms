# E-23c — S3 Z-Mahmood distillation arm (arm E) — RESULT (2026-09-17/18)

## Label generation (PASSED as engineering)
- diacritizer/scripts/e23_zm_label.py: 12,000 qcri-gatesafe articles -> 280-char
  paragraphs, marks stripped -> ZM-BiLSTM vocalized 142,370 paragraphs =
  3.06M words (CPU, ~80 paras/s after warmup; probe first).
- Pack (e23_pack_zm.py, disk-light direct-shuffle single pass; drive E had
  ~4 GB free so e23a_gold_v3q checkpoints were trimmed - final/ + best.pt +
  gate CSV kept): v3qz tokens = 3,119,320 rows = v3q 2,803,948 + ZM 315,372;
  val byte-identical to v3.

## Arm E (configs/diac_e23c_zm.yaml, exact arm-C recipe on v3qz)

| gate | arm C (v3q) | **arm E (v3qz)** | delta |
|---|---|---|---|
| fadel_test | 36.44 | 39.88 | +3.44 worse |
| sadeed25 | 48.67 | 52.72 | +4.05 worse |
| wikinews2024 | 58.13 | 61.44 | +3.31 worse |
| wikinews2014 | 50.11 | 56.15 | +6.04 worse |
| mean | 48.42 | **52.55** | +4.13 WORSE |

Trajectory: 58.8 (500) -> 51.5 (1500) -> 51.5 (2000) -> 52.6 (2500); the run
crashed once mid-training and auto-resumed cleanly (train.py), then the shape
of decline resumed identically - result NOT a resume artifact.

## Verdict: ZM distillation DID NOT help. NOT adopted.

**Not** the same outcome as E-23a (gold + machine-label data): the 2026-10-06
verdict audit corrected E-23a to a WIN (all four gates at step 2000) once the
gold's wn2024 baseline was fixed from 48.65 to 57.58. E-23c is the arm that
genuinely fails. Mechanism retained: label-STYLE conflict from a rule-(a)
qualified teacher. Our pool mixes several human label
styles; adding ZM's independent label distribution (its own nn-choices on
names/idioms) makes a SMALL model average conflicting styles instead of
locking onto the human gold style. Every machine-supplement so far:
  +2.7 mean on micro when from QCRI (E-22), -4.1 on micro from ZM labels
  (E-23c); on gold 30M from QCRI (E-23a) it is a WIN of -2.56 mean4 at step
  2000 (corrected 2026-10-06), not -1.45.
Reading: gain from machine labels on the micro was SUPPLEMENT QUANTITY, not
quality; once the pool is saturated by human labels, a NEW machine style
hurts no matter the teacher quality. Best micro stays arm C (v3q).

## Artifacts
- configs/diac_e23c_zm.yaml; runs/diac/e23c_zm/{gate_eval.csv,final}
- data/diac/v3qz/tokens (3.1M rows, regenerable)
- diacritizer/scripts/e23_zm_label.py, e23_pack_zm.py

## Next (per E-23 plan): S4 BiLSTM A/B (user-gated launch) is the remaining
approved stage; S2 classical-only acquisition paused pending user appetite.
