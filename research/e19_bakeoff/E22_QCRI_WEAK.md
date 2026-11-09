# E-22 — micro arm C: arm-A recipe + QCRI weak data (2026-09-17)

USER GO: micro arm C = arm A + QCRI 5M words (after gate-dedup); keep the
improvement gate at ARM-A level or QCRI is dropped. Cache line DROPPED for
this model.

## Data work
- models/e19/qcri-src Wikipedia_20240420.diac.jsonl copied to
  data/diac/raw/qcri_diac_clone/.
- Gate shingle dedupe (diacritizer/scripts/e22_qcri_gate_dedup.py, 10-word
  Arabic shingles vs fadel test/wn2014/wn2024/sadeed, drop at >=1% coverage):
  **kept 24,882 articles / dropped 7,952 (24%)** -> qcri_gatesafe.jsonl.
  Dropped = gate-boundary material (mostly the wn2014 overlap).
- Windows/pack (e22_pack_qcri.py, same normalize/window/parse contract as
  prepare_data + same label classes via passthrough.parse):
  **362,621 QCRI windows** added to the 2,441,327 v3-train windows -> v3q
  tokens (train 2,803,948 rows, val COPIED UNCHANGED from v3 = 160,630 rows
  so arm C differs from arm A ONLY by the extra train data).
- First packing run silently grabbed the 'url' field (41 chars) instead of
  'text' -> re-run added the real text; earlier v3q build was discarded.

## Training
- configs/diac_micro_a_qcri.yaml (phase micro_c): identical schedule to arm A
  (2500 steps, batch 32, lr 4e-4, ctx 128, 12L/128h/512ffn/4H/2KV).

## Result (fill after run)

| gate | arm A (v3 only) | arm C (+QCRI v3q) | delta |
|---|---|---|---|
| fadel_test | 38.16 | 36.44 | -1.72 |
| sadeed25 | 51.40 | 48.67 | -2.73 |
| wikinews2024 | 60.96 | 58.13 | -2.83 |
| wikinews2014 | 53.87 | 50.11 | -3.76 |
| mean | 51.10 | 48.34 | -2.76 |

User decision rule: if arm C is NOT an improvement over arm A, drop QCRI.
**VERDICT: IMPROVES.** Arm C beats arm A on ALL FOUR gates (-1.7..-3.8 DER,
mean -2.76; the table's 48.42/-2.68 cell was an arithmetic slip - its own rows
sum to 48.34). Biggest gain precisely where QCRI's modern-wiki domain adds the
most new material (wn2024/wn2014), but classical gates also gained (more
vowel-dense supervised text helps the 2.98M stack learn mark placement).
KEEP QCRI as a micro-model training source. NOTE arm C still ~3-4 DER behind
the 30M gold (33.23/45.68/48.99 + wn24 57.58) - gap narrow but real.");

## Artifacts
- diacritizer/scripts/e22_qcri_gate_dedup.py, e22_pack_qcri.py
- data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl (raw/, gitignored)
- data/diac/v3q/tokens (regenerable, untracked 2.6 GB)
- configs/diac_micro_a_qcri.yaml
