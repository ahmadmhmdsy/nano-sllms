# E-23 plan — leverage Z-Mahmood & upgrade our D-line (user-approved 2026-09-17)

User approval: implement all E-21_final-answer section-3 suggestions as
user-approved experiments, SORTED BY MOST EXPECTED GAIN FIRST. Every stage
updates TASKS/HANDOFF/MEMORY/EXPERIMENTS as it lands. Single-GPU rule holds
(one train at a time). Research-only throughout.

## S1 — E-23a: gold 30M + QCRI weak windows (BIGGEST expected gain)

- Why first: E-22 proved +362k gate-deduped QCRI windows cut the 2.98M
  micro's mean gate-DER by -2.7. The 30M gold can absorb the same signal at
  its own capacity; the micro's biggest deltas were on the wn gates, which
  are exactly gold's weakest (48.99/48.65). Modern-MSA data at 30M scale is
  the single most promising gate move available.
- Setup: continue from the SAME v3q tokens (arm C already built them).
  Train gold-arch SFT from runs/diac/stage1lm/final with tokens_phase v3q
  (same shape, so warm-start works); identical stage2b hparam schedule.
- Cost: ~4-5 GPU hours at gold's step rate. Decision rule: adopt v3q for the
  production line only if gold's wn2024/wn2014/fadel/sadeed gates improve;
  otherwise park QCRI for micro work only.

## S2 — E-23b: self-tagged modern-MSA expansion (data hunt behind the same lever)

- RESULT 2026-09-18: hunt found NO open modern-MSA labeled source; user
  approved option A (classical tashkeela-family expansion). 1.2M gate-deduped
  classical windows packed (v3t = 4,003,948 rows); micro check arm at arm C's
  exact budget -> 38.69/50.70/59.93/52.55 mean 48.72 vs arm C 48.42: worse on
  EVERY gate -> **NOT adopted**; arm C stays. Report:
  research/e19_bakeoff/E23B_CLASSICAL_S2.md. S2 CLOSED.

## S3 — E-23c: distill Z-Mahmood into the micro stack (rule (a)-qualified teacher)

- Why third: Z-Mahmood is the only model that beats gold outright
  (12.60 abdou-heldout / mean ~19 on our gates) - its predictions on
  UNLABELED modern MSA text are user-rule-(a)-eligible as soft labels.
- Setup: run Z-Mahmood over a large clean modern-MSA unlabeled pool (news/
  wiki raw from our corpora dirs), produce(vocalized, bare) pairs, pack them
  as a THIRD data source, then either micro-SFT (fast check) or fold into the
  gold train pool.
- Cost: Z-Mahmood CPU inference ~5 words/s = ~24M chars/day; on GPU much
  faster (single-GPU discipline applies). Verify its vocab/hardware claim by
  checking the model load speed on GPU if GPU currently free.

## S4 — E-23d: bidirectional-BiLSTM A/B at equal data (architecture probe)

- Why fourth: highest information-per-GPU-hour but not directly gate-moving.
  The 4.5M BiLSTM WITH BAHDANAU attention beat a 30M transformer by 5-9 DER
  mean on our gates; equal-data A/B isolates architecture from training pool.
- Setup: arm D1 = Z-Mahmood's 3x256-BiLSTM + attention reimplemented
  inside our bench/gate plumbing (e23_bilstm_model.py), arm C's EXACT
  budget (batch 32, 2500 steps, fp16, AdamW wd 0.1) on v3q tokens. D2 =
  arm C itself (same steps; numbers already frozen at 48.42).
- RESULT 2026-09-17: BiLSTM loses at ZM-native lr 1e-3: 45.77/58.09/
  62.90/56.55, mean **55.83** vs arm C 48.42 (+7.41 worse). Our 4e-4 stalls
  the LSTM entirely. **NOT adopted; char transformer stays** (no user gate
  triggered). Report: research/e19_bakeoff/E23D_BILSTM_AB.md. S4 CLOSED.

## S5 — E-23e: Z-Mahmood + gold cross-agreement validator (qualtiy infrastructure)

- Why last for GAIN but first for processing order: it is cheap and reusable
  from S2 onward; it adds a data-quality filter rather than gate points
  directly. Run both models on candidate texts, flag strong disagreements
  (word-level label divergence > threshold) for human review or auto-drop.
- Also usable as a qualifier: a text BOTH models vocalize identically is
  high-confidence material (BOTH-side agreement = soft PASS).

## Order/dispatch

1. S1 (gold+QCRI v3q) - launch when GPU idle, user-gated by hours only.
2. S5 validator while S1 trains (CPU).
3. S2 data hunt while S1 trains (CPU+net).
4. S3 distill labels (needs S5's validator for quality control).
5. S4 architecture A/B (finisher; informs the NEXT gold arch).

Every stage closes with Task row / ledger row / HANDOFF bullet / MEMORY
lesson (or SKIPPED/PASS marker) and a scoped commit as established in
E-19..E-22. Report derives from the gate CSVs only (never hand-typed).

## Risks / notes

- The license question on QCRI re-redistribution remains unresolved -> only
  internal research use, no redist.
- The 'rival-quality label' caution from E-21 stands for Z-Mahmood too: its
  outputs are usable as TRAINING signal by rule (a) approval, never as refs.
- S4 needs a re-implementation or vendored copy of the BiLSTM inside our
  repo (models/e19/zmahood-src has the code, MIT-licensed) - vendoring only
  in research/.
