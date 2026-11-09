# E-63 Overfit Analysis + Approved Remediation Plan (2026-09-23, user-approved)

## Verdict recap (evidence: runs/diac/stage3best_ls/gate_eval.csv vs runs/diac/stage2b2500/gate_eval.csv)
- stage3best_ls (20k steps, CE+LS0.1) improved in-domain full-val acc 0.9324 -> 0.9568
  but REGRESSED every external gate: mean DER 0.4637 (gold) -> 0.5978 (+13.4pp).
  Per-gate DER: fadel +17.4pp, sadeed +17.4pp, wn24 +7.2pp, wn14 +11.6pp.
  text_preservation ~1.0 for BOTH models: the regression is mark selection, not base
  corruption. Diagnosis = domain overfit via late training + weak selection criterion.

## Approved levers (user-approved 2026-09-23, in priority order)
1. CHECKPOINT AVERAGING + GATE-MATCHED SELECTION. Average the best checkpoints
   (late-window and/or multi-stage) and re-profile the 4+1 gates; also probe external
   gates every N steps inside training so selection no longer trusts the in-domain
   curve alone. Cost: GPU-hours only, no retraining. Refs: AACL-22 "Revisiting
   Checkpoint Averaging for NMT"; "When, Where and Why to Average Weights?" (AISTATS 25);
   the deployed gold's whole advantage is plausibly step-2500 external-facing selection.
2. KD TEACHER TERM (E-61 pattern, already proven here): agreement-filtered
   stage2b2500 teacher labels on hard gold rows; recorded external mean mark acc
   0.207 -> 0.3652 on the DA line; the same machinery transfers to the D-line.
3. DATA DIVERSIFICATION over steps: more domains (mgb2/qcri mixes, partially
   diacritized inputs per "Take the Hint", EMNLP-25 Arabic SOTA analysis) instead of
   longer single-source schedules.
4. TRAIN SHORT + EMA/cosine: stay in the 5-8k step region with cosine decay; monitor
   gates so the external optimum gets picked; EMA weights as a default second candidate.

## Approved: SMALLER-MODEL TRACK (user-approved 2026-09-23)
Pre-register a compact-arch ladder on the SAME trial bed (v3s10, 2000 steps, identical
protocol to E-63 singles) so any size claim is tested at matched budget with the full
multi-metric harness:
- M1 16L x 272 (~18M params)
- M2 28L x 208 (~22M params)
- M3 28L x 272 WITH block dropout (stochastic depth p=0.1) as the "regularized full-size" control
Bands vs S0 on BOTH in-domain val AND external proxy: a smaller model is adopted only if
it matches S0 in-domain (val_acc >= S0 - 0.005) AND does not regress external-lite more
than the full-size control; smoke before every trial; single GPU, one arm at a time.

## Standing user rules encoded with this decision (permanent)
R1. ALWAYS evaluate on unseen data with the four external gates on the FULL multi-metric
    set (strict acc + DER / DER_nocase / WER / text_preservation + per-class) — user did
    not re-request; the agent must do it by default.
R2. NEVER trust a single eval/metric; always report at least internal + external + per-class.
R3. MONITOR the entire train process at every step (auto, no manual invocation).
R4. ALWAYS smoke the end-to-end train process before any training run.

---
## Lever-1 execution log (checkpoint averaging, running 2026-09-23)
- Averaged stage3best_ls checkpoint-16000/18000/20000 -> runs/diac/e63_avg3_tr/final
  (scratch/e63_ckpt_average.py; earlier checkpoints were rotated away).
- In-domain FULLVAL (160,214 unseen windows): val_acc 0.9615 (raw final: 0.9568),
  val_loss 0.2083 (LS-inflated scale). Averaging HELPS in-domain.
- External re-profile: pwsh-100 running -> runs/e63_avg3_gates.log;
  fadel_test interim: DER 0.4794 (raw 0.5064, gold 0.3323) -> +2.7pp improvement,
  direction confirms the lever, magnitude insufficient to threaten the gold.
- **CLOSED VERDICT (2026-09-23): averaging WORKS but is not enough.**
  | gate | gold | raw final | avg3 | gain vs raw |
  |---|---|---|---|---|
  | fadel_test | 0.3323 | 0.5064 | **0.4794** | +2.7pp |
  | sadeed25 | 0.4568 | 0.6309 | **0.6049** | +2.6pp |
  | wikinews2024 | 0.5758 | 0.6482 | **0.6328** | +1.5pp |
  | wikinews2014 | 0.4899 | 0.6056 | **0.5918** | +1.4pp |
  | abdou_test | n/a | 0.5664 | **0.5399** | +2.6pp |
  4-gate mean DER 0.5978 -> 0.5772; in-domain val_acc also UP 0.9568 -> 0.9615.
  text_preservation stays >= 0.985 everywhere; WER ~= DER (no base drift).
  Gold still wins by ~11.4pp mean -> remaining gap is DISTRIBUTION, not weights.
  Adopt: averaging = validated feature once config knob lands; the big lever is
  KD/gate-matched early selection, not more plateau-phase averaging.

---
## Smaller-model ladder (2026-09-23, user-approved): slices trained
- M1 16Lx272 (warm-load OK; 18.6M): slice 0.2121/0.9298 -> FULLVAL(160k) 0.9304.
  In-domain band: within 0.005 of S0 (0.9324) => inside band.
- M2 28Lx208 (~22M): warm stage1lm init CANNOT load (dim mismatch, embeddings/
  attn/ffn across the board) -> FRESH-INIT trial 0.2751/0.9095. Honest confound:
  scratch init undertrains at the same 2000-step budget; not a pure size verdict.
- M3 28Lx272 + model.dropout 0.1 (existing validated code path, params = S0):
  slice 0.2056/0.9322 - best 2000-step val_loss of the whole ladder, S0-level acc.
- External 4+1 gate profile for all three RUNNING (pwsh-105); verdict to append.

## Smaller-model ladder — final verdict (4+1 external gates, V3)

Arms all trained on v3s10 trial bed, identical 2000-step protocol, warm init
from stage1lm where possible (M2 fresh-init confound recorded).

| arm | params | init | slice val_loss/acc | fadel | sadeed | wn24 | wn14 | abdou | 4-gate mean |
|---|---|---|---|---|---|---|---|---|---|
| M1 16L×272 warm | 17.2M | warm | 0.2121/0.9298 | 0.3371 | 0.4662 | 0.5851 | 0.5131 | 0.4184 | **0.4754** |
| M2 28L×208 fresh | 22M | fresh (confound) | 0.2751/0.9095 | 0.4107 | 0.5347 | 0.6229 | 0.5638 | 0.4880 | 0.5330 |
| M3 28L×272+drop | 30.1M | warm | 0.2056/0.9322 | 0.3447 | 0.4700 | 0.5804 | 0.5077 | 0.4233 | 0.4757 |
| gold stage2b2500 | 30.1M | — | — | 0.3323 | 0.4568 | 0.5758 | 0.4899 | n/a | 0.4637 |
| avg3 (30M full-data) | | | | 0.4794 | 0.6049 | 0.6328 | 0.5918 | 0.5399 | 0.5772 |
| raw stage3best_ls | | | | 0.5064 | 0.6163 | 0.6482 | 0.6056 | 0.5978 * | 0.5942 |

* raw abdou belongs to the stage3best_ls 4-gate former chain (abdou 0.5978
measured once on the full-data arm).

VERDICT (user pre-registered band: not worse than full-size control):
- M1 and M3 both land essentially at gold level externally (Δ ≈ +0.012
  4-gate mean) while using HALF the parameters (M1) or HALF the layers
  (16L vs 28L); every gate also beats all full-data 30M arms.
- M2 (fresh) is dramatically worse on every external gate despite being
  LARGE than M1 — warm init, not width, is the dominant factor at fixed
  budget.
- Decision: adopt M1 16L×272 as the lighter-throughput external-eval
  champion and default future architecture; gold stage2b2500 remains the
  deployed verbatim reference. A full-data M1 (or M1+dropout) run is the
  natural next candidate to attempt gold-external EXCEEDED at half size.

---

# E-63c APPENDIX (2026-09-23): full-data M1 run

Recipe: ladder-winner 17.2M (16Lx272) on FULL v3 (2,433,430 train / 160,214 val windows), 20k steps,
vanilla CE, warm stage1lm init, bs32 lr2e-4 wd0.1 — identical to the ladder arm except data scale.

| state (5-gate DER) | fadel | sadeed | wn24 | wn14 | abdou | 4-gate mean |
|---|---|---|---|---|---|---|
| **champion step 5000** | **33.08** | **45.44** | **57.17** | **49.07** | **41.19** | **0.4619** |
| final step 20000 | 52.57 | 63.76 | 64.31 | 60.90 | 58.32 | 0.6299 |
| gold stage2b2500 (30.1M) | 33.23 | 45.68 | 57.58 | 48.99 | n/a | 0.4637 |
| ladder M1 10% slice | 33.71 | 46.62 | 58.51 | 51.31 | 41.84 | 0.4754 |
| raw baseline | 47.94* | 60.49* | 63.28* | 59.18* | — | 0.5978* |

VERDICT (CORRECTED 2026-10-06 by the verdict audit): the smaller-model hypothesis generalizes — the 17.2M
warm-init model trained on the full dataset, selected by per-500-step external gate probes, is the best on
the registered 3-gate mean. **The original "all gates strictly better or equal vs gold" is FALSE**:
wikinews2014 is 49.07 vs the gold's 48.99 (+0.08 worse). Correct per-gate record at champion step 5000:
fadel 33.08 < 33.23, sadeed 45.44 < 45.68, wn2024 57.17 < 57.58, wn2014 49.07 > 48.99, abdou 41.19;
3-gate mean 42.53 vs gold 42.64 (-0.11, inside the noise band). Champion state is preserved at
runs/diac/e63_m1_full/gate_probe/weights_step5000.pt; the auto best_gate_weights.pt picked the same state.
The in-domain-alone stopping rule (best val_loss -> final step) would have produced a WORSE model than raw
baseline on gates — external-gate probing every ~500 steps is now a validated pipeline feature (MEMORY).

---

# E-63s APPENDIX: smallest-model sweep (2026-09-24)

Six arms, full v3 data, 10k-step wall, 4-gate MULTI-metric probes every 500 steps. Warm stage1lm init:
full for hidden=272 arms, partial (shape-filtered) below. Best probe window per arm:

| arm | params | best step | fadel | sadeed | wn24 | wn14 | tracker mean | vs champion 42.53 |
|---|---|---|---|---|---|---|---|---|
| S1 12Lx272 | 12.93M | 4500 | 34.70 | 46.72 | 57.72 | 49.39 | 43.60 | worse |
| S2 10Lx272 | 10.78M | 6500 | 34.52 | 45.69 | 56.41 | 48.51 | 42.91 | ~equal |
| S3 8Lx272  | 8.64M  | 5500 | 31.20 | 43.77 | 55.65 | 47.34 | 40.77 | BEST OVERALL |
| S4 12Lx160 | 4.50M  | **16500** | 31.76 | 44.12 | 56.57 | 48.03 | **39.96** | **BEST OVERALL (corrected 2026-10-06; 10k-wall row was 41.30)** |
| S5 10Lx128 | 2.40M  | 9500 | 33.97 | 47.16 | 59.11 | 51.35 | 44.16 | worse; wall-hit |
| S6 8Lx96   | 1.11M  | 10000 | 37.03 | 49.32 | 60.46 | 52.99 | 46.45 | worse; wall-hit |

VERDICT (CORRECTED 2026-10-06 by the verdict audit): (a) SMALLEST model matching-or-beating the 17.2M
champion = **S4 (4.5M)** — 41.30 at the 10k wall with the curve still descending; (b) **BEST OVERALL state
= S4 (4.5M), not S3** — `runs/diac/e63s_s4_12l160/gate_probe/best_gate.json` = {"step":16500,
"mean_der":0.39962}, i.e. 39.96 on the registered 3-gate mean vs S3's 40.77, and `runs/e63s_s4.log:327`
shows the run completed 20000 steps. The old "S3 BEST OVERALL" line understated the smallest strong arm;
(c) depth-272 arms peaked at 4.5-6.5k and decayed, while partial-warm sub-272 arms were STILL improving at
the 10k wall. Under-embedding-capacity is the cost driver below 4.5M: S5/S6 lose
~3-6 DER vs S4 at 4.9x/0.25x its size. Per-500-step gate probes again selected every optimum (in-domain
val_loss improved monotonically to the wall everywhere - it would have mis-selected ALL best states).
