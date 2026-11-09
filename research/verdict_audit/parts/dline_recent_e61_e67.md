# Verdict audit - D-line recent (E-061, E-063, E-064, E-065, E-066, E-067)

Auditor: primary agent (this session). Date: 2026-10-06.
Scope: every recorded verdict/headline number in the recent D-line records, re-checked
against the raw artifacts on disk. Both error directions were hunted: false FAILs
(metric read backwards) and false PASSes (unsupported wins).

Method: re-derived each headline from the cited raw artifact (`runs/diac/*/gate_eval.csv`,
`runs/diac/*/monitor/log.jsonl`, `scratch/prof_v21/*.metrics.json`) rather than from
prose. GATE_EXTERNAL (train.py:39) = (fadel_test, sadeed25, wikinews2014); **lower DER is better**.
A "3-gate mean" below is over GATE_EXTERNAL; a "4-gate mean" adds the contaminated
wikinews2024 stamp (the two are NOT interchangeable and are mixed in the registry - see D-1).

## Audit labels

| id | recorded verdict | audit label | one-line evidence |
|---|---|---|---|
| E-061 | partial (MIXED, 2/3 gates) | **CONFIRMED** | 4-gate mark_accuracy mean 0.3654 recomputed from `scratch/prof_v21/*.metrics.json` vs A3 0.207; v21a/v21c match |
| E-063 | partial (S4 external overfit, gold stays) | **CONFIRMED** | gold 3gate 0.4264 @2500 vs S4 full `stage3best_ls` 0.5810 @20000; M1 0.4253, M2 0.5030, M3 0.4408, avg3 0.5587 |
| E-064 | PASS both arms, A2 winner | **CONFIRMED (AMBIGUOUS metric set)** | A1 3gate 0.4615 @18000 vs A2 0.4577 @38000 = recorded 46.15 vs 45.77; val_loss 0.23969/0.22164/0.25892 exact |
| E-065 | completed; F1 0.4067, F2 peak 0.3974 then overfit | **CONFIRMED** | 0.3974 @115000, 0.6343 @180000, 0.5768 @200000, val_loss best 0.14767 @195000 - all exact |
| E-066 | in_progress ("R1 PROGRESS step 47300") | **STALE + unreported result** | R1 log ends step 109500, no summary; real best 3gate **0.4205 @40000** |
| E-067 | in_progress / none (was FAIL) | **INVERTED - corrected this session** | P1 0.4299 @32000 < C0 0.4577 @38000 = 6.07% better, not a regression |

## Detail

### E-061 - CONFIRMED (all three sub-verdicts reproduce)
- v21a: in-domain rebuilt-val markpos 0.6824 vs gate 0.6696 - recorded in `research/EXPERIMENTS.md:875`. PASS as stated.
- v21b: per-gate mark_accuracy read back from artifacts: fadel 0.3487, sadeed 0.3703, wn24 0.3730, wn14 0.3696
  => 4-gate mean **0.3654**, matching the recorded 0.3652 to rounding. Control = attempt-3 0.207, gate = 0.227.
  PASS as stated. **This is the strongest external transfer on record for the D-line.**
- v21c: per-class floor violation at shadda_fatha 0.00 / shadda_dammatan 0.01 - FAIL as stated.
- MIXED -> `partial` is the honest label. The "0.207 -> 0.3652" phrasing also appears in
  `research/e63_overfit_report_2026-09-23.md:19` as "the DA line" - same numbers, same line, no conflation.
- Minor: the registry `evidence` list omits the metrics artifact; add `scratch/prof_v21/` (fragile, scratch is
  declared throwaway). Recommend promoting the five `*.metrics.json` into a durable path.

### E-063 - CONFIRMED (external regression real, gold really stays)
- Gold `stage2b2500` 3gate 0.4264 @2500 (matches the recorded 0.426339/0.4264).
- Full-data S4/smoothing finalist `stage3best_ls` 3gate **0.5810** @20000 - a 15.5-point regression, i.e. the
  external gates did regress as recorded, and `stage2b2500` is correctly still the deployed gold.
- Arm checkpoints, **all independently re-scored** (2026-10-06): M1 `e63_m1_full` best 3gate **0.4253 @5000**
  (re-derived from `gate_probe/step5000_*_pred{,.ref}.txt` with `tools/score_probe.py` and from its own CSV row:
  fadel 0.3308 / sadeed 0.4544 / wn14 0.4907), M2 0.5030, M3 0.4408, avg3 0.5587, `stage3best_ls` 0.5810.
  **Correction to my own earlier wording**: M1's 0.4253 is *marginally better* than the gold's 0.4264 (0.11 pp -
  inside the noise band, so "gold stays deployed" remains defensible, but "none beats the gold" is too strong).
  - `e63_m1_tr` (0.4388) is a **different** run from `e63_m1_full` (0.4253) - the registry uses "M1" for both.
    Naming trap; always cite the run dir.
- **Clean-instrument cross-check (new, 2026-10-06)**: `tools/score_gate_bench.py` re-scored the raw
  `gate_bench/*.prd.txt` pairs with `eval_metrics.score_corpus` for all five runs that have them. Every
  3-gate mean reproduces the registered number to 4 decimals: `stage3best_ls` 0.5810, `e63_m2_tr` 0.5030,
  `e63_m3_tr` 0.4408, `e63_avg3_tr` 0.5587 (and `e63_m1_tr` 0.4388 = its own CSV exactly). Evidence:
  `evidence/clean_bench_e63.json`, `evidence/m1_full_probe5000.json`.
- **Schema drift noticed**: `e63_m2_tr`, `e63_m3_tr`, `e63_avg3_tr` gate CSVs use a legacy header
  `phase,src,der,wer,den,pres` while newer runs use `step,gate,DER,WER,DER_nocase,text_preservation,lines,pred`.
  Any aggregator that assumes the modern header KeyErrors on the legacy files (this happened during the audit).
  Also `e63_avg3_tr` carries 5 gates (adds `abdou_test`), so a naive mean over all rows is a 5-gate mean.
- The single-arm S0-S5 verdicts ("S4 wins the sliced ladder") rest on trial-bed val curves, not on gate CSVs:
  `e63_s0_tr`..`e63_s5adj_tr` have **no** `gate_eval.csv`. Not re-verifiable from gates; treat as
  val_loss-only evidence.

### E-064 - CONFIRMED numbers, AMBIGUOUS gate set (the one real hygiene defect here)
- Reproduced: A0 `e63s_s7_18l64_adamw` best_val_loss **0.25892** @20000, val_acc 0.9136;
  A1 `..._lr4e4` **0.23969** @20000, val_acc 0.9204, best 3gate **0.4615** @18000;
  A2 `..._ext40k` **0.22164** @40000, val_acc 0.9260, best 3gate **0.4577** @38000.
  All match the record's 0.23969/0.22164/0.25880-0.2589 and 46.15/45.77 exactly.
- **Defect D-1**: the record's finding text reports "A1 ... best gate mean 49.80 @18000 vs A0's 51.76 @20000"
  while the verdict reports "A1 46.15 vs A2 45.77". Both are arithmetically correct but over **different
  gate sets**: 49.80 = A1's FOUR-gate mean @18000 (0.4980) and 51.76 = A0's four-gate mean @20000 (0.5176),
  whereas 46.15/45.77 are THREE-gate means. A reader comparing 46.15 with 49.80 sees a phantom contradiction.
  Fix: label every gate mean as 3-gate (GATE_EXTERNAL) or 4-gate, and use only the 3-gate as the registered metric.
- The "val improves while gates regress" finding for A1 (val 0.2431 -> 0.2397, gates fadel 36.43 -> 38.75 etc.)
  is directionally correct (higher DER = worse) and is the useful durable lesson.

### E-065 - CONFIRMED (every headline exact)
- F1 @100000: val_loss 0.1671, val_acc 0.9449, best 3gate 0.4067. Matches.
- F2 peak 3gate **0.3974 @115000** (best in the project); degradation 0.5446 @170000, **0.6343 @180000**,
  **0.5768 @200000**; val_loss best 0.14767 @195000. All match the record exactly.
- Conclusion "internal val is not a usable selection metric past the peak; best_gate_weights.pt @115000 is the
  deployment checkpoint" is supported. `runs/diac/e63s_s7_18l64_adamw_fullv3/gate_probe/best_gate_weights.pt`
  exists (4,385,294 bytes) alongside per-step weights and per-step predictions.

### E-066 - STALE RECORD; the registered question is already answered and the answer is not recorded
- Registry status is `in_progress` and the verdict text still reads "R1 PROGRESS 2026-10-05 step 47300/200000".
- Actual state on disk: `runs/diac/e63s_s7_18l64_reg_d02_ls01/monitor/log.jsonl` has on_start + 21 on_eval
  + 1143 on_step events and **ends at step 109500 with an on_save and no summary.json** - the run was
  interrupted (machine/job stop), never completed and never formally closed. No `early_stop`/`final` event.
- Real result: best 3gate **0.4205 @40000**, then flat 0.42-0.43 to 105000.
- Matched-step comparison against the F2 baseline (11 shared probe steps):
  R1 wins steps 10000-40000 (-0.0869 at 10000, -0.0320 at 20000, -0.0161 at 30000, -0.0094 at 40000),
  loses 50000-105000 (+0.0043, +0.0049, +0.0082, -0.0016, +0.0029, +0.0197, +0.0191); wins 5/11.
- So the registered question ("does dropout 0.2 + label smoothing 0.1 **raise the external-gate peak** and
  **delay** the full-v3 overfit") answers **PASS on delay, FAIL on peak**: 0.4205 > 0.3974. The regularization
  kept the gates flat instead of climbing to 0.63, but the ceiling got worse, not better.
- **Defect D-2 (new, important)**: with label smoothing 0.1 the val_loss floor rises from ~0 to ~0.56
  (entropy of the smoothed 15-class target), so R1's val_loss (**0.6789** @105000, val_acc 0.9491) is NOT
  comparable with the unregularized runs' val_loss (F2 0.1477, val_acc 0.9514). Any arm-to-arm val_loss
  comparison across a label-smoothing arm is invalid; compare val_acc or gates instead.
  `diacritizer/src/model.py:282` passes `label_smoothing=self.loss_label_smooth` straight into the criterion.
- Action: close E-066 with FAIL-on-peak/PASS-on-delay and mark R1 `interrupted`, or re-run to completion if the
  delay result needs the full 200000-step curve.

### E-067 - INVERTED, corrected earlier this session
- Recorded FAIL ("warm start HURTS", -6.1%) from a backwards metric read. Artifacts: P1 best 0.4299 @32000 vs
  C0 0.4577 @38000 = 6.07% relative BETTER; better at all 10 matched steps; val_acc 0.9338 vs 0.9260.
  Corrected to PASS. The C0-R confirmation run is in flight (see below).

## Cross-cutting defects for the consolidation

- **D-1 (metric-set ambiguity)**: 3-gate vs 4-gate means mixed without labels (E-064, and any other record
  quoting "gate mean"). Fix: registered metric = 3-gate GATE_EXTERNAL; always label.
- **D-2 (label-smoothing loss floor)**: val_loss is not comparable across smoothing arms (E-066 R1, E-063 S4).
- **D-3 (schema drift)**: legacy `phase,src,der,...` vs modern `step,gate,DER,...` gate CSVs.
- **D-4 (unclosed runs)**: E-066 R1 stopped at 109500/200000 with no summary and stayed `in_progress`; the
  registry needs an `interrupted` end state or a closure note.
  Proof it was interrupted, not an undocumented early stop: `diacritizer/scripts/train.py` writes
  `mon.on_end(...)` -> `monitor/summary.json` on BOTH exit paths (early stop at :431 and natural completion at
  :469) and exports `final/model.pt` at :458-465. R1 has **neither** `summary.json` nor `final/`, and its log
  ends mid-run on an `on_save` at step 109500 - the process was killed. Recoverable evidence survives though:
  `best.pt` (best val_loss) and `gate_probe/best_gate_weights.pt` (gate-best, step 40000) are both present, so
  the result is usable without a re-run.
- **D-5 (evidence path fragility)**: E-061's headline numbers live in `scratch/` (declared throwaway) and the
  registry's evidence list does not name them.

## Live context

- `e67_c0_rep` (C0-R, the unseeded-init-draw replicate of E-064 A0/C0) is training as background job
  `pwsh-134` under the one-job rule. Verified: single repo python (PID 20268), GPU 774 MiB, no co-run,
  R1's process is gone. Through step 4000 C0-R is **better than A0 at every matched eval**
  (val_loss 0.7137/0.5471/0.4749/0.4340 vs A0 0.7520/0.5746/0.4997/0.4499; acc +0.007..+0.015),
  i.e. the unseeded init draw already moves the curve by more than the E-067 2% decision band.
  This is the measurement that decides whether the P1 win survives.
