# E-19..E-23 D-line bake-off - verdict audit

Audit scope: the nine bake-off reports under research/e19_bakeoff/, the E-19..E-23 rows in
research/EXPERIMENTS.md, and the priors prose in research/experiments.yaml (lines 983-987 and
1067-1070). Read-only: no training, inference, or eval command was run.

Metric direction (registered). Every gate number in this audit is DER %, and LOWER IS BETTER.
Authority: diacritizer/scripts/train.py:39 defines
GATE_EXTERNAL = (fadel_test, sadeed25, wikinews2014), explicitly excluding wikinews2024 as
contaminated; diacritizer/scripts/train.py:204-220 selects the best checkpoint by the LOWEST mean
DER over GATE_EXTERNAL (mean_der < prev), written to runs/diac/<run>/gate_probe/best_gate.json.
Per-run gate rows live in runs/diac/<run>/gate_eval.csv.

How numbers were checked. Each headline number was re-derived from the cited artifact by
re-implementing the registered scorer (diacritizer/src/eval_der.py: base letter + mark-string
comparison, multi-ref via tab-separated alternatives, equal-length positional alignment,
word-weighted aggregate) and re-scoring the cited prediction/reference text. Scorer validated by
reproducing the whole known table: models/e19/fadel2500.gold.raw.pred.txt vs
models/e19/inputs/fadel2500.ref.txt = 33.23 (gate CSV 33.23); the four E-20 gold+cache predictions
= 32.88 / 45.13 / 48.65 / 40.90; the four Z-Mahmood predictions = 4.35 / 18.85 / 29.73, all exact.

## The baseline error that drives the E-23 findings

Gold wikinews2024 DER is 57.58, not 48.65.
- runs/diac/stage2b2500/gate_eval.csv (step 2500, gate wikinews2024) DER = 0.5758289374529013.
- research/wn2024_full_metrics.md: stage2b2500 der_registered 57.57 / der_anyalt 57.66.
- Re-scoring models/e19/wn2024.gold.raw.pred.txt vs models/e19/wn2024.gold.raw.pred.ref.txt = 57.57.
- Corroboration: research/EXPERIMENTS.md line 47 (E-17) records the stage2final wn24 peak as 57.5;
  E-12/E-14 wn2024 sit at 60-63 (research/EXPERIMENTS.md lines 43-44).
48.65 is gold's wikinews2014 + word-cache value: research/e19_bakeoff/E20_CACHE_SMALLMODEL.md row
"wn2014 | 48.99 | 48.65" (reproduced exactly). E-21 copied it into the wn2024 column, E-23a..E-23d
inherited it as the gold yardstick, and it reached the plan priors. Correct gold baselines:
4-gate mean 46.37; registered 3-gate GATE_EXTERNAL mean 42.64.

Equal-step and best-step gate tables (all from gate_eval.csv / best_gate.json):
- gold stage2b2500 @2500: fadel 33.23 sadeed 45.68 wn2024 57.58 wn2014 48.99, mean4 46.37, mean3EXT 42.64.
- e23a @2500: 33.54 / 45.33 / 55.71 / 47.78, mean4 45.59, mean3EXT 42.22.
- e23a @2000 (registered best, best_gate mean_der 0.4020): 32.87 / 43.36 / 54.65 / 44.37, mean4 43.81, mean3EXT 40.20.
- micro arm C @2500 (registered best): 36.44 / 48.67 / 58.13 / 50.11, mean4 48.34, mean3EXT 45.07.
- micro arm A @2500 (registered best): 38.16 / 51.40 / 60.96 / 53.87, mean4 51.10, mean3EXT 47.81.
- micro arm B best @2000: mean3EXT 50.08.
- e23b_t @2500: 38.69 / 50.70 / 59.93 / 52.55, mean4 50.47; best @2000 mean3EXT 47.01.
- e23c_zm @2500: 39.88 / 52.72 / 61.44 / 56.15, mean4 52.55; best @2000 mean3EXT 48.18.
- e23d_bilstm_lr1e-3 @2500: 45.77 / 58.09 / 62.90 / 56.55, mean4 55.83; best @2000 mean3EXT 53.68.

## E-19 - external bake-off (research/e19_bakeoff/RESULTS.md)

Recorded verdict: Z-Mahmood QUALIFIES (a); Etherll FAILED; mishkala rule-(b) sadeed specialist;
Fine-Tashkeel FAILS. Metric: DER, lower better. Compared on OUR gate slices (gold = stage2b2500).

- Z-Mahmood 4.35 / 18.85 / 29.73: CONFIRMED exactly (models/e19/fadel2500.zmahood.pred.txt,
  sadeed2500.zmahood.pred.txt, wn2014.zmahood.pred.txt vs models/e19/inputs/{fadel2500,sadeed2500,wn2014}.ref.txt).
  abdou held-out 12.60 CONFIRMED (models/e19/abdou_heldout.zmahood.pred.txt).
- mishkala 39.37 / 35.79 / 50.24: CONFIRMED exactly.
- Gold reference row in RESULTS.md line 5 says fadel 33.20 and abdou 41.64; artifacts give
  33.23 and 41.46 (MISCITED; E-20/E-21 use the correct 33.23/41.46).
- Etherll wn2014 ~97.0: CONFIRMED (models/e19/wn2014.etherll.pred.txt and
  models/e19/etherll.wn60.pred.txt score 97.09 / 96.60). Etherll fadel 49.7 (20-line probe):
  NO_ARTIFACT - no fadel Etherll prediction exists under models/e19/.
- Fine-Tashkeel FAILS: INVERTED (FALSE NEGATIVE). The recorded fadel 91.60 with "preservation 10%"
  and sadeed 70.55 are reproduced ONLY by scoring the 256-line and 500-line slice predictions against
  the FULL 2500-line and 1612-line reference files, so every unscored trailing line counts as an
  error: fadel256 pred vs full inputs/fadel2500.ref.txt = 91.60, preservation 10.22% (recorded
  91.60 / 10%); sadeed500 pred vs full inputs/sadeed2500.ref.txt = 70.55 (recorded 70.55); the
  full-file wn2014 run is unaffected either way = 37.62 (as recorded). Correctly sliced to their own
  line counts, the SAME cited artifacts score fadel 21.33, sadeed 23.27, wn2014 37.62 - i.e. it beats
  gold by 11.90 / 22.41 / 11.37 DER on all three gates and would satisfy rule (a) (>5 DER), not fail.

## E-19b - contamination audit (research/e19_bakeoff/CONTAMINATION.md)

Recorded verdict: PASS with notes. Metric: DER, lower better.
- cache-ON == cache-OFF: CONFIRMED (models/e19/*.zmahood_cache.pred.txt = 4.35 / 18.86 / 29.73).
- Word-order shuffle probe: CONFIRMED exactly - gold 43.68 / 48.74 / 55.51, Z-Mahmood 30.96 / 38.69 / 50.47
  (models/e19/*.shuf.gold.pred.txt, *.shuf.zmahood.pred.txt vs models/e19/inputs/*.shuf.ref.txt).
- abdou heldout Z-Mahmood 12.60 CONFIRMED; gold quoted 41.64 is MISCITED (artifact 41.46).
- The cache-overlap table (sentence hits 0-0.3%, word coverage 4.2-11.8%) was NOT independently
  reproduced - it needs a scan of models/e19/zmahood-src/word_cache.json.gz via
  scratch/e19/zm_contamination.py; the cited cache artifact exists.

## E-20 - cache / small-model trim (research/e19_bakeoff/E20_CACHE_SMALLMODEL.md)

Recorded verdict: cache helps but tiny (~0.3-0.6 DER). Metric: DER, lower better.
- CONFIRMED exactly from models/e19/*.gold.raw.pred.txt vs *.goldcache.pred.txt:
  fadel 33.23 -> 32.88 (-0.35), sadeed 45.68 -> 45.13 (-0.55), wn2014 48.99 -> 48.65 (-0.34),
  abdou 41.46 -> 40.90 (-0.55). Note this is the artifact where 48.65 legitimately belongs to wn2014.
- Param math (30.11M / 2.98M / 3.14M / 2.00M): UNSUPPORTED - prose only; no param artifact cited and
  build_from_config was not run. E-23b independently states 2,976,128 for the 12-128-512 config.
- QCRI corpus facts (32,834 articles / 5.07M words): not reproduced.

## E-21 - micro arms + cache test (research/e19_bakeoff/E21_MICRO.md)

Recorded verdict: arm A WINS arm B; train-time cache adds ~nothing (FALSIFIED); QCRI labels are good
but machine. Metric: DER, lower better.
- Arm A vs B: CONFIRMED. Registered best (GATE_EXTERNAL 3-gate, best_gate.json) A 47.81 @2500 vs
  B 50.08 @2000; at the quoted step 2500 the 4-gate means are 51.10 vs 53.59 (all rows match
  runs/diac/micro_a/gate_eval.csv and runs/diac/micro_b/gate_eval.csv).
- Train-time cache test: CONFIRMED. Re-scoring runs/diac/micro_a/gate_probe and
  runs/diac/micro_b/gate_probe e21_<gate>_pred.txt vs e21_<gate>_cache.pred.txt against
  e21_<gate>_pred.ref.txt gives deltas fadel -0.025/-0.027, sadeed -0.002/-0.002, wn2024 0.000,
  wn2014 0.000 for A/B; the quoted table (38.16->38.13 etc.) matches exactly.
- QCRI validator DER: CONFIRMED. models/e19/qcri.gold_vs.pred.txt vs models/e19/qcri.sample.ref.txt
  = 45.01; models/e19/qcri.zm_vs.pred.txt = 56.10 (recorded 45.0 / 56.1).
- Gold row MISCITED: the E-21 table lists gold wikinews2024 = 48.65 with footnote "gold cache-merged",
  but 48.65 is gold's wn2014+cache value and 48.99 is gold wn2014 raw; gold wn2024 is 57.58. Hence the
  gold mean ~44.1 is wrong (true 46.37) and "micro A lands ~7-8 DER behind gold" is inflated - the
  artifact gap is 4.73 (4-gate) or 5.17 (registered 3-gate). This is the origin of the 48.65 error.
- Artifact gap: cited runs/diac/micro_*/cache_eval.json does NOT exist (NO_ARTIFACT); the cache claim is
  nonetheless checkable from the cited e21_*_cache.pred.txt files. runs/diac/micro_a_lm/final and its
  gate CSV also do not exist (only runs/diac/micro_a_lm/best.pt); micro_a/micro_b/micro_c/final do exist.

## E-22 - micro arm C (+QCRI) (research/e19_bakeoff/E22_QCRI_WEAK.md)

Recorded verdict: IMPROVES - arm C beats arm A on all four gates. Metric: DER, lower better.
- CONFIRMED directionally on the artifact runs/diac/micro_c/gate_eval.csv @2500 vs micro_a @2500:
  fadel -1.72, sadeed -2.73, wn2024 -2.83, wn2014 -3.76 (all better). MISCITED minor: the recorded
  means are 51.10 -> 48.42 (delta -2.68); the artifact mean is 48.34 (delta -2.76). The E-23* reports
  then reuse 48.42 as the arm-C yardstick.

## E-23a - gold 30M + QCRI (research/e19_bakeoff/E23A_GOLD_V3Q.md)

Recorded verdict: MIXED - "REGRESSED on wikinews2024 by ~7 DER and hurt the mean", NOT-ADOPTED.
Metric: DER, lower better. Comparison: E-23a vs gold stage2b2500.
- Per-gate E-23a values and trajectory CONFIRMED from runs/diac/e23a_gold_v3q/gate_eval.csv.
- The verdict is INVERTED because the gold wn2024 baseline is the wn2014+cache value (48.65) rather
  than 57.58 (see baseline section). Against the real artifact baseline:
  * step 2500: E-23a wn2024 55.71 vs gold 57.58 = 1.87 BETTER (recorded "+7.06 WORSE"); mean4
    45.59 vs 46.37 = 0.78 BETTER (recorded "+1.45 WORSE"); only fadel is worse, by 0.31.
  * registered best step 2000 (runs/diac/e23a_gold_v3q/gate_probe/best_gate.json mean_der 0.40201):
    32.87 / 43.36 / 54.65 / 44.37 - BETTER than gold on ALL FOUR gates; registered 3-gate
    GATE_EXTERNAL mean 40.20 vs gold 42.64 (2.44 better).
- The doc's own decision rule ("adopt only if gold's wn2024/wn2014/fadel/sadeed improve -> they
  don't all improve") holds only at the final step, where fadel is 0.31 worse; at the selected best
  checkpoint best.pt (step 2000) all four gates improve. The recorded rationale and the NOT-ADOPTED
  mechanism are therefore false even though the two-probe means are close.
- Documented self-note (line 18) admits the 48.65 came from the E-21 cache experiments.

## E-23b - classical S2 expansion (research/e19_bakeoff/E23B_CLASSICAL_S2.md)

Recorded verdict: worse on every gate, NOT adopted; mean 48.72 vs arm C 48.42 (+0.30).
Metric: DER, lower better.
- Direction CONFIRMED: at step 2500 arm T is worse on every gate (38.69/50.70/59.93/52.55 vs
  36.44/48.67/58.13/50.11); registered best 3-gate 47.01 (@2000) vs arm C 45.07, i.e. 1.94 worse.
- MISCITED: the recorded mean 48.72 does not exist in runs/diac/e23b_t/gate_eval.csv. The step-2500
  row averages to 50.47, not 48.72 (50.47-48.34 = +2.13 vs arm C's true mean). E-23b's understated
  mean therefore flatters the losing arm, but does not change the verdict.

## E-23c - ZM distillation arm E (research/e19_bakeoff/E23C_ZM_DISTILL.md)

Recorded verdict: +4.13 worse, NOT adopted. Metric: DER, lower better.
- CONFIRMED from runs/diac/e23c_zm/gate_eval.csv @2500: 39.88 / 52.72 / 61.44 / 56.15, mean 52.55;
  all four deltas vs arm C are positive (worse). Minor MISCITED: the trajectory's "51.5 (2000)" is
  actually 51.22 in the CSV; the CSV row for step 1000 has only the fadel gate.
- Mechanism claim (ZM label-style conflict) is interpretive, not audited here.

## E-23d - BiLSTM A/B (research/e19_bakeoff/E23D_BILSTM_AB.md)

Recorded verdict: BiLSTM LOSES to arm C (+7.41) and to arm A; NOT adopted. Metric: DER, lower better.
- CONFIRMED from runs/diac/e23d_bilstm_lr1e-3/gate_eval.csv: trajectory 62.84/58.21/57.16/56.12/55.83
  exactly; +7.49 vs the artifact arm-C mean 48.34 (recorded +7.41 vs the recorded 48.42); loses to
  arm A 51.10. The void runs/diac/e23d_bilstm arm (~0.997) is correctly labeled VOID.
- FALSE POSITIVE sub-claim: "It does beat arm E (52.55)". Arm E is better (52.55 < 55.83), so D1
  LOSES to arm E. The sentence's sign is inverted under the registered lower-is-better metric.
- MISCITED: the E-23d yardstick table lists gold as 44.14; the artifact gold mean is 46.37
  (using gold wn2024 57.58).

## Prior propagation (docs/plans/2026-10-05-diacritization-generalization-path.md, research/experiments.yaml)

- T3 / experiments.yaml line 1067 "E-23d +7.41": direction CONFIRMED (magnitude ~+7.5 against the
  artifact arm-C mean).
- T4 / experiments.yaml lines 1068-1069: the E-23a "+1.45 worse / machine data hurt the saturated
  gold" prior is INVERTED - E-23a is better than gold on the mean and, at the registered best step,
  on all four gates. The E-23b (worse every gate) and E-23c (+4.13) priors are CONFIRMED.
- T5 / line 1069-1070 cache priors (-0.3..-0.6 gold, ~0 micro): CONFIRMED.
- T8 / line 1087 "E-23c +4.13": CONFIRMED.
- The generalization-path doc's GATE_EXTERNAL definition (3 gates, wn2024 excluded) is consistent with
  train.py:39 and is the metric under which E-23a is a win; the doc's T4 prior nonetheless carries the
  false E-23a sign because it was taken from the E-19..E-23 narrative rather than the artifacts.

## Recheck table

id | recorded verdict | metric+direction | cited artifact | recheck | corrected one-line verdict
---|---|---|---|---|---
E-19 Z-Mahmood | QUALIFIES (a) | DER, lower better | models/e19/*.zmahood.pred.txt + models/e19/inputs/*.ref.txt | CONFIRMED | 4.35 / 18.85 / 29.73, abdou 12.60 reproduced exactly.
E-19 gold reference | reference | DER, lower better | models/e19/*.gold.raw.pred.txt | MISCITED | true fadel 33.23 (not 33.20), abdou 41.46 (not 41.64).
E-19 Etherll | FAILED | DER, lower better | models/e19/wn2014.etherll.pred.txt, models/e19/etherll.wn60.pred.txt | CONFIRMED | wn2014 96.6-97.1 reproduced.
E-19 Etherll fadel 49.7 | FAILED | DER, lower better | (none) | NO_ARTIFACT | no fadel Etherll prediction artifact.
E-19 mishkala | rule-(b) specialist | DER, lower better | models/e19/*.mishkala.pred.txt | CONFIRMED | 39.37 / 35.79 / 50.24 reproduced.
E-19 Fine-Tashkeel | FAILS | DER, lower better | models/e19/fadel256.finetashkeel.pred.txt, models/e19/sadeed500.finetashkeel.pred.txt | INVERTED | slice-vs-full-ref length bug; correctly sliced FT 21.33 / 23.27 / 37.62, beats gold on all three gates.
E-19b cache-ON==OFF | PASS | DER, lower better | models/e19/*.zmahood_cache.pred.txt | CONFIRMED | identical to no-cache.
E-19b shuffle probe | PASS | DER, lower better | models/e19/*.shuf.*.pred.txt + models/e19/inputs/*.shuf.ref.txt | CONFIRMED | 30.96 / 38.69 / 50.47 vs 43.68 / 48.74 / 55.51 exact.
E-19b abdou heldout | PASS | DER, lower better | models/e19/abdou_heldout.*.pred.txt | MISCITED | ZM 12.60 confirmed; gold is 41.46 not 41.64.
E-20 cache gain | tiny gain (~0.3-0.6) | DER, lower better | models/e19/*.gold.raw.pred.txt + *.goldcache.pred.txt | CONFIRMED | exact deltas; this is where 48.65 = wn2014+cache legitimately lives.
E-20 param math | PASS | params | (prose only) | UNSUPPORTED | not reproduced; 2.98M corroborated by E-23b 2,976,128.
E-21 arm A wins B | PASS | DER, lower better | runs/diac/micro_a/gate_eval.csv + micro_b | CONFIRMED | registered 47.81 vs 50.08; quoted 51.10 vs 53.59.
E-21 cache adds ~nothing | FALSIFIED | DER, lower better | runs/diac/micro_{a,b}/gate_probe/e21_*_cache.pred.txt | CONFIRMED | deltas -0.027..0.000 reproduced for both arms.
E-21 gold row wn2024 48.65 | reference | DER, lower better | runs/diac/stage2b2500/gate_eval.csv | MISCITED | 48.65 is gold wn2014+cache; gold wn2024 is 57.58, gold mean 46.37 not 44.1.
E-21 cache_eval.json | cited artifact | n/a | runs/diac/micro_*/cache_eval.json | NO_ARTIFACT | file absent (micro_a_lm/final absent too); claim checkable from the e21 cache preds.
E-21 QCRI validator | good-but-machine | DER, lower better | models/e19/qcri.gold_vs.pred.txt, models/e19/qcri.zm_vs.pred.txt | CONFIRMED | 45.01 / 56.10 exact.
E-22 arm C improves | IMPROVES | DER, lower better | runs/diac/micro_c/gate_eval.csv | CONFIRMED | all four gates better; mean -2.68 stated vs -2.76 actual.
E-23a | MIXED / +1.45 worse / NOT-ADOPTED | DER, lower better | runs/diac/e23a_gold_v3q/gate_eval.csv + gate_probe/best_gate.json + runs/diac/stage2b2500/gate_eval.csv | INVERTED | wrong gold wn2024; E-23a is better on the mean and, at best step 2000, on all four gates.
E-23b | worse every gate, mean 48.72 | DER, lower better | runs/diac/e23b_t/gate_eval.csv | MISCITED | step-2500 mean is 50.47 not 48.72; direction (worse) confirmed.
E-23c | +4.13 worse / NOT adopted | DER, lower better | runs/diac/e23c_zm/gate_eval.csv | CONFIRMED | 52.55 reproduced; trajectory 2000 is 51.22.
E-23d | loses, +7.41 / NOT adopted | DER, lower better | runs/diac/e23d_bilstm_lr1e-3/gate_eval.csv | CONFIRMED | 55.83 reproduced; +7.49 vs artifact arm-C mean 48.34.
E-23d beats arm E | sub-claim WIN | DER, lower better | runs/diac/e23c_zm/gate_eval.csv | INVERTED | 55.83 > 52.55, so D1 LOSES to arm E.
E-23d gold yardstick | reference | DER, lower better | runs/diac/stage2b2500/gate_eval.csv | MISCITED | gold mean is 46.37 not 44.14.
experiments.yaml / plan E-23a prior | negative prior | DER, lower better | as E-23a above | INVERTED | drop the E-23a "+1.45 worse / machine data hurt gold" prior; the other E-23 priors stand.

Counts: CONFIRMED 12, MISCITED 5, INVERTED 4, NO_ARTIFACT 2, UNSUPPORTED 1 (24 rows).

## Uncheckable / limitations

- The E-19b cache-overlap percentages (CONTAMINATION.md section 2) were not reproduced; they require
  scanning models/e19/zmahood-src/word_cache.json.gz (scratch/e19/zm_contamination.py). Cited artifact exists.
- E-20 param counts and QCRI corpus statistics are prose-only; build_from_config was not run.
- E-19 runtime figures (s/line, lines/s, the 5.8 GB WDDM cap) and the E-21 QCRI sample stats
  (4.14 marks/word-base, 99.8% vowel coverage) are not reproducible from the cited metric artifacts.
- research/experiments.yaml has NO diacritizer/E-019..E-023 registry rows: per its preamble (lines 5-10)
  only E-061 was migrated and the other legacy EXPERIMENTS.md rows are historical exceptions. The
  authoritative E-19..E-23 records are therefore research/EXPERIMENTS.md plus research/e19_bakeoff/*.md;
  the yaml carries them only inside plan prose (lines 983-987, 1067-1070).
- The E-19 Fine-Tashkeel and E-19 Z-Mahmood verdicts depend on whether the slice-vs-full-ref scoring is
  considered a report defect or an intended comparison; the slice is labeled a slice, so the full-ref
  scoring is read here as a measurement bug.
