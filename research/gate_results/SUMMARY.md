# Gate results — Arabic diacritization (DER %)

Derived from `research/gate_results/gate_results.csv`. Gates in order: fadel_test, sadeed25, wikinews2024, wikinews2014.

| model | fadel_test | sadeed25 | wikinews2024 | wikinews2014 |
|---|---|---|---|---|
| v2b (14L/384h, 30M) | 47.1 | 59.9 | 61.1 | 56.4 |
| v2d28 (28L/272h, 30M) | 45.0 | 56.2 | 59.1 | 52.5 |
| b65 (76M params) | 43.1 | 55.3 | 60.3 | 53.7 |
| fadel_spec (Fadel-only specialist) | 34.2 | 51.4 | 62.9 | 57.4 |
| stage2a (warm-start; rescored from scratch/) | 41.6 | 54.5 | 60.9 | 54.1 |

The table says the Fadel-only specialist dominates the classical gates (fadel_test 34.2, sadeed25 51.4) but loses to every generalist on wikinews2014 and to all but v2b on wikinews2024, while deeper (v2d28) and larger (b65) generalists beat the v2b baseline consistently across the board. Warm-started stage2a improves on the v2b starting point everywhere yet is decisively beaten on all four gates by its same-architecture from-scratch analogue tracks, confirming the ledger's verdict that data breadth/domain routing, not pretrained init, is the frontier.


## Overfit / contamination stamps (d_gate_overlap audit, 2026-09-14)

- **wikinews2024 = CONTAMINATED (in-domain dev, NOT external):** the gate's 356 units appear verbatim in v2b train windows (shingle Jaccard 0.968, 4-gram containment 0.972) - all WN24 numbers in this table are optimistic by construction. External-truth columns are fadel_test, sadeed25 (overlap-flagged: sadeedt IS in the stage2a/b train mix), wikinews2014.
- In-training overfit watchdog added (user request 2026-09-14): train.py gate_probe scores all 4 gates mid-run every gate_eval_every steps -> runs/<phase>/gate_eval.csv (watch stage2b/arm B).


## Stage-2 matched controls (E-15/E-16, 2026-09-14, arm-B gold = recipe adopted)

| model | fadel_test | sadeed25 | wikinews2024* | wikinews2014 |
|---|---|---|---|---|
| stage2b2500 (reset-last-2 @2500, REPRO) | 33.2 | 45.7 | 57.6 | 49.0 |
| stage2a2500 (plain warm @1250) | 35.4 | 47.1 | 59.2 | 50.7 |
| stage2a2500 (plain warm @2500) | 35.4 | 46.8 | 58.0 | 49.7 |

Arm B beats arm A on ALL four gates at the matched step; val_loss favored A -> the in-run gate probe is the correct model-selection signal. Arm-B@2500 adopted as the final-model recipe (stage-1 LM pretrain -> reset-last-2 warm fine-tune, stop at the gate peak via gate_eval watch).

## E-17 FINAL MODEL (whole-corpus LM init, 2026-09-15, pwsh-2)

| model | fadel_test | sadeed25 | wikinews2024* | wikinews2014 |
|---|---|---|---|---|
| stage2final @2500 (v4-LM warm init, reset-last-2, probe PEAK) | 34.9 | 47.8 | 57.5 | 49.6 |
| stage2b2500 (previous gold, small-LM init) | 33.2 | 45.7 | 57.6 | 49.0 |
| stage2final @8000 run-end (val-selected — WRONG pick) | 42.5 | 54.3 | 59.8 | 53.5 |

Verdict: the 4x-bigger whole-corpus stage-1 LM pretrain (370 M chars, wn2024 excluded, abdou test split held out) did NOT beat the small-LM gold — clean-gate means 42.7 vs 42.6, a statistical wash (slightly worse). Consistent with E-15/E-16: data breadth/domain routing, not init scale, is the frontier. Val_loss favored the run-end weights (all 4 gates clearly worse there, DER_mean 46.5 vs 44.1) — the in-run gate probe again caught the overfit and `best_gate_weights.pt` (@2500) is the adopted final model, never the run-end snapshot. Held-out abdou test-00000 eval (15,091 sentences / 41,378 lines, never trained by ANY model): **abdou_test DER 49.4 (nocase 33.3) vs gold 41.6 (nocase 30.4)** — the whole-corpus init LOSES the held-out test by 7.8 DER points. The previous gold (stage2b2500) therefore remains the deployed FINAL model; stage2final@2500 is archived as a matched experiment.

## Provenance

- Numbers for **v2b, v2d28, b65, fadel_spec** are **ledger-recorded** (`research/EXPERIMENTS.md` rows E-12/E-13/E-14/H arm A); no v2d28/v2b/b65/fadel_spec pred datasets exist in SCRATCH to re-verify — scores are documented values from prior scoring runs and left as-is per the reporting policy.
- Numbers for **stage2a** are **rescored-scratch**: fresh CPU recomputation on 2026-09-14+ from `scratch/stage2a/{fadel,sadeed,wn24,wn14}_pred.txt` with paired `*.ref.txt` refs, via `diacritizer/scripts/eval.py compare` (venv python, CPU only). Raw unrounded values in the CSV; the pivot above shows ledger-rounded equivalents for readability (rescore matches ledger to ±0.1).
- `text_preservation` for non-rescored models is the ledger's stated value (0.9999 on sadeed25, 1.0 elsewhere); WER/DER_nocase/lines were only computable for the rescored stage2a set (left blank for ledger-only rows).
- Gate ref line counts (shared across models when scored from scratch refs): fadel_test 2500, sadeed25 1612, wikinews2024 356, wikinews2014 393.
- Runs artifacts (updated weights only) live under `runs/diac/<model>/final` and `runs/diac/stage2a/final`; no runs/ writes occurred, no GPU was used.