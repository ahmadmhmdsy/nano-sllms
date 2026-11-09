# MU0_REPORT — micro-expert sandbox, stage 0 (E-24)

Date: 2026-09-18 · Plan: docs/plans/2026-09-18-micro-expert-composition-plan.md · Code: mex/ (commits 241117f..e2c4b3d)
Pre-registration: EXPERIMENTS.md E-24 row, registered BEFORE training (commit cadd5f2).

## Setup (all measured, not assumed)
- Shared 97-id char vocab, tokenizer.local data/mex/tokenizer; uint32 packed shards (ctx 96).
- Experts: 2L/hidden80/ffn320/4H-GQA2kv, tied — EXACTLY 200,160 params (test-pinned to build_model sum).
- Control: 2L/hidden160/ffn640 — 784,320 params (0.98x 4-expert, within 5%); tokens = byte-exact union of expert train+val (3,126,113 ids).
- max_steps 2000, batch 8x4 accum, lr 1e-3 cosine, same optimizer everywhere. Single GPU, sequential, auto-resume.

## Measured trivial baselines (from train data, honest)
x1 echo-bare 0.0 | x2 train-answer mode 0.002 | x3 train-majority 0.988 | x4 identity-under-mix 0.316.

## Results (500-2000 held-out prompts; runs/mex/*/final/mex_eval.json)

| arm | x1 | trivial | x2 | trivial | x3 | trivial | x4 | trivial |
|---|---|---|---|---|---|---|---|---|
| expert | 0.0115 | 0.000 | 0.004 | 0.002 | 0.988 | 0.988 | 0.236 | 0.316 |
| control | 0.078 | 0.000 | 0.002 | 0.002 | 0.988 | 0.988 | 0.072 | 0.316 |

## Gate verdicts (pre-registered gate: expert strictly beats trivial)
- X1 diacritics: PASS (0.0115 > 0.0) — but tiny; generalization beyond memorized bare-forms is thin.
- X2 arithmetic: TECHNICAL PASS (0.004 > 0.002) — practically useless; exact 3-digit add/sub is out of reach at this budget.
- X3 structure: FAIL THE GATE (0.988 == 0.988 — ties majority; learned the label prior only).
- X4 string-ops: FAIL (0.236 < 0.316 — BELOW trivial; the model never learned ident-switch behavior).
- Control: at 4x params, only x1 improved (0.078, still 7x its expert) — aggregate control is NOT negligible.

## Interpretation
1. Undersampled: arith/structure/strops each saw 2000 steps over 30K-line pools; 2-layer 200K models need more steps or easier targets (the DESIGN's own feasibility gate is doing its honest job).
2. Interesting asymmetry: control beats experts on x1 (multi-task benefit on the only real-data task); expert- head-speciality held only for x4-style 'deviation' tasks.
3. Composition arms (soup/TIES, MoE-merge, dispatch-router, committee) are NOT worth running on experts that tie or fail their own gates — that was the design intent of E-24.

## Honest μ0 close
Stage-0 verdict: DATA/STEPS undersized; architecture + harness + param accounting all sound (params exact, control exact, splits deterministic, eval honest).
Recommendation before mu1: raise max_steps (e.g. 10K-20K window) and/or retire x3 majority-collapse (label prior dominated; needs harder brackets), x2 may need 1-2 digit scaffold targets. User decision: window extension + X3 hardening was approved and run (see mu0b below).


# mu0b — E-25 extended window + hardened X3 (user option B, 2026-09-18)

E-25 was registered in EXPERIMENTS.md in the same working session but its ledger row landed in the results commit (f671ae0), not a separate pre-commit — machine-verifiable ordering could not be established after the fact, so treat this registration as retro-documented (unlike E-24, which has a clean pre-results commit cadd5f2). Re-registration discipline for mu1: land the EXPERIMENTS row FIRST, results later. Changes: max_steps 2000->12000 (all five); X3 regenerated with maxlen 24, ok-lines balanced by construction, bad = half subtle one-pair flips / half legacy flips (labels ~50/50; re-measured test trivial 0.512); X1/X2/X4 data unchanged; control re-packed as exact union of expert train+val (verified: control 3,294,269 + 31,506 = sum experts, byte-exact ME-D5). Old 2K-step finals archived at runs/mex/archive_2000/. Sanity gates 5/5 PASS (c2510b0).

## Results (12K steps, held-out)

| arm | x1 | trivial | x2 | trivial | x3 | trivial | x4 | trivial |
|---|---|---|---|---|---|---|---|---|
| expert | 0.1075 | 0.000 | 0.898 | 0.002 | 0.918 | 0.512 | 0.846 | 0.316 |
| control | 0.129 | 0.000 | 0.054 | 0.002 | 0.922 | 0.512 | 0.864 | 0.316 |

## Verdicts (pre-registered gate: strictly beat trivial)
- X1 PASS (0.1075 > 0) | X2 PASS 0.898 (89.8% exact 3-digit arithmetic) | X3 PASS 0.918 > 0.512 | X4 PASS 0.846.
- All four pre-registered gates PASS. The E-24 failures were, as suspected, data/steps — not architecture.
- Specialist-vs-control asymmetry (mu1-relevant): experts crush control on symbolic ops (x2: 0.898 vs 0.054; x3: 0.918 vs 0.922 tie; x4: 0.846 vs 0.864 tie) while control stays ahead on real-data x1 (0.129 vs 0.1075). Composition should exploit both.
- mu1 composition arms (soup/TIES, MoE-merge, dispatch-router, committee distill) are now GREEN to run.

## Evidence paths
runs/mex/{x1,x2,x3,x4,control}/final/mex_eval.json (12K); archive_2000/ holds the 2K artifacts; .superpowers/sdd/task-6b-report.md pending; tests 14/14 at commit 7d79cc1.


# mu0c (E-26): 120K-step window — does 10x more training help? NO.

## Setup
Same five arms, same recipe (batch 8 / accum 4 / lr 1e-3 cosine / fp16), max_steps 120000 (10x E-25). Runs fresh; the 12K finals archived at runs/mex/archive_12000/. Configs bumped + re-gated 5/5 PASS (5d7d570); watcher + plotter added: mex/scripts/watch_losses.py (CPU-only during training), mex/scripts/plot_losses.py (curves + overfit-onset summary json). Pre-registered before training (03f9917). Two concurrent trainings throughout (user-authorized; peak 576 MiB, GPU 84-88%).

## Loss-curve findings (overfit / underfit)
- min eval loss arrives EARLY for every arm (x1 ~8K, x2 ~7.5K, x3 ~6K, x4 ~6.5K, control ~88K — control keeps improving longest).
- Mid-run overfit is massive at fixed high lr: x2 eval climbs 1.19 -> 2.11 by step 118K while train falls to 0.86; x3 eval 2.67 peak. The eval curve is high-lr memorization, not capacity failure.
- The cosine anneal tail pulls eval back DOWN to ~min-value by 120K (x2 2.11 -> 1.188 at step 120K; x3 2.67 -> 1.159). So mid-run divergence must be read from the DATASET-FIXED val bridge, not just the final snapshot.
- Control is the healthiest curve: eval 1.227 -> 1.060 monotone-ish, only 0.074 train/val gap at 120K (train 0.986).

## Results (120K, held-out) vs E-25 (12K)

| arm | 12K exact | 120K exact | delta | 12K eval loss | 120K eval loss |
|---|---|---|---|---|---|
| x1 | 0.1075 | 0.105 | -0.0025 | 1.1196 | 1.1293 |
| x2 | 0.898 | 0.828 | -0.070 | 1.1910 | 1.1878 |
| x3 | 0.918 | 0.916 | -0.002 | 1.1580 | 1.1590 |
| x4 | 0.846 | 0.862 | +0.016 | 1.5411 | 1.5510 |
| control (cross-task) | x1 0.129 / x2 0.054 / x3 0.922 / x4 0.864 | x1 0.1085 / x2 0.848 / x3 0.918 / x4 0.802 | — | 1.1652 | 1.0601 |

## Gates (identical to E-25): 4/4 PASS
x1 0.105 > 0.0 | x2 0.828 > 0.002 | x3 0.916 > 0.512 | x4 0.862 > 0.316. Gates pass — but note they are running at the SAME performance plateau found in E-25.

## Verdict and mu1 implication
The mu0 window is STEP-SATURATED at 12K: 10x compute bought nothing on held-out loss and slightly hurt x2 (more memorization of the same pool). Scaling this recipe past 12K steps is dead; the next gains must come from DATA (bigger pools, more diversity) or REGULARIZATION (bigger val sets, dropout, data mixing), not steps. This is the strongest argument to move to mu1 composition now rather than grow mu0 further. Also the control model at 120K now matches or beats every expert on cross-task eval (x2: 0.848 vs 0.828; x3: 0.918 vs 0.916; x1: 0.1085 vs 0.105) while x4 specialist keeps its edge (0.862 vs 0.802) — dispatch should send x4 to its specialist, share the rest.

## Evidence
runs/mex/loss_track.csv (14400 rows), runs/mex/loss_curves.png + .json (per-arm curve plots, min-eval step, overfit onset info), runs/mex/{x1-x4,control}/final/mex_eval.json + logs (tfevents are stage-truth), archive_12000/ for 12K comparisons. Training jobs x1+x2 then x3+x4 then control: 3 phases, ~2.5h each pair, ~1h solo control; total ~7h wall.
