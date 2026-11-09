# Plan: E-63 trial-bed arm ladder (user request, 2026-09-22)

User request (verbatim intent): use ~5-10% of the diac train data as a trial
bed; test shared-KV, a new shared-layer architecture, an optimizer switch and
a loss switch on it — each feature ALONE, then combinations of the winners —
with multi-metric eval on unseen data; then train the best combination on the
FULL dataset. Smoke-train end-to-end before every run.

## Trial bed (fixed once, never retuned per arm)

- Train slice: FIRST 243,343 v3 train windows = 10.0% (data/diac/v3s10/tokens,
  built by scratch/e63_build_slice.py as plain copies - absolute val refs stay
  in data/diac/v3 untouched).
- Val for in-training arm eval: FIRST 9,613 v3 val windows (unseen; a strict
  subset of the reinforcements-free val pool, never trained on by any arm).
- Matched budget: every arm = batch 32, accum 1, 2000 steps, eval_every 200,
  ctx 128, fp16 autocast, identical minibatch stream (same slice length =>
  the fixed rng(777) stream draws IDENTICAL batches across arms - the fair
  matched-comparison property the arm-C standard used).
- Full arch except where the feature dictates: 272/28L/8h/2kv/1088, warm
  stage1lm init (except where noted).

## Arms

Singles (one factor each vs S0 control):

| arm | feature | change |
|---|---|---|
| S0 | control | plain 28L arch, AdamW, CE - the baseline |
| S1 | shared-KV | kv_pattern cycle3 (the E-62 feature) |
| S2 | shared LAYERS | model.shared_layers 3: only 3 distinct blocks cycled over the 28 layer positions (~11.5M params vs 30.1M). NO warm init init path change; stage1lm loader (strict=False) fills the 3 modules, resets disallowed |
| S3 | optimizer | Muon (Newton-Schulz orthogonalized momentum on 2D hidden weights) + AdamW(2e-4) on embeddings/head/1D; muon_lr 2.5e-2 |
| S4 | loss | label smoothing 0.1 |
| S5 | loss | logit-adjusted CE (E-60a attempt-3 lever; tau 1.0, per-class prior from the train slice) |

Combination stage (only winners above placebo margins; each a NEW arm,
pre-registered here before any of them launch - the singles decide which of
C1/C2/C3 get run):

- C1 = S1 (SKV) + best-of {S2, S3.arch-agnostic} if it wins
- C2 = S1 + best loss variant
- C3 = kitchen sink of every winning ingredient (SKV + top other + top loss +
  winner optimizer if distinct)

Selection bands (pre-registered): an arm is a WINNER if val_loss @2000
improves >= 2% relative vs S0 AND val_acc does not fall below S0 - 0.005
(bands, not one number). Ties -> prefer the simpler arm. Placebos (within
noise of S0) are dropped without combination.

## Multi-metric + unseen-data eval

- In-training: val_loss + val_acc on the 9,613-row unseen val slice.
- At arm closure (finalists): FULL val pool (160,214 windows) loss/acc via
  scratch/e63_final_eval.py (read-only, no training; also DER-lite via the
  existing eval pipeline on a slimmed gate subset only for the arms entering
  the combination decision).
- The FINAL full-data run gets the standard full gate_probe (4 gates, DER/WER
  + text_preservation CSV) at its registered gate steps.

## Smoke-before-every-run (hard rule)

Every config below has a *_smoke.yaml twin: total_steps 60, eval_every 60,
save_every 60 - launched first and only if it completes end-to-end does the
real run launch. Smoke run dirs get cleaned after the PASS printer.

## Discipline

- One GPU arm at a time (single-GPU exclusivity); managed pwsh background
  jobs only (2026-09-22 lesson).
- EXPERIMENTS.md E-63 row opened NOW (this file is the registration); per-arm
  rows appended as they close, honest FAILs recorded as FAILs.
- The stage2b2500 gold dir stays untouched.
- Full-dataset final run config: phase stage3best_; launcher ONLY after the
  combination decision is recorded.
