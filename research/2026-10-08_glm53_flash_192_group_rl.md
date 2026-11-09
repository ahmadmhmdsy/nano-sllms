# E-116 - A6: the 192-group RL arm - the published greedy endpoint was a step-budget gap, not a recipe ceiling

- Date: 2026-10-08
- Task: TASKS row 161 (pre-registered) - driver `scripts/regime_match.py --stage a6`
- Init: `runs/pretrain/checkpoint-0100`, sha256 `44d740f77fa0398bd2dc4b3af8d60ec7182e050c84643920550655e8be2f5984`
  (gate PASS against the pre-registration before launch)
- Nested repo commit: `624f4ca` (local-only; never pushed)

## Question

A4 (E-114) trained 96 groups and reached greedy 12/24, short of the published 16/24.
Is that shortfall a property of the recipe (a ceiling), or simply too small a group budget?

## Hypothesis (pre-registered in TASKS 161 before the first GPU launch)

Extending the identical recipe from 96 to 192 groups raises greedy above 12/24 and reaches the
published 16/24 endpoint, and the first 96 groups must be bit-identical to A4's (pure continuation).

## Rig

```
scripts/train_rl.py --output runs/rl-regime-a6 --initial-checkpoint runs/pretrain/checkpoint-0100
  --groups 192 --checkpoints 96,144,192 --group-size 16 --tasks-per-family 32
  --families increment,double,even --temperature 0.35 --learning-rate 5e-5
  --train-scope last-block-head --reward-mode binary --seed 31415 --amp-dtype fp32 --vram-cap-gib 4.1
```

Unchanged from A4 except `--groups 96 -> 192` and `--checkpoints`. Step 96 was NOT re-measured:
G1 proved the file bit-identical to A4's endpoint, so the E-114 receipts are reused verbatim.

- Cost: RL wall clock 907.7 s (receipt `elapsed_seconds` 902.064, `train_seconds_total` 900.69),
  192/192 groups, 0 resume events, 0 groups updated outside the cap.
- VRAM: peak allocated 1.495 GiB / reserved 2.986 GiB on an 8 GiB Quadro RTX 4000,
  `spilled_to_shared_memory: false`. Trainable parameters 2,190,152 (last-block head only).
- Evals: greedy 144 85.7 s, pass@8 144 113.9 s, greedy 192 84.0 s, pass@8 192 104.0 s.
- Total E-116 GPU spend ~21.6 min (RL + the four pre-registered evals); the all-8 probe is extra.
- `runs/rl-regime-a6/checkpoint-0192/model.pt` = 103,045,899 B.

## Results

| groups | greedy solved/24 | gains/losses | McNemar p | pass@8 (3 fam) solved/24 | g/l | p |
|---|---|---|---|---|---|---|
| 96 (A4, reused) | 12/24 | 7/0 | 0.015625 | 15/24 | 6/1 | 0.125 |
| 144 | 13/24 | 8/0 | 0.0078125 | 14/24 | 5/1 | 0.21875 |
| 192 | **16/24** | 11/0 | 0.0009765625 | **17/24** | 7/0 | 0.015625 |

Per-family at 192 (before -> after, baseline = pretrain/checkpoint-0100):

- greedy: increment 0 -> 8, double 5 -> 8, even 0 -> 0
- pass@8: increment 2 -> 8, double 8 -> 8, even 0 -> 1

Post-hoc, NOT pre-registered - all-8 pass@8 at 192:
30/64 -> 30/64, 8 gains / 8 losses, McNemar p = 1.0, bootstrap 95% [-3.1, +16.4] pp.
Per family: increment 2 -> 8, even 1 -> 2, nonnegative 0 -> 1 | square 8 -> 3, reverse 3 -> 0 |
double 8 -> 8, list_sum 8 -> 8, absolute 0 -> 0.

## Gates and verdicts

- **G1 equivalence - PASS.** `sha256(runs/rl-regime-a6/checkpoint-0096/model.pt) ==
  sha256(runs/rl-regime-a4/checkpoint-0096/model.pt)` (log line `A6 EQUIVALENCE a6-0096 == a4-0096:
  True`). The a6 run is a pure continuation of the A4 trajectory; the only difference is 96 more
  groups. This is what licenses reusing the E-114 step-96 receipts instead of re-measuring.
- **G2 still-rising - PASS.** Greedy 12 -> 13 -> 16; every window non-decreasing, last window +3.
- **G3 published-endpoint parity - PASS.** 16/24 equals the published greedy endpoint exactly.
- **Falsifier did not fire** (a6-0096 != a4-0096 would have meant the schedule diverged before the
  fork; it did not).
- **pass@8 3-family (pre-registered at 144/192) - PASS.** 15 -> 14 -> 17/24; the 192 gain is
  significant (7/0, p = 0.0156).
- **Post-hoc all-8 - FAIL to improve.** Flat at 30/64 (published 34/64).

## What this closes

1. **The A4 shortfall was a step-budget gap, not a recipe ceiling.** The same init, recipe and seed
   reach the published greedy endpoint exactly once the budget is doubled.
2. **E-115's greedy reading was right; E-115's pass@8 reading is refuted.** The E-115 report
   recorded that 3-family pass@8 "saturated at 72 groups" (12, 12, 15, 15). At 192 it resumed:
   15 -> 14 -> 17/24. The saturation label was an artefact of the short budget.
3. **The residual, stated honestly:** the published run reached 16/24 in 96 groups (1,536 rollouts);
   we needed 192 (3,072). That is a ~2x per-group learning-*efficiency* gap, not a capability gap.
   Nothing in E-116 explains it; it is the next thing worth studying.
4. **The gain is not free.** All-8 is flat because the untrained families regress by as much as the
   trained ones gain (square 8 -> 3, reverse 3 -> 0). Greedy and 3-family metrics hide this; only
   the all-8 probe exposes it.

## Levers to carry

- **A longer RL budget is a real lever** on this init: 2x the groups bought +4 greedy of 24
  (12 -> 16), but the marginal return per doubling is modest and it costs untrained-family accuracy.
- **Keep the G1-style equivalence check.** Hashing the shared checkpoint proves "same trajectory,
  more steps" and lets the earlier receipts be reused instead of re-measured - it saved two evals.
- **`schedule_sha256` is budget-scoped, not trajectory-scoped.** A4 `0c2327ee...` vs A6
  `8da15758...` differ although the first 96 groups are provably identical (the hash covers all
  192 groups). Compare checkpoint hashes, never schedule hashes, across budgets.

## Mistakes and gotchas

- E-115 pre-registered a monotonicity boolean for a curve that can dip by a single task. E-116's
  gate was phrased as "still rising over the final window", which is what should have been
  pre-registered the first time (recorded in MEMORY.md).
- An all-8 metric on a 3-family arm is post-hoc by construction; it is labelled POST-HOC here and
  is not part of the TASKS 161 pre-registration.

## Scale-transfer caveats

Single init, single seed (31415), one GPU, fp32 (sm_75 has no bf16). The parity is a *metric* match
on greedy only: 3-family pass@8 (17 vs 19) and all-8 (30 vs 34) still trail the published run, which
used bf16 on an RTX 3080 Ti with torch 2.5.1. Treat "matches the published endpoint" as
"matches the published greedy endpoint at 2x the rollout budget".

## Evidence files

- `runs/rl-regime-a6/training-receipt.json`, `progress.jsonl`, `schedule.json`
- `runs/rl-regime-a6/checkpoint-0096|0144|0192/` (0096 bit-identical to `rl-regime-a4/checkpoint-0096`)
- `runs/eval/confirm-greedy-rl-a6-ckpt0144.json`, `confirm-greedy-rl-a6-ckpt0192.json`
- `runs/eval/confirm-pass8-rl-a6-ckpt0144.json`, `confirm-pass8-rl-a6-ckpt0192.json`
- `runs/eval/comparison-a6-ckpt0144-greedy.json`, `-ckpt0144-pass8.json`,
  `-ckpt0192-greedy.json`, `-ckpt0192-pass8.json`, `comparison-a6-ckpt0192-pass8-all.json`
- `runs/regime/a6-curve.json`; `runs/regime/regime.log` (A6 RECEIPT / A6 EQUIVALENCE lines)
