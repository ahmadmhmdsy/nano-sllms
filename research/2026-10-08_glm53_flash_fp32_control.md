# E-114 - Is the RL endpoint capped by the fp16/torch-2.14 step? (fp32-control arm, TASKS row 159)

Follow-up to E-113 (research/2026-10-08_glm53_flash_regime_match.md). E-113 refuted the
headroom explanation and left exactly one suspect standing: the RL step itself under
fp16/torch-2.14 - a lower per-group conversion rate from reward into greedy correctness,
with fp16's activation range in the last block suggested as the mechanism. This experiment
isolates the dtype/optimizer path: same init, same 96-group schedule, same seed, only
--amp-dtype fp16 -> fp32.

## Question

Does the RL magnitude gap close when the optimizer/dtype path changes from fp16 to fp32,
holding the init checkpoint and the task schedule fixed?

## Hypothesis

Pre-registered in TASKS 159 BEFORE the first GPU launch (verbatim block at the end of
TASKS.md):

- **P1 (primary)** - arm A4 greedy confirm gain >= +12 tasks (published +16; A2 +7).
- **P2** - A4 pass@8 gain over the 3 trained families >= +8 (published +10; A2 +5).
- **P3** - A4 greedy endpoint >= 13/24, strictly above A2's 12/24.
- **P4 (manipulation)** - A4's init-time exact-rollout count is reported next to A2's
  406/1536; exact equality is NOT required, because fp32 changes the sampled tokens.

Falsification: an A4 greedy gain <= +8 refutes the fp16-conversion hypothesis and moves the
suspect to the factors A2 and A4 share. The prediction was that P1-P4 all pass.

## Rig

- **Data / model** - unchanged from E-113: byte tokenizer (vocab 260), context 192, 12
  layers, dim 192, 6 heads, 8 experts top-2 + 1 shared, 4 residual streams; 25,730,592 total
  / 9,805,344 active. RL trains the last block head only: 2,190,152 trainable.
- **Arm A4** - RL from runs/pretrain/checkpoint-0100 (sha256
  44d740f77fa0398bd2dc4b3af8d60ec7182e050c84643920550655e8be2f5984, the A2/E-112 init), 96
  groups x 16 = 1,536 rollouts, families increment,double,even, 32 tasks/family, temperature
  0.35, lr 5e-5, last-block-head, binary reward, seed 31415, checkpoints 24/48/72/96,
  --amp-dtype fp32, allocator cap 4.10 GiB.
- **Paired control A2** - runs/rl (E-112): identical in every field except --amp-dtype fp16.
- **Manipulation checks, enforced in code before the run** - the init file hash must equal the
  pre-registered constant, and the three baseline files must exist and reference
  checkpoint-0100; the driver aborts otherwise. Logged after the run: amp_dtype and
  schedule_sha256. This is what makes A4-vs-A2 a paired comparison.
- **Machine** - Quadro RTX 4000 8 GiB (sm_75), torch 2.14.0+cu126, Python 3.12.9, Windows;
  7,727 MiB free at launch (idle GPU), cap 4.10 GiB = 54%. Published reference: RTX 3080 Ti
  12 GiB bf16, torch 2.5.1+cu121, Linux.

## Results table

Every number cites its file. Baselines are E-112's own checkpoint-0100 evals, reused rather
than re-measured so A2 and A4 are paired on the identical before-file.

| stage | params | wall-clock | in-run evidence (file) | independent eval (file) | verdict |
|---|---|---|---|---|---|
| A4 RL (fp32) | 2,190,152 trainable | **464.5 s** (A2 fp16: 440.9 s; published: 155.5 s) | 1,536 rollouts, **407** exact (A2: 406; published: 417), **78** updated groups (A2: 77; published: 73), peak **1.495 GiB** (A2: 0.939; published: 0.936), no spill; runs/rl-regime-a4/training-receipt.json | - | - |
| A4 greedy confirm | same | 82.4 s | - | **5/24 -> 12/24** (+7, 7 gains / 0 losses, exact McNemar **p=0.0156**, bootstrap **+12.5 to +50.0 pp**); runs/eval/comparison-a4-greedy.json | **FAIL** (P1: +7 < +12) |
| A4 pass@8, 3 trained families | same | 103.9 s | - | **10/24 -> 15/24** solved@8 (+5, 6 gains / 1 loss, p=0.125), exact **35/192 -> 72/192**; runs/eval/comparison-a4-pass8.json | **FAIL** (P2: +5 < +8) |
| A4 pass@8, all 8 families | same | 274.3 s | - | **30/64 -> 30/64** (0, 6 gains / 6 losses, p=1.0), exact **110/512 -> 122/512**; runs/eval/comparison-a4-pass8-all.json | net **0** |
| A4 endpoint | same | - | - | greedy **12/24** (A2: 12/24; published: 16/24) | **FAIL** (P3: 12 < 13) |
| P4 manipulation | same | - | init exact **407** vs A2's 406 of 1,536; schedule_sha256 identical | - | **PASS** |

Arm comparison. The dtype path is the only variable:

| arm | init | amp dtype | init exact (of 1,536) | updated groups | greedy gain | pass@8 gain (3 families) | pass@8 gain (all 8) |
|---|---|---|---|---|---|---|---|
| A2 (E-112) | pretrain step 100 | fp16 | 406 | 77 | +7 (5 -> 12/24) | +5 (10 -> 15/24) | 0 (30 -> 30/64) |
| **A4 (E-114)** | pretrain step 100 | **fp32** | **407** | **78** | **+7 (5 -> 12/24)** | **+5 (10 -> 15/24)** | **0 (30 -> 30/64)** |
| A1 (E-113) | dense step 90 | fp16 | 150 | 57 | +8 (0 -> 8/24) | +11 (2 -> 13/24) | +9 (9 -> 18/64) |
| published | bf16 step 100 | bf16 | 417 | 73 | +16 (0 -> 16/24) | +10 (9 -> 19/24) | +9 (25 -> 34/64) |

Per-family A4 greedy, solved before -> after (8 tasks each): double 5 -> 7, increment 0 -> 5,
even 0 -> 0. A2 gained its entire +7 in double (0 -> 8) and nothing in increment - the same
total, allocated differently. Per-family A4 pass@8, all 8 families, solved before -> after:
increment 2 -> 7, double 8 -> 8, square 8 -> 6, reverse 3 -> 0, list_sum 8 -> 8, even 1 -> 1,
absolute 0 -> 0, nonnegative 0 -> 0.

## Gates and verdicts

- **P1 - FAIL.** ACTUAL greedy gain **+7** (5/24 -> 12/24); REQUIRED >= +12. This is A2's
  number exactly, from the same init under the same schedule.
- **P2 - FAIL.** ACTUAL pass@8 gain over the 3 trained families **+5** (10/24 -> 15/24);
  REQUIRED >= +8. Again A2's number exactly.
- **P3 - FAIL.** ACTUAL endpoint **12/24**; REQUIRED >= 13/24. The endpoint did not move at
  all - it is A2's endpoint, not the published 16/24.
- **P4 - PASS (manipulation).** The init produced **407/1,536** exact rollouts against A2's
  **406/1,536** (0.25% apart), the receipt reads amp_dtype fp32, and schedule_sha256 is
  0c2327ee7cd08db42a32ea78e69e590d80592d5068d7d6deca216c8b948fbc2e - the same hash A1, A2
  and A3 all carry. fp32 changed neither the reward signal nor the outcome.
- **Falsification clause - TRIGGERED.** The pre-registration said an A4 gain <= +8 refutes the
  fp16-conversion hypothesis. ACTUAL +7.

**Headline verdict: the fp16-conversion hypothesis is REFUTED.** Changing the dtype/optimizer
path from fp16 to fp32 moved no endpoint metric: greedy +7, pass@8 +5 and all-8 +0 are
identical to A2's fp16 arm, while the init's reward signal and the number of reward-bearing
updates were reproduced to one rollout (407 vs 406) and one group (78 vs 77).

The refutation also kills the specific mechanism E-113 proposed. E-113 suggested that fp16's
activation range in the last block restricts what a head-only update can express. fp32 has
fp32's exponent range - strictly wider than fp16's - and changes nothing, so the constraint is
not representational. The mechanism is dead, not merely unconfirmed.

## What the refutation leaves standing

The factors A2 and A4 share, in order of testability on this machine:

1. **torch 2.14.0+cu126 vs the published 2.5.1+cu121** - the last remaining stack difference
   (optimizer, autograd and RNG paths). Separating it needs a second environment, so it is
   user-gated.
2. **The device.** The publication ran bf16 on an RTX 3080 Ti; bf16 does not exist on sm_75,
   so this cannot be separated locally. A4's fp32 result is the closest available control and
   it is null, which makes a pure-precision explanation unlikely.
3. **The published number itself.** 73 reward-bearing updates and 417/1,536 exact rollouts are
   matched, so the run is the same size; but no published per-group trace exists to compare
   update-by-update conversion, so the claim that the publication's updates are *better* is
   untested rather than refuted.
4. **A real plateau.** Every arm we have run lands at +7 or +8 greedy and 12-15/24 at the
   endpoint, never 16/24. Whether that is a ceiling of this recipe at this scale is now the
   simplest surviving explanation.

## Levers to carry

- **A dtype change is a one-field A/B, not a re-architecture.** fp16 -> fp32 cost 5%
  wall-clock (464.5 s vs 440.9 s) and 0.56 GiB more peak (1.495 vs 0.939 GiB) at 25.7M
  params, so 'is it the numerics?' is cheap to answer and worth answering before blaming the
  stack.
- **Match the manipulation checks, not just the recipe.** The init hash, the schedule hash,
  the exact-rollout count and the updated-group count together proved the two arms differed
  only in amp_dtype - that is what turns 'two similar runs' into a paired experiment.
- **Report the per-family allocation, not only the total.** A2 and A4 both gained +7 greedy
  but allocated it differently (A2: all in double; A4: 5 in increment + 2 in double). A single
  total would have hidden a real behavioural difference between two otherwise identical arms.
- **Kill the mechanism, not just the variable.** Testing fp32 tests the *range* mechanism as
  well as the dtype variable, because fp32 has the wider exponent range that fp16 lacks.

## Mistakes and gotchas

- **The receipt has no top-level exact/updated totals.** They live only inside groups_detail,
  so the driver's own log line printed exact=None updated=None; summing groups_detail is the
  fix. Promoting those two totals into the receipt would remove a foot-gun.
- **The before-file is reused, not re-verified.** A4 pairs against E-112's checkpoint-0100
  evals by design; the in-code check asserts only that the file references checkpoint-0100, so
  a corrupted baseline with the right path in it would pass. Re-measuring costs ~80 s and would
  close that hole.
- **fp32 was only 5% slower than fp16 here** (464.5 s vs 440.9 s). Generation dominates at
  25.7M params, so the usual 'fp32 costs 2-4x' intuition does not transfer to this scale.

## Scale-transfer caveats

- Licensed at **25.7M parameters, context 192, one init, one dtype change, one schedule**.
  Nothing here speaks to larger models or longer RL.
- **fp32 here means no autocast and no GradScaler on a head-only update.** It does not test
  fp32 master weights with fp16 compute, a mixed-precision variant that was not run.
- The all-8 pass@8 view is net 0 with p=1.0 and a bootstrap interval spanning zero: a null,
  not evidence that no effect exists.
- The confirmation set is 24 tasks for greedy and 192 rollouts for pass@8 over the trained
  families; the all-8 view is 64 tasks / 512 rollouts.
- The nested clone remains a local gitlink; its runs/ are gitignored, so these receipts live
  only on this machine.

## Evidence files

Nested repo: runs/rl-regime-a4/training-receipt.json,
runs/rl-regime-a4/checkpoint-{0024,0048,0072,0096},
runs/eval/confirm-greedy-rl-a4.json, runs/eval/confirm-pass8-rl-a4.json,
runs/eval/confirm-pass8-all-rl-a4.json, runs/eval/comparison-a4-{greedy,pass8,pass8-all}.json,
runs/regime/regime.log, runs/regime/children.log. Paired control:
runs/rl/training-receipt.json and runs/eval/confirm-*-pretrain-0100.json (E-112). Driver:
scripts/regime_match.py --stage a4 (new). Predecessors: E-113
research/2026-10-08_glm53_flash_regime_match.md, E-112
research/2026-10-08_glm53_flash_clone_validation.md.
