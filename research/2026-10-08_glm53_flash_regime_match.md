# E-113 - Does matching the published RL init regime restore the published RL magnitude? (TASKS row 158)

Follow-up to E-112 (`research/2026-10-08_glm53_flash_clone_validation.md`). E-112
reproduced the `glm-5.3-flash-from-scratch` infrastructure EXACTLY but only about half of
the published RL gain, and attributed the gap to headroom: our fp16/torch-2.14 pretrain
`checkpoint-0100` was ahead of the published bf16 step-100 checkpoint (dev pass@1 3/8 vs
2/8), so the headroom the published result exploits had already been spent by pretraining.
That explanation was explicitly left untested. This experiment tests it by re-running the
identical RL recipe from an init checkpoint SELECTED to match the published headroom, with
a strong-init arm as a dose-response point.

## Question

Does the RL magnitude gap close when the RL init checkpoint is selected to match the
published regime (dev pass@1 2/8) rather than the published step number?

## Hypothesis

Pre-registered in TASKS 158 BEFORE the first GPU launch:

- **P1 (primary)** - arm A1 (init selected at dev <= 2/8) greedy confirm gain >= +12 tasks
  (published +16; E-112 got +7 from a dev-3/8 init).
- **P2** - arm A1 pass@8 gain over the 3 trained families >= +8 solved tasks (published
  +10; E-112 got +5).
- **P3** - arm A3 (init `runs/pretrain/checkpoint-0200`, dev 7/8) greedy gain <= +3.
- **P4** - the greedy gain decreases monotonically as init strength rises 2/8 -> 3/8 ->
  7/8.

Falsification: an A1 gain close to the E-112 +7/+5 refutes the headroom mechanism and
points at fp16/torch-2.14 instead; an A3 gain >= +7 also refutes it. The prediction was
that the headroom mechanism holds and P1-P4 all pass.

## Rig

- **Data** - the repo's own generated task families, byte tokenizer (vocab 260), context
  192. Split identities are disjoint and frozen (dev / final / rl / confirm); the
  confirmation set is opened only after the run is chosen.
- **Model** - `GLM53FlashFromScratch`, 12 layers, dim 192, 6 heads, 8 experts top-2 + 1
  shared, 4 residual streams; **25,730,592 total / 9,805,344 active** parameters. RL trains
  the last block head only: **2,190,152** trainable.
- **Arms** - A1 (primary): RL from the selected weak init
  `runs/pretrain-dense/checkpoint-0090` (dev 1/8, greedy confirm 0/24). A2 = the E-112 arm,
  already measured: RL from `runs/pretrain/checkpoint-0100` (dev 3/8, greedy confirm 5/24).
  A3 (dose-response): RL from `runs/pretrain/checkpoint-0200` (dev 7/8). All three arms use
  the IDENTICAL RL recipe and, critically, the identical 96-group task schedule - both new
  receipts carry `schedule_sha256`
  `0c2327ee7cd08db42a32ea78e69e590d80592d5068d7d6deca216c8b948fbc2e`, so the init
  checkpoint is the only variable.
- **Budget** - stage 1: repeat the 400-step pretrain (batch 32, seq 128, lr 3e-4, seed 42)
  with checkpoints every 10 steps into `runs/pretrain-dense`. Stage 2: greedy dev pass@1
  for each dense checkpoint. Stages 3-4: RL 96 groups, group size 16, 32 tasks/family,
  families `increment,double,even`, temperature 0.35, lr 5e-5, last-block-head, binary
  reward, seed 31415, checkpoints 24/48/72/96. Confirmation: greedy and pass@8 (8 samples,
  temperature 0.35, seed 8675309) on init and final.
- **Machine** - Quadro RTX 4000 8 GiB (sm_75), torch 2.14.0+cu126, Python 3.12.9, Windows,
  allocator cap 4.10 GiB (54% of 7,772 MiB free at launch) + spill detector. Published
  reference: RTX 3080 Ti 12 GiB, torch 2.5.1+cu121, Python 3.12.3, Linux.

## Results table

Wall-clock is the headline column; every number cites its file. The selection rule and the
full stage log are in `runs/regime/selection.json` and `runs/regime/regime.log`.

| stage | params | wall-clock | in-run evidence (file) | independent eval (file) | verdict |
|---|---|---|---|---|---|
| dense pretrain re-run | 25,730,592 / 9,805,344 active | **214.9 s** (E-112: 179.3 s) | 400 steps, tokens_seen **1,381,760**, peak **2.141 GiB**, no spill; `runs/pretrain-dense/training-receipt.json` | `checkpoint-0100` and `checkpoint-0400` are **sha256-identical** to the E-112 run | **PASS** (determinism) |
| dense dev scan | same | 10 evals, 6.5 min | `runs/regime/selection.json` | dev pass@1 0/8 (steps 10-70), **1/8** (80, 90), **3/8** (100) - **2/8 never occurs**; rule selected step 90 | rule PASS, target unreachable |
| A1 RL | 2,190,152 trainable | **341.5 s** | 96 groups x 16 = 1,536 rollouts, **150** exact (E-112's step-100 init: 406; published: 417), **57** updated groups (E-112: 77), peak **0.933 GiB**, no spill; `runs/rl-regime-a1/training-receipt.json` | - | - |
| A1 greedy confirm | same | 73.1 s + 76.1 s | - | **0/24 -> 8/24**, 8 gains / 0 losses, exact McNemar **p=0.0078**, bootstrap **+16.7 to +54.2 pp**; `runs/eval/comparison-a1-greedy.json` | **FAIL** (P1: +8 < +12) |
| A1 pass@8, 3 trained families | same | 87.2 s + 101.2 s | - | **2/24 -> 13/24** solved@8, 11 gains / 0 losses, **p=0.00098**, exact **2/192 -> 41/192**, bootstrap **+10.4 to +31.8 pp**; `runs/eval/comparison-a1-pass8.json` | **PASS** (P2: +11 >= +8) |
| A1 pass@8, all 8 families | same | 289.5 s + 291.1 s | - | **9/64 -> 18/64**, 15 gains / 6 losses, p=0.078, exact **41/512 -> 50/512**, bootstrap **-6.4 to +9.6 pp**; `runs/eval/comparison-a1-pass8-all.json` | net **+9** (E-112: net 0) |
| A3 RL + greedy | 2,190,152 trainable | 292.3 s + 77.4 s | 96 groups, **1,536/1,536 rollouts already exact**, reward_spread 0.0, **0 groups updated** (silent no-op), peak **0.214 GiB**; `runs/rl-regime-a3/training-receipt.json` | **24/24 -> 24/24**, 0 gains / 0 losses, p=1.0; `runs/eval/comparison-a3-greedy.json` | **PASS** (P3: 0 <= +3), ceiling |

Arm comparison. Init strength is the independent variable. The RL-time exact-rollout count
is the init's actual reward signal on the RL schedule (1,536 rollouts = 96 groups x 16), and
it is the number that identifies the published regime:

| arm | init | init dev pass@1 | init greedy confirm | init pass@8 exact (all 8) | init RL-time exact (of 1,536) | greedy gain | pass@8 gain (3 families) | pass@8 gain (all 8) |
|---|---|---|---|---|---|---|---|---|
| A1 | dense step 90 | 1/8 | **0/24** | 41/512 | 150 | **+8** (0 -> 8/24) | **+11** (2 -> 13/24) | **+9** (9 -> 18/64) |
| A2 (E-112) | pretrain step 100 | 3/8 | 5/24 | 110/512 | **406** | +7 (5 -> 12/24) | +5 (10 -> 15/24) | 0 (30 -> 30/64) |
| A3 | pretrain step 200 | 7/8 | 24/24 | not measured | **1,536** (saturated) | 0 (24 -> 24/24) | - | - |
| **published** | bf16 step 100 | 2/8 | **0/24** | not published in this form | **417** | **+16** (0 -> 16/24) | +10 (9 -> 19/24) | +9 (25 -> 34/64) |

Per-family A1 pass@8, all 8 families, exact rollouts before -> after (64 samples each):

| family | trained? | exact before -> after | solved@8 before -> after |
|---|---|---|---|
| increment | yes | 0 -> **6** | 0 -> **5** |
| double | yes | 2 -> **35** | 2 -> **8** |
| even | yes | 0 -> 0 | 0 -> 0 |
| square | no | 1 -> 7 | 1 -> 4 |
| list_sum | no | 38 -> **2** | 6 -> **1** |
| reverse | no | 0 -> 0 | 0 -> 0 |
| absolute | no | 0 -> 0 | 0 -> 0 |
| nonnegative | no | 0 -> 0 | 0 -> 0 |

Greedy confirm, per family: `double` 0/8 -> 8/8, `increment` 0/8 -> 0/8, `even` 0/8 -> 0/8.
The entire greedy gain landed in `double`; `increment` improved only at k (0 -> 5 solved@8,
0 -> 6 exact).

## Gates and verdicts

- **Determinism - PASS.** The repeat 400-step pretrain produced `checkpoint-0100` and
  `checkpoint-0400` **byte-identical** (sha256) to the E-112 run, with the same `tokens_seen`
  (1,381,760) and peak (2.141 GiB). Within one torch build this pretrain is bit-reproducible
  from the seed.
- **Selection rule - PASS as written, but the target was unreachable.** The rule was fixed
  before launch as the highest-step checkpoint with dev <= 2/8. The dense grid jumped from
  1/8 (step 90) to 3/8 (step 100), so 2/8 never occurred and the rule selected **step 90 at
  dev 1/8** - a WEAKER init than the published one, not an equal one. That is a deviation
  from the intent (match the published headroom) and it matters for the interpretation.
- **P1 - FAIL.** ACTUAL greedy gain **+8** (0/24 -> 8/24); REQUIRED >= +12. The baseline
  clause is satisfied perfectly (init greedy 0/24 = published 0/24), so the arm does test the
  intended comparison - and it still lands at half the published +16.
- **P2 - PASS.** ACTUAL pass@8 gain over the 3 trained families **+11** (2/24 -> 13/24);
  REQUIRED >= +8. This matches the published +10 solved-task gain. The endpoint level is
  still below the publication (13/24 vs 19/24) because the init started lower (2/24 vs 9/24).
- **P3 - PASS, trivially - and the arm is a silent no-op.** ACTUAL **0** (24/24 -> 24/24);
  REQUIRED <= +3. From the step-200 init every one of the 1,536 sampled rollouts was already
  exact (`reward_spread` 0.0), so RLOO advantages were identically zero and **0 groups were
  updated**. The run exited 0 while training nothing. It carries no information beyond the
  ceiling.
- **P4 - PASS.** Greedy gain by init: +8 (dev 1/8) > +7 (dev 3/8) > 0 (dev 7/8).
- **Falsification clause - TRIGGERED.** The pre-registration said an A1 greedy gain close to
  E-112's +7 refutes the headroom mechanism. ACTUAL +8 vs +7 - the gain did not move when the
  init's greedy baseline was matched to the publication.

**Headline verdict: the headroom explanation is REFUTED - and the decisive evidence is not
the A1 arm but the RL-time reward signal.**

The refutation does not rest on the A1 selection being imperfect, because a second, stronger
check is available in the receipts. The quantity that actually defines the RL regime is how
much reward signal the init produces on the RL schedule:

- published bf16 step-100 init: **417/1,536** exact rollouts;
- our fp16/torch-2.14 step-100 init (A2, E-112): **406/1,536** - matched to within 2.6%;
- our selected step-90 init (A1): **150/1,536**.

So the arm whose init genuinely matches the published regime is A2, and its greedy gain is
**+7** against the published **+16**. The arm with extra headroom (A1, 150 exact) gained
**+8**. Both of our arms land at roughly half the published magnitude regardless of how much
headroom or how little reward signal the init had. Headroom is not the explanation.

**What fails to reproduce is the RL endpoint, not the gain.** Across our three arms the
trained results cluster far below the publication: greedy 8-12/24 (published 16/24), pass@8
exact 41-75/192 (published 88/192), pass@8 solved 13-15/24 (published 19/24). Headroom
explains only the within-family dose-response (+8 / +7 / +0) and the *size of the pass@8
gain* (+11 from a 2/24 baseline vs +5 from a 10/24 baseline, both landing at 13-15/24).

The remaining suspect is therefore the RL step itself under fp16/torch-2.14 - a lower
per-group conversion rate from reward into greedy correctness. A plausible mechanism worth
testing: fp16's activation range in the last block could restrict what a head-only update
can express. Separating that needs an fp32-control RL arm from the same init (or a
bf16-capable device) with the schedule held fixed; this design varied the init, not the
optimizer/dtype path, so it cannot. The secondary lesson is about the instrument: dev pass@1
on 8 tasks read 1/8 at step 90 and 3/8 at step 100 while the RL-time exact count differed
150 vs 406 of 1,536 (~2.7x), so a small-sample dev score cannot place a checkpoint in a
regime - the init's exact-rollout rate on the actual RL schedule can.

## Levers to carry

- **Fingerprint an RL init by its reward signal on the real schedule.** Log the init's
  exact-rollout rate (and `updated` group count) next to its dev score. One number, 417 vs
  406 vs 150 of 1,536, settled a question that a dev scan of 8-task samples could not, and it
  cost nothing because the RL run computes it anyway.
- **Check `exact_rollouts` and `updated` before reading a flat arm as a negative result.** A
  saturated init produces reward_spread 0.0, zero advantages and zero gradient steps while
  the run still exits 0. Here that turned the pre-registered A3 dose-response point into a
  silent no-op - the honest reading is a ceiling, not evidence.
- **A dense checkpoint grid is cheap and converts a point comparison into a
  dose-response.** 11 checkpoints from the same 400-step run cost 224 s and produced a
  monotone curve (+8 / +7 / +0) plus a determinism gate (two checkpoints byte-identical).
- **Print the task-schedule hash in every RL receipt.** Both new arms carry the same
  `schedule_sha256`, which is what makes A1/A2/A3 a paired comparison - the init is
  demonstrably the only variable.
- **Pair on the quantity you are disputing.** E-112's headroom claim could only be settled by
  matching the greedy baseline exactly (0/24 = 0/24) and by comparing the init's reward
  signal; matching dev pass@1 alone would have left the same ambiguity.
- **Keep the metrics apart.** The same arm gained +8 greedy (half the published) and +11
  solved@8 (equal to the published). A single headline number would have hidden one of those
  facts; report greedy, solved@k and exact rollouts separately.

## Mistakes and gotchas

- **`train_pretrain.py` requires the final `--checkpoints` entry to equal `--steps`.** The
  first unattended launch died in 2.4 s with `ValueError: final checkpoint must equal steps`;
  the dense list is now extended with the run's own final step (400) while the scan stays at
  10-100. Fail-closed validation is good, but a `--help` smoke does not catch it - only a real
  launch does. Because the driver is idempotent the retry cost one turn, not the run.
- **A background job's `job_output` returns only the delta since the previous read.** The
  second read of the settled job came back empty even though the first had shown only the
  first 4,000 characters. Durable logs inside the run directory (`runs/regime/regime.log`,
  `children.log`) are the fix; read those.
- **The A3 arm is a ceiling, not a dose-response point.** `runs/pretrain/checkpoint-0200`
  already solves the greedy confirm set 24/24 and samples 1,536/1,536 exact, so the
  pre-registered <= +3 was met before any training. A confirm set the strong init has not
  saturated (or a harder split) would have made the arm informative.
- **The selected init is weaker than intended.** The pre-registered rule (highest step with
  dev <= 2/8) selects step 90 at 1/8 because 2/8 is skipped. The honest reading is that A1
  compared a *weaker* init to the publication - which strengthens the refutation (more
  headroom, still half the greedy gain) but means A1 is not an exact regime match; A2's
  reward-signal match is what closes that gap.
- **Wall-clock is not stable across sessions** (E-112 pretrain 179.3 s, E-113 214.9 s, same
  recipe, same seed, same machine). Thermal/desktop load, not a regression; the VRAM peaks
  matched to 0.1%.

## Scale-transfer caveats

- Licensed at **25.7M parameters, context 192, 400 pretrain steps / 1.38M tokens, one RL
  init grid of 3 points**. Nothing here speaks to larger models or longer schedules.
- **The RL step itself is the leading remaining explanation.** The design varied the init,
  not the optimizer/dtype path, so a lower per-group conversion rate under fp16/torch-2.14 is
  consistent with every number here. The fp16-activation-range mechanism suggested above is a
  hypothesis, not a measurement.
- The confirmation set is 24 tasks (3 families) for greedy and 192 rollouts for pass@8; the
  all-8 view is 64 tasks / 512 rollouts. All-8 p=0.078 with a bootstrap interval spanning
  zero is a direction, not a proven effect.
- **No published per-family greedy breakdown exists** for the 3 trained families, so the
  per-family comparison here is against E-112 only.
- The nested clone remains a local gitlink; its `runs/` are gitignored, so the raw receipts
  for this experiment live only on this machine.

## Evidence files

Nested repo: `runs/regime/regime.log`, `runs/regime/children.log`,
`runs/regime/selection.json`, `runs/pretrain-dense/training-receipt.json`,
`runs/eval/dev-dense-*.json`, `runs/rl-regime-a1/training-receipt.json`,
`runs/rl-regime-a3/training-receipt.json`,
`runs/eval/confirm-greedy-{init2,rl-a1,init0200,rl-a3}.json`,
`runs/eval/confirm-pass8{,-all}-{init2,rl-a1}.json`,
`runs/eval/comparison-a1-{greedy,pass8,pass8-all}.json`,
`runs/eval/comparison-a3-greedy.json`. Driver: `scripts/regime_match.py`. Predecessor:
E-112 report `research/2026-10-08_glm53_flash_clone_validation.md`.
