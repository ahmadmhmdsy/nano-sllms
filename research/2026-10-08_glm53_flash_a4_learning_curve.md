# E-115 - Is the A4 RL endpoint a step-budget plateau? (learning-curve read, TASKS row 160)

Eval-only follow-up to E-114. E-114 exonerated the dtype/optimizer path (fp32 reproduced fp16
to the task) and left the RL endpoint at 12/24 greedy against the published 16/24. This
experiment asks the next cheapest question: is that endpoint where the *step budget* stops it,
or is the confirm curve still climbing when the 96-group budget runs out?

## Question

Is the A4 RL endpoint a step-budget plateau, or is the confirm curve still rising at the
96-group budget?

## Hypothesis

Pre-registered in TASKS 160 BEFORE the first GPU launch (verbatim block at the end of
TASKS.md), with the reading fixed in advance:

- **PLATEAU** if greedy solved(96) == greedy solved(72), or if the curve is non-monotone.
- **STILL RISING** if greedy solved(96) > greedy solved(72) and the curve is monotone
  non-decreasing.
- Either way, report pass@8 for the same checkpoints.
- Falsifier for the PLATEAU reading: any greedy gain after 72 groups.

Design: EVAL ONLY - no training. Re-evaluate the checkpoints A4 had already saved, against the
same before-file E-114 used. The 96-group point is REUSED from E-114's own receipts
(comparison-a4-greedy.json, comparison-a4-pass8.json) rather than re-measured, so the curve's
endpoint and E-114's endpoint are the same measurement by construction.

## Rig

- **Checkpoints** - runs/rl-regime-a4/checkpoint-{0024,0048,0072,0096} (saved by E-114's
  --checkpoints 24,48,72,96). No training, no new weights.
- **Before-file** - runs/eval/confirm-greedy-pretrain-0100.json and
  confirm-pass8-pretrain-0100.json (E-112's checkpoint-0100 evals, the same paired baseline
  E-114 used).
- **Evals** - scripts/evaluate.py (greedy, confirm split, 8 per family, 3 trained families =
  24 tasks) and scripts/evaluate_passk.py (pass@8, same split, 8 samples, same families),
  identical code/seeds/split to E-114. New driver stage: regime_match.py --stage a5.
- **Budget** - 6 evals + 6 comparisons, 11 min 7 s wall-clock total (09:07:25 -> 09:18:32) on
  the Quadro RTX 4000, cap 4.10 GiB, no training, no spill.

## Results table

| groups | greedy solved/24 | vs before (gains/losses, exact McNemar p) | pass@8 solved/24 (3 families) | vs before (gains/losses, p) |
|---|---|---|---|---|
| 24 | 8/24 | +3 (3/0, p=0.25) | 12/24 | +2 (2/0, p=0.5) |
| 48 | 8/24 | +3 (3/0, p=0.25) | 12/24 | +2 (2/0, p=0.5) |
| 72 | 7/24 | +2 (3/1, p=0.625) | 15/24 | +5 (5/0, p=0.0625) |
| 96 (E-114) | **12/24** | **+7 (7/0, p=0.015625)** | **15/24** | **+5 (6/1, p=0.125)** |

Per-family greedy solved before -> after:

| groups | double | increment | even |
|---|---|---|---|
| 24 | 5 -> 8 | 0 -> 0 | 0 -> 0 |
| 48 | 5 -> 8 | 0 -> 0 | 0 -> 0 |
| 72 | 5 -> 7 | 0 -> 0 | 0 -> 0 |
| 96 | 5 -> 7 | **0 -> 5** | 0 -> 0 |

Per-family pass@8 solved before -> after:

| groups | double | increment | even |
|---|---|---|---|
| 24 | 8 -> 8 | 2 -> 4 | 0 -> 0 |
| 48 | 8 -> 8 | 2 -> 4 | 0 -> 0 |
| 72 | 8 -> 8 | **2 -> 7** | 0 -> 0 |
| 96 | 8 -> 8 | 2 -> 7 | 0 -> 0 |

Wall-clock per eval: greedy 115.3 s at 72 groups (the 24/48 points ran in the same
~110 s range), pass@8 158.7 s at 72 groups. Comparisons are sub-second.

## Gates and verdicts

- **Literal pre-registered label - PLATEAU.** The rule's second clause fired: the greedy curve
  is non-monotone (8, 8, **7**, 12), because it fell by one task at 72 groups.
- **Substantive reading - NOT SATURATED.** The falsifier written into the pre-registration
  (any greedy gain after 72 groups) is met and exceeded: the final 24-group window delivered
  **7 -> 12, +5**, the largest greedy jump anywhere on the curve. pass@8 also fails to show a
  96-group ceiling in the same way - see below.
- **The rule was mis-specified, and I am reporting both.** A monotonicity boolean cannot tell a
  one-task dip from saturation. On a 24-task confirm set one task is 4.2 pp, so the 48 -> 72
  dip (8 -> 7) is inside the noise the clause was meant to protect against, while the 72 -> 96
  gain (+5, +20.8 pp) is five times larger. The right pre-registration is 'gain in the final
  window > 0 with the paired test', not a monotonicity flag. The literal label is recorded
  here as fired; the substantive conclusion is the opposite.
- **pass@8 reached its endpoint 24 groups early.** The 3-family pass@8 curve is 12, 12, 15, 15:
  flat from 72 groups, and equal to the E-114 endpoint. So the two metrics disagree about
  where the curve ends, and the pass@8 evidence is the one piece that genuinely supports a
  72-group saturation reading.
- **pass@8 led greedy by 24 groups.** increment pass@8 solved went 2 -> 7 at 72 groups while
  its greedy solved was still 0; greedy only moved 0 -> 5 at 96. The family's policy was
  already sampling correct answers before it could produce one greedily.
- **The gain migrated between families.** Greedy double reached 8 by 24 groups, held 8 at 48,
  then fell to 7 at 72 and 96; the endpoint's +5 comes entirely from increment going 0 -> 5 at
  96. The total 7 -> 12 hides a handoff - the endpoint gain is a late increment event, not a
  broad improvement.

**Headline: the A4 endpoint is not a step-budget plateau on greedy evidence, but pass@8 says
the policy stopped improving 24 groups before the budget ended.** The pre-registered
STILL-RISING branch applies: a longer RL arm (e.g. 192 groups) is a live lever and must be
pre-registered as its own experiment before any further GPU spend.

## Levers to carry

- **Read the curve off the checkpoints you already saved before buying more steps.** Six evals
  and 11 minutes, no training, answered a question that a 192-group run (~25 min plus evals)
  would otherwise have answered implicitly and expensively.
- **Reuse the endpoint receipt instead of re-measuring it.** Mapping the 96-group point to
  E-114's own tag makes the curve's last point and the published endpoint the identical
  measurement, which removes an eval-variance seam exactly where the conclusion is decided.
- **Prefer pass@8 when greedy looks flat, and greedy when pass@8 looks flat.** Here they
  disagreed (greedy still rising at 96, pass@8 flat from 72), and that disagreement is the
  finding - it bounds where the policy actually stopped improving.
- **Track per-family curves, not the total.** The endpoint's entire gain is one family
  converting late; a total-only curve would have shown a smooth climb and hidden the event.

## Mistakes and gotchas

- **My own pre-registered rule was too blunt** (monotonicity boolean vs last-window gain). It
  fired PLATEAU on a one-task dip while the substantive answer was the opposite. Recorded as a
  MEMORY lesson; the fix is to pre-register the final-window gain and its paired test.
- **The 24 and 48 group points are identical** (8/24 greedy, 12/24 pass@8, same gains/losses
  against the before-file). The comparison receipts record counts, not task identities, so I
  cannot say from these files whether the same tasks were solved at both points - only that the
  totals are equal. A per-task diff would need the raw confirm receipts.
- **pass@8 at 72 groups has p=0.0625** against the before-file; the 96-group value (p=0.125) is
  weaker because it has one loss. Neither is below 0.05, so the pass@8 reading is directional,
  not significant.

## Scale-transfer caveats

- Licensed at **25.7M params, context 192, one init (checkpoint-0100), one recipe, one seed,
  greedy over 24 tasks and pass@8 over 24 tasks**. A one-task dip or a +5 jump is 4.2 pp /
  20.8 pp here; on a larger confirm set the same curve would be smoother and the monotonicity
  question less noise-dominated.
- The curve is read from checkpoints saved every 24 groups, so nothing can be said about
  behaviour between those points.
- Whether a 192-group run would continue the greedy climb is NOT established here - this
  experiment only shows the 96-group budget did not exhaust the greedy curve.

## Evidence files

Nested repo: runs/eval/confirm-greedy-rl-a4-ckpt{0024,0048,0072}.json,
confirm-pass8-rl-a4-ckpt{0024,0048,0072}.json, comparison-a4-ckpt{0024,0048,0072}-{greedy,pass8}.json,
the reused E-114 endpoint receipts comparison-a4-{greedy,pass8}.json,
runs/regime/a5-curve.json (the parsed curve), runs/regime/regime.log + children.log. Driver:
scripts/regime_match.py --stage a5 (new; nested commit 1a54e28). Parser:
scratch/curve_e115.py. Predecessor: E-114 research/2026-10-08_glm53_flash_fp32_control.md.
