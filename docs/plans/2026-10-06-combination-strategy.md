# Combination strategy — combining the project's proven wins

Date: 2026-10-06. Companion to `research/WHAT_WORKS.md` (levers) and
`research/archive/diacritizer_failures.md` (dead ends). Status: PROPOSAL for
discussion; nothing here is registered until it becomes a pre-registered row in
`research/experiments.yaml`.

## The evidence base

Confirmed levers, with their strength:

| # | lever | evidence | state in current best |
|---|---|---|---|
| L1 | Stage-1 char-LM warm start | E-067 P1 / E-068 C1: +3.74 % vs a random-init control | **IN** |
| L2 | ctx 256 | E-067 P9: −6.39 % best-of-run, −4.02 % matched | **IN** |
| L3 | Gate-probe checkpoint selection | E-15…E-17; val_loss demonstrably misleads | **IN** |
| L4 | Sliding-window voting at inference | E-071: −1.34 %, free, p ≤ 3.2e-11 | **IN** |
| L5 | AdamW + 8-bit optimizer | Milestone B / E-12: VRAM headroom at parity | **IN** |
| L6 | Plain same-tokenizer KD | E-03: −6.71 %, ahead at every matched step | **NOT TRIED on D-line** |
| L7 | Soup as a post-hoc knob | E-07 / E-10: near-linear, no knee | **NOT TRIED on D-line** |
| L8 | Depth over width at matched params | E-13: −2.0…−3.9 pp at zero extra params | NOT TRIED at 1.08 M |
| L9 | Domain specialization + routing | E-12 arm A: Fadel 47.1 → 34.2 (−12.9 pp) | NOT TRIED at 1.08 M |
| L10 | Embedding transplant | E-05: −6.24 %, gap grows | superseded by L1 here |

Refuted axis, do not spend here: parameter scaling and architecture swaps (E-13,
E-14, E-23d), data volume and new label styles (E-23b, E-23c), postproc repair
(E-18).

## Reading the evidence

Three of the last four confirmed wins (L1, L2, L4) were either an **init** lever or
an **inference-time** lever. The training-side levers have plateaued: E-070 showed
extra steps past the plateau buy nothing on the gates while val_acc keeps rising.
So the highest expected-value moves are **inference-side combination** and **a new
training signal (distillation)** — not more steps, and not more parameters.

## Proposal, sequenced by cost

### Tier 1 — inference-only, no training, both checkpoints already exist (~half a day)

**T1-a. Checkpoint ENSEMBLE of step 15000 + step 40000.**
E-072 established these two are *complementary*: step 15000 leads the registered DER,
step 40000 leads `der_canonical` and macro_f1. Ensembling averages logits
(or probabilities) from both — deterministic, free at training time, and it combines
with voting. This is the single cheapest untried move with positive evidence behind
it. Registered metric decides.

**T1-b. Weight SOUP of the same two checkpoints.** A different mechanism (weight
space rather than output space). E-07/E-10 showed soup is near-linear in alpha with
no knee, so a small alpha sweep (e.g. 0.25/0.5/0.75) is cheap and informative. If
soup and ensemble both help, they are not redundant and can be stacked.

**T1-c. Voting window/stride sweep at ctx 256** (128/64 vs 192/96 vs 256/128). The
E-071 pre-registered secondary was never run. Cheap, and it bounds how much of the
voting gain is window-specific.

### Tier 2 — training, strong external evidence, never applied on this line (~10 h per arm)

**T2-a. Plain-KD from our own 30.1 M gold (stage2b2500) into the 1.08 M student.**
E-03's plain-KD is the strongest confirmed training lever in the repo and the D-line
has never used it; the teacher is 28× larger and same-tokenizer, which is the
condition E-03 requires.

**Design constraint, and it matters:** E-23c failed by distilling an EXTERNAL
teacher's *label style* into a saturated pool (−4.13). So this arm must distill from
**our own gold, keeping our label schema**, and must be compared at matched tokens
against C1/C4b — not against best-of-run.

**T2-b. E-072's metric-dependence as a training target.** Step 40000 is better on
`der_canonical` and macro_f1 while step 15000 is better on the registered DER.
If a deployment wants the canonicalised reading, the checkpoint choice changes. Worth
deciding explicitly which metric ships before spending on more training.

### Tier 3 — highest known upside, highest cost (N runs)

**T3-a. Domain-routed specialists.** The largest single move in D-line history was a
domain specialist (−12.9 pp on its own domain) that lost elsewhere. With a 1.08 M
model, 2–3 specialists plus a router is affordable in a way it never was at 30 M.
This is the only untried route with a *historical* effect size larger than everything
since. It also has a known failure mode to design around: the specialist loses
off-domain, so the router is not optional.

### Tier 4 — code work, no implementation exists

**T4-a. P2 variable-length representation** and **T4-b. P7 sentence dropout**. No
`var_len` or `sentence_dropout` support exists anywhere in
`diacritizer/`, so these are implementation tasks, not config arms.

## Recommendation

1. Run **Tier 1 now** — it is free, needs no GPU training window, and T1-a has direct
   evidence (E-072) that the two checkpoints are complementary.
2. Then **T2-a (plain-KD)** as the next training arm: strongest untried lever,
   strongest external evidence, one clean single-lever pre-register.
3. Hold **T3-a (specialists)** for an explicit user decision, because it multiplies
   training runs.
4. Do **not** spend on parameters, architecture swaps, or more steps past the
   plateau — all three are refuted on this line.

## Open question for the user

Which metric ships? The project selects on the registered 3-gate mean DER, but
E-072 showed step 40000 is better on `der_canonical` and macro_f1. That choice
determines whether T1-a/T2-a optimise toward step 15000 or step 40000.
