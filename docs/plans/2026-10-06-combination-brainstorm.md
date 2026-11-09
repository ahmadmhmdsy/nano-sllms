# Combination brainstorm — stacking every proven win (2026-10-06)

Supersedes the Tier list in `docs/plans/2026-10-06-combination-strategy.md` where the
two disagree. Written after an artifact inventory, so every claim below rests on
something I opened this session.

## 1. What I verified first (this changes the plan)

| fact | evidence |
|---|---|
| **64 probe checkpoints exist**, three lineages | C1 `e67_c1_ft64_fullv3/gate_probe/weights_step5000..200000` = 40 probes at ctx 128; C4 `e69_c4_ctx256/gate_probe` = 12; C4b `e70_c4b_ctx256/gate_probe` = 12, both at ctx 256 |
| Two already-extracted members | `runs/diac/e70_c4b_ctx256/eval/best15000/model.pt`, `.../best40000/model.pt` |
| **A working KD stack already exists in-repo** | `mex/src/kd.py` (`class TeacherEnsemble`), `mex/scripts/distill_cache.py` (teacher-ensemble logits cache), `mex/scripts/train_distill.py` |
| **A working soup implementation already exists** | `mex/scripts/merge_soup.py` — uniform mean **and** TIES sign-election, fp32, dtype-preserving |
| Teachers available | `stage2b2500`, `stage2final`, `stage2a2500`, `stage3best_ls`, `stage1lm` (114.9 MB each), plus `fadel_spec` (114.5 MB) |
| No D-line ensemble tooling | `diacritizer/scripts/e67_p6_vote.py` loads a single model; ensembling needs a small extension |
| An averaging-era arm exists but is not evidence | `runs/diac/e63_avg3_tr` (28L/272h/1088ffn, LS 0.1, stage1lm init) scored der 0.4794/0.6049/0.5918/0.5399 — an order of magnitude off the current model, so it says nothing about averaging at 1.08 M |

**The headline: this is cheaper than I told you.** I framed KD and soup as
untried work. They are untried *on the D-line*, but both have working
implementations in the mex line that need adapting, not writing.

## 2. Re-analysis of my own earlier suggestion

| earlier claim | verdict | correction |
|---|---|---|
| Tier 1: ensemble step15000+step40000 | **holds, understated** | it is not 2 members, it is 24 clean members in the ctx-256 group. Hand-picking two was arbitrary; greedy forward selection on a dev set is strictly better |
| Tier 2: plain-KD is the strongest untried lever | **holds, cheaper** | the machinery exists; the work is config + a cache build, not a new trainer |
| Tier 3: specialists, "your call" | **holds, but I missed a leak** | choosing a specialist per gate by gate score is selection on the test set. See §6 — this is the most important section |
| "Which metric ships blocks everything" | **I now think that was the wrong question** | we do not have to choose. Report all three every time, keep registered 3-gate mean DER as the *gate*, and let the deployment question be answered by the secondary columns. Nothing is blocked by it |
| "Tier 4 skip" | holds | P2/P7 are code work |

## 3. The axis map — what is genuinely orthogonal

The pipeline has nine independent places a proven lever can attach. Orthogonal
axes compose; two levers on the *same* axis usually do not.

| axis | proven lever (recorded effect) | status |
|---|---|---|
| init | char-LM warm start, E-067/E-068 (+3.74 %) | IN, proven |
| arch | depth-over-width at matched params, E-13 (−2.0…−3.9 pp) | untried at 1.08 M |
| context | ctx 128 → 256, E-067 P9 (−6.39 % best-of-run) | IN, proven; **axis never swept past 256** |
| objective | plain same-tokenizer KD, E-03 (−6.71 %) | untried on D-line |
| objective | joint LM + diacritic loss (LM head already exists) | untried anywhere |
| schedule | warm restart from the plateau peak | untried; motivated by E-070 |
| data | domain specialization, E-12 arm A (−12.9 pp in-domain) | untried at 1.08 M |
| selection | gate-probe selection, E-15…E-17 | IN — **but it selects on the gates** |
| aggregation | soup (E-07/E-10, near-linear) + logit ensemble | untried on D-line |
| decode | sliding-window voting, E-071 (−1.34 %, free) | IN, proven |
| decode | confidence / agreement weighting, window-offset TTA | untried refinements |

## 4. The stack — "combine all", as a layered design

Read top-down; each layer composes with the one above unless noted.

- **L0 init** — char-LM warm start (proven). Keep.
- **L1 architecture** — depth-over-width at ~1.08 M params (E-13). Zero extra params.
- **L2 context** — extend past 256 to 384. This is the single strongest lever's
  unexplored axis. *Constraint: disk.* v3ctx256 is 5.70 GB and `E:` has ~10.8 GB
  free, so 384 (≈8.5 GB) is feasible but tight; 512 is not.
- **L3 objective** — diacritic CE + λ·LM CE + KD from our own teacher. KD and the
  LM term are different regularizers and can coexist; both keep our label schema,
  which is the E-23c constraint.
- **L4 schedule** — a **new cosine cycle from the plateau peak**, not more steps
  on the old one. E-070 showed the old schedule's tail buys nothing; a restart is
  a different mechanism, not an extension.
- **L5 data** — per-domain fine-tune of the L3 student.
- **L6 selection** — **on a held-out dev set, never on the gates** (§6).
- **L7 aggregation** — soup *within* a lineage, ensemble *across* lineages (§5-A/B).
- **L8 decode** — voting (proven) + confidence weighting + window-offset TTA.
- **L9 reporting** — gates scored once, with bootstrap CI and McNemar, all three
  metrics reported.

## 5. The creative-but-grounded ideas

**A. Two-level aggregation: soup within a basin, ensemble across basins.**
Adjacent probes in one run sit in the same loss basin, so averaging their
*weights* is safe (E-07/E-10 found soup near-linear). Different lineages (C4 vs
C4b, different init parents) sit in different basins, so averaging their weights
is not meaningful — but averaging their *logits* is, because decorrelated errors
are exactly what an ensemble exploits. Nobody has combined the two levels on this
line. Design: soup C4's probes into one model, soup C4b's probes into another,
then logit-ensemble the two soups.

**B. Trajectory ensemble by greedy forward selection.**
We have 24 clean ctx-256 members. Rather than hand-picking, run Caruana-style
greedy selection: start from the best single member, repeatedly add the member
that most improves the dev metric. This is the standard ensemble-selection
method, it is free (inference only), and it will find members I would not have
guessed.

**C. Ensemble-teacher self-distillation — the keystone.**
Build the best ensemble from A+B, then distil *that ensemble* into one fresh
1.08 M student warm-started from the char-LM. This converts every free ensemble
gain into a single deployable model, and it stacks with L1/L3/L4. It is exactly
what `mex/scripts/distill_cache.py` was built to do, and because the teacher is
our own model the label-style trap that killed E-23c does not apply.

**D. Warm restart instead of more steps.**
E-070's plateau is a schedule fact, not a capacity fact: val_acc kept rising
while the gates did not. A fresh cosine cycle (possibly with a brief LR re-warm)
from the peak checkpoint is the untried response to a plateau. Cheap relative to
a new run.

**E. Joint LM + diacritic objective.**
`train_lm.py` already notes the label head is unused and LM logits are the tied
embedding product, so adding a λ·LM CE term to `train.py` is a small change. It
is a regularizer on a 1.08 M model trained on 2.4 M windows — the regime where
auxiliary objectives usually help. (Note: this is *not* the refuted MTP aux head,
which was a separate prediction head; this is a loss term on the existing tied
head.)

**F. Context beyond 256.** The best lever's axis was never swept past its first
setting. Highest-confidence unexplored direction, gated only by disk.

**G. Decode refinements.** E-071 used plain majority voting. Two cheap upgrades:
weight each window's vote by the model's confidence, and add window-offset TTA
(predict at offsets 0, ±32, ±64 and merge). Both are orthogonal to ensembling and
compose with it.

**H. Cross-context ensembling — flagged as a stretch, not a plan.** C1's 40
probes are ctx 128; the 24 clean members are ctx 256. Feeding C1 a 256-wide
window makes it extrapolate over positions it never trained on (RoPE), so I would
not put this in the main line. It is a cheap thing to try *once the clean
ctx-256 stack is measured*, as a diversity source, and I would expect it to fail
or help only marginally.

## 6. The leak I missed — and the fix that matters most

**Gate-probe selection selects on the registered gates.** `best_gate_weights.pt`
is chosen because it scored best on the very corpora we then report. That is
selection on the test set, and E-072 already caught it in the act: on an
independent corpus the ordering between step 15000 and step 40000 *inverts* on
`der_canonical` and macro_f1. E-072 was necessary precisely because the selection
had no independent check.

This is the rule adopted today ("selection is not measurement"), applied to the
D-line's own history. The fix is structural, not statistical:

1. **Designate a selection dev set that is not a gate.** `abdou_test` (15,091
   sentences, never trained on, not registered) is already doing this job
   informally in E-072. Make it the formal dev set.
2. **Select every future checkpoint, ensemble member, and soup weight on dev.**
3. **Score the registered gates once**, at the end, with CI and McNemar, and
   report that number as the verdict.
4. **Per-corpus specialist routing is the same leak** in a bigger form: picking
   the best model per gate by gate score guarantees a gain that is partly
   selection. Route by a dev-trained router or a dev-chosen rule, or report the
   routed number with the selection caveat attached.

If we adopt only one thing from this document, adopt this. Every other idea
produces a number that this problem can inflate.

## 7. What NOT to combine (refuted on this line)

Parameters/architecture swaps (E-13, E-14, E-23d), external corpora and new label
styles (E-23b, E-23c), label smoothing (harmful: 0.5810 vs 0.3996), MTP auxiliary
head (E-42), postprocessing repair (E-18, byte-identical), skew-KL (E-03),
mount-fill (E-11), stage-2 without reset (E-14), whole-corpus v4 init (E-17).
Combining any of these with the stack adds cost and noise, not accuracy.

## 8. Ranked plan

| phase | work | GPU | expected |
|---|---|---|---|
| 0 | score existing members on dev; greedy ensemble; soup within lineage; ensemble across; voting + confidence + TTA. **Inference only.** | no training window | the free tier; E-072 says the members are complementary |
| 1 | build the ensemble teacher cache (`distill_cache.py` adapted) | inference | enables phase 2 |
| 2 | KD student from the ensemble teacher, char-LM warm start, λ·LM term, warm-restart cycle | one window | the keystone: single model at ensemble-ish accuracy |
| 3 | ctx 384 rebuild + depth-24/48 at 1.08 M | disk + windows | extends the two strongest structural levers |
| 4 | domain specialists from the phase-2 student + dev-trained routing | N windows | largest in-domain move historically |

Phase 0 needs no GPU training window and is where I would start today.

## 9. The metric decision — recommendation, not a question

Keep **registered 3-gate mean DER as the gate** (it is pre-registered, and
switching it now would break comparability with every prior verdict). Add
**`der_canonical` and macro_f1 as mandatory secondary columns** in every verdict.
If the deployment target is readable Arabic, canonical is the number that matters
and it will be visible; if it is research comparability, the gate is unchanged.
The block dissolves — report all three, choose none.

## 10. Open questions for you

1. **Dev set**: do you accept `abdou_test` as the formal selection dev set, and
   the gates as score-once? This is the one decision that changes how every later
   number should be read.
2. **Disk**: ctx 384 needs ~8.5 GB against ~10.8 GB free. Reclaim space, or skip
   to depth-over-width first?
3. **Scope**: phase 0 only (free, today), or phase 0 + commit to the phase-2 KD
   run?

## 11. Status update — what has since been MEASURED (2026-10-06, after E-075)

The three open questions in §10 are answered: (1) **accepted** — `abdou_test` is the
formal selection dev set and the registered gates are reported score-once; (2)
**depth-over-width first**; (3) **adaptive** — each phase is chosen from the previous
phase's result.

### 11.1 Phase 0 results, in the order they landed

| experiment | lever (axis) | result | verdict |
|---|---|---|---|
| **E-073** | uniform weight soup of the 12 C4b probes (aggregation) | 3-gate **0.352209**, abdou **0.349044** vs best single probe 0.374792 / 0.375141 = **-6.03% / -6.96%** | **WIN** — new project best, no training |
| E-073 negative control | soup of C4b **+** C4 (cross-basin) | **+19.89%** (0.449353) | proves same-basin membership is load-bearing |
| **E-074** | is `abdou_test` a valid leak-free selector? | Spearman rho **+0.909**, 95% CI [+0.658, +1.000], = **94.4% of the 0.963 ceiling**; abdou argmin inside gate top-3 | **USABLE** — selection on abdou is validated |
| **E-075** | sliding-window voting **on top of** the soup (decode) | gates 0.352195 → **0.346827** (-1.52%); abdou 0.349032 → **0.341475** (-2.17%); wins on all five corpora | **WIN** — the two levers **compose additively** |
| **E-076** | greedy (Caruana) dev-weighted soup vs uniform (aggregation) | *running* | pre-registered: GREEDY WINS / TIE / GREEDY WORSE |

**The stack so far: 0.346827 on the registered 3-gate mean**, down from 0.374792
before the soup and 0.369702 before voting — **-7.46% against the best single probe**,
with no training, no new data, and no new checkpoint. §4's L7 and L8 are now IN.

### 11.2 What this changes in the plan

- **§3 axis map**: `aggregation` moves from *untried on D-line* to **IN, proven**;
  `decode` gains a second confirmed entry (voting composes with a merged model, not
  only with a single one).
- **§5-A/B (two-level aggregation, greedy selection)**: the soup half is now measured
  and wins; the greedy-selection half is E-076. The *ensemble-across-basins* half is
  now in doubt — E-073's `all24` control says averaging weights across basins costs
  +19.89%, so if cross-basin combination is to work it must be at the **logit** level
  (prediction/ensemble), which remains untried and is still worth one arm.
- **§6 (the leak)**: resolved in principle. E-074 supplies the missing piece — a dev
  set whose rank agreement with the registered metric is measured against the metric's
  own reliability ceiling, rather than merely asserted. Future selection uses abdou.
- **§2 Tier 1 correction** ("not 2 members, 24 clean members; greedy selection is
  strictly better than hand-picking") is now the live question in E-076.
- **§4 L2 / §10 question 2 (disk)**: **22.272 GiB free** after the user-authorized
  reclamation, so ctx 384 (~8.5 GB) is fundable — the disk constraint on L2 is gone.
  Depth-over-width (L1) still goes first, per the user's answer, and needs no disk at
  all because it reuses the existing `v3ctx256` bed.

### 11.3 Revised next steps

1. **E-076** — finish greedy soup selection; if GREEDY WINS, the deployed model is the
   greedy soup and the uniform soup is superseded as the anchor.
2. **Voting-window sweep** — E-075's -1.52% is bounded by `window == ctx`, so 192/96
   and 128/64 on the merged model are the next free lever. Select the window on abdou
   dev, report the gates once.
3. **Cross-basin logit ensemble** — one arm, to test §5-A's second half directly.
4. **Phase 1: depth-over-width at 1.08 M** (E-13), reusing the `v3ctx256` bed.
5. Then phase 2 (KD student from the merged teacher), phase 3 (ctx 384), phase 4
   (specialists + dev-trained routing), each gated on the phase before it.

### 11.4 The correction I owe the record

§5-B was written on the assumption that a reliable member-to-member quality gradient
and an averaging benefit are in tension. E-073 refuted that: member **ordering** and the
**averaging** benefit are orthogonal — averaging cancels basin-curvature noise, it does
not erase member identity. The practical consequence is that E-074's +0.909 rank
agreement does **not** argue against merging, and a future reader should not treat it
as if it did.
## §12 — E-076 result: greedy weighting REFUTED, and the reason matters more than the verdict

**Verdict: GREEDY WORSE (rejected).** Recorded in `research/experiments.yaml` diacritizer/E-076.

| arm (both voted 256/128) | fadel | sadeed | wn2014 | wn2024 | abdou | gate_mean3 |
|---|---|---|---|---|---|---|
| uniform soup (E-073/E-075) | 0.267606 | 0.366991 | 0.405884 | 0.520723 | 0.341475 | **0.346827** |
| greedy soup (E-076) | 0.269509 | 0.368711 | 0.409494 | 0.523926 | 0.343443 | 0.349238 |

Greedy loses on both readings (+0.70% gates, +0.58% abdou) — outside the pre-registered 0.5% TIE
band, so the rule fires as GREEDY WORSE. It also loses on the single pass (0.354793 / 0.351044 vs
0.352195 / 0.349032).

### The mechanism, and it is the transferable part

`weights.json` shows the final counts as `[0,0,1,0,1,0,1,0,0,1,0,0]`: R=4 greedy rounds put support
on **four** of the twelve members at 0.25 each. The uniform soup puts 1/12 on **all twelve**. The
greedy search therefore never had the uniform solution in its search space — it optimised faithfully
within a strictly smaller family and its dev trace improved monotonically
(0.375332 → 0.358318 → 0.354069 → 0.351603) without ever approaching the answer the reference arm
already had.

So this refutes **R=4 greedy**, not weighted averaging. A larger R, or a uniform-seeded greedy walk,
is a different and still-untested arm. The generalisable rule is in `MEMORY.md`: **before comparing a
search to a baseline, check the baseline is inside the search space** — otherwise the comparison
measures the support restriction, not the weighting scheme.

### What this changes in §11.3

Item 1 is resolved: **the uniform soup stays the deployed anchor** at 0.346827, and the greedy line is
closed rather than superseded. Items 2–5 stand unchanged and are now pre-registered as E-077
(voting-window sweep), E-078 (cross-basin logit ensemble) and E-079 (depth-over-width, paired).

### Bonus: the composition replicated

On this *different* soup, voting again beat the same model's single pass by −1.57% gates / −2.17%
abdou, against E-075's −1.52% / −2.17%. Two soups, two independent readings, same additive
composition — one replication is a result, two is a property.

