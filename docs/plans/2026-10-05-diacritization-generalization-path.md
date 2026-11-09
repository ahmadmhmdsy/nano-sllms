# Plan: Diacritization generalization path — diagnosis, techniques, pilot arms

Date: 2026-10-05 (revised 2026-10-06 against the repo's own measured evidence)
Status: plan written; E-067 pre-registered; P1 complete (pwsh-130/131) - verdict
CORRECTED 2026-10-06 to PASS (see §0 correction note); C0-R variance replicate launched
Origin: user request 2026-10-05 — the diacritizer models are failing to generalize
("only focused on a limited feature"), can't diacritize entire sentences or generalize
to unseen words. Search for techniques/layers/strategies, schedule pilot arms on small
data, combine winners, document what NOT work. Smoke first, pilot on the same data
slice, always eval on the four registered gates with multi-metrics.

---

## 0. Revision note (2026-10-06) — why this plan changed

The first draft listed eight techniques as "worth trying" from literature alone. A
sweep of the repo's own experiment registry (`research/EXPERIMENTS.md`,
`research/e19_bakeoff/*`) shows **five of the eight are already measured here**,
most with negative verdicts. This revision:

- **Demotes T1 (char-MLM warm-start)** from "the single biggest lever" to a
  **falsification arm**: warm-start has been tested at three scales in this repo
  (E-14 272h, E-17 272h, E-21-B 128h) and lost every time. P1 is kept only
  because the 64-hidden boundary is new and the arm is cheap.
- **Removes T5 (word-cache corrector) from the pilot arms.** It is measured
  (E-20/E-21): -0.3..-0.6 DER on gold, ~0 on micro. It is a cheap deployment
  add-on, not a pilot arm. It moves to the combine phase only.
- **Flags T3/T4/T8 with their measured priors** (E-23d BiLSTM, E-23b/c machine
  data, E-23c distillation) so we do not re-run them as if untried.
- **Removes E-23a from the negative list (2026-10-06 audit).** It was recorded
  NOT-ADOPTED against a wrong gold wn2024 baseline (48.65 = the gold's wn2014 +
  word-cache row); the measured gold wn2024 is 57.58, and rescored E-23a beats
  the gold on all four gates (3-gate 40.20 vs 42.64 at step 2000).
- **Corrects the pilot budget** from 2000 to **40000 steps** so every arm is
  comparable to the C0 control (E-064 A2, 40000 steps, best GATE_EXTERNAL mean
  45.77 @ 38000). A 2000-step arm cannot be compared to a 40000-step control.
- **CORRECTION (2026-10-06, after P1 landed).** P1 was first recorded as FAIL on
  an INVERTED metric. The registered metric is the mean DER over GATE_EXTERNAL
  and **lower is better**: `diacritizer/scripts/train.py:215` supersedes
  `best_gate_weights.pt` only on `mean_der < prev`, and `research/EXPERIMENTS.md`
  E-064 calls A2's 46.14 @40000 a *regression* from 45.77 @38000. Under the
  correct direction P1's best **0.4299 @32000 BEATS C0's 0.4577 @38000 by 6.1
  percent relative**, and P1 is better at all ten matched probe steps, with
  higher val_acc (0.9338 vs 0.9260) and lower val_loss (0.1991 vs 0.2216). P1
  therefore **PASSES** the pre-registered >= 2 percent band. Shape-matched
  char-LM warm-start is negative 3/4 (E-14 272h, E-17 272h, E-21-B 128h) and
  **positive at 64h (P1)**, so **L3 is NOT closed**. Caveat: `train.py` has no
  seed, so this is one draw; the pre-registered confirmation is **C0-R**
  (same-recipe replicate, `runs/diac/e67_gen/c0_rep.yaml`).
- **ADDED (2026-10-06).** Two gaps found while correcting the record: (i) every
  arm must report **`der_canonical`** beside the registered DER, because the
  registered DER is byte-order sensitive (MEMORY 2026-10-02; mishkala moves 7.25
  DER points); (ii) the plan's L2 lever is weaker than stated - the tokens are
  pre-packed `[n, ctx]` arrays, so a context arm needs **re-tokenization**
  (`tokenize_data.py --ctx`), not a config flip. 42.2 percent of prepared
  windows exceed 128 chars, so the arm is still worth running. **P9** added.

---

## 1. Diagnosis — why the model plateaus

The D-line has been stuck: E-063 (label smoothing won in-domain, arch/optimizer arms
FAILed), E-064 (lr bump won, step budget bound the run), E-065 (F2 overfit full v3
after ~5-6 epochs: gate 0.3974 @ 115000 -> 0.5768 @ 200000 while val_loss kept
falling), E-066 (R1 dropout+LS killed at step 109500, gates flat since 40000 and
behind F1/F2 at matched steps). We have been polishing the SAME architecture with
regularization/optimizer tweaks and hitting the same wall. The model has four
compounding limitations:

| # | Limitation | Evidence | Consequence |
|---|---|---|---|
| L1 | **Char-level, no word identity** | EMNLP 2025: variable-length (word+char) repr beat char-only; ACL 2020: word-level identity is what char-level loses | Model learns local char n-grams only — a "limited feature." Can't leverage Arabic morphology (root ك-ت-ب yields كَتَبَ، كُتُبٍ، يَكْتُبُ). |
| L2 | **ctx 128 — can't see the sentence** | Error analysis (Abandah et al.): 24% of case-ending errors are long-distance dependency (coordination, verb-subj/obj), an artifact of limited context | Model can't resolve case endings that depend on sentence position or long-range syntax. "تشكيل الجملة" (whole-sentence diacritization) is impossible. |
| L3 | **Effectively random init** | E-066 FINDING: stage1lm warm-start transferred 0/255 tensors (272-hidden LM vs 64-hidden s7). E-064 A2, E-065 F1/F2, E-066 R1 were all random init despite "warm-start" labels. **P1 fixed the shape (shape_skipped 0) and WON: 0.4299 @32000 vs C0 0.4577, +6.1 percent relative, better at all 10 matched steps** | Model has no prior understanding of Arabic; learns everything from the diacritization corpus alone. **L3 has a working lever at 64h (P1) and is NOT closed.** **ACTION TAKEN 2026-10-06: E-068 / C1 now applies the P1 warm start to the F2 full-v3 recipe (job pwsh-137; launch log `tensors_in 165 / shape_skipped 0`) — the first time the project-best recipe has actually been warm-started.** |
| L4 | **Single-pass, no review** | User diagnosis: "can't review, re edit, re use, benefit from other past result." Literature: statistical corrector (DER 4.00->2.80), sliding-window voting (WER 4.49) both add inference-time review the model lacks | Model commits to one forward pass. Can't look up how a similar word was diacritized, can't see a word in multiple contexts, can't correct its own output. |

These are not knob problems — they are capability gaps. The levers that address them
are below, now ranked by **measured prior** (not literature promise).

---

## 2. What does NOT work (do not repeat)

Evidence-backed negatives and our own closed arms. Each is a trap we already fell
into or that the literature has ruled out. **This section is the reason the arm set
is small.**

| Technique | Verdict | Evidence |
|---|---|---|
| **CRF layer** | NO improvement | EMNLP 2025 tested it, found none (consistent with Elmallah et al. 2024). Our 15-label head already can't double-mark. |
| **AraBERT / subword transfer** | HURTS at char level | EMNLP 2025: from-scratch transformer BEAT one initialized with AraBERT embeddings — distribution shift (subword vs char). |
| **stage1lm warm-start as implemented** | Silent no-op | E-066 FINDING: 0/255 tensors fit (272-hidden vs 64-hidden). Every "warm-started" run was random init. |
| **Shape-matched char-LM warm-start** | **WORSE at 128h/272h (3/3), WINS at 64h (P1)** | E-14 (272h, 302k win): gates 41.6/54.5/60.9/54.1 = worse than best from-scratch on ALL 4 gates, H REJECTED. E-17 (272h, 1.23M win, 27k-step LM): REJECTED, "init scale NOT the lever, 3rd confirmation." E-21-B (128h, 4k-step LM): mean 53.59 vs scratch 51.10 = +2.49 worse. **P1 (64h, shape_skipped 0): 0.4299 @32000 vs C0 0.4577 = 6.1 percent BETTER, better at every matched step** - the first warm-start win in this repo. **C0-R CONFIRMED 2026-10-06: the byte-identical replicate best 0.4539 @32000 vs C0 0.4577, so the no-seed init draw is worth only 0.82 percent, and P1 beats C0-R by 5.31 percent at all ten matched steps. The win survives.** |
| **Pure BiLSTM / attention arch** | FAIL band | E-23d: ZM BiLSTM+attn (4.5M) on our I/O contract = 55.83 mean vs arm C 48.34 = **+7.49 worse** (+9.46 vs the gold's 46.37), NOT adopted; char transformer stays. |
| **Machine-labeled data on the saturated gold** | **MIXED — E-23a WINS (corrected 2026-10-06)** | **E-23a is CORRECTED from "NOT adopted" to a WIN.** The recorded "+1.45 worse (wn24 +7.06)" used a wrong gold wn2024 baseline: 48.65 is the gold's wn2014 + word-cache row (E-20), while the measured gold wn2024 is 57.58. Rescored against the same gate CSVs, E-23a's best step 2000 (3-gate 40.20 / 4-gate 43.81) beats the deployed gold (42.64 / 46.37) on **all four gates**. E-23b (classical expansion): worse on every gate (4-gate 50.23). E-23c (ZM distillation): +4.85 worse (4-gate 51.22). **Exception:** E-22 (QCRI on the 2.98M micro) = -2.76 BETTER — machine data helps small models, and E-23a shows it can help the saturated gold too. |
| **Word-cache / majority-vote corrector** | Small gain, not a lever | E-20: gold + cache = -0.3..-0.6 DER. E-21: micro + cache = -0.003..-0.03 DER (zero). "Cache-lookup alone will NOT close the gap; cheap deployment add-on, not a teaching substitute." |
| **More dropout / label smoothing on same arch** | Plateau | E-066 R1: dropout 0.2 + LS 0.1 went flat at gate 0.4205 @ 40000 and fell behind F1/F2. |
| **Muon optimizer** | NaN | E-064 root cause: corrupted Newton-Schulz coefficient c=2.7885 (canonical 2.0315). |
| **Logit-adjusted CE** | Rejected on unseen data | E-063 S5: rejected. |
| **Shared-KV / shared-layer** | FAIL band | E-063 S1/S2: within noise of control. |
| **2.5x params (b65)** | Not the SOTA gap | b65 (76M params): -1..-4.6 pp DER, but ~50 pp gap to SOTA remains. |
| **Depth x2 (v2d28)** | Depth > params, but 10x too small | v2d28: -2.0..-3.9 pp DER vs v2b, but both axes 10x smaller than the SOTA gap. |
| **BPE / subword tokenization** | Breaks alignment | Fine-Tashkeel: BPE creates source/target segments of different lengths; diacritization needs strict char-level source-target alignment. |
| **Longer context alone** | Mixed | EMNLP 2025: short-range context usually sufficient (Mubarak 2019b), BUT 24% of case-ending errors are long-distance. |
| **Arch/optimizer feature ladder on gold** | FAIL band | E-63: "arch/optimizer features FAIL band, loss levers only improve in-domain, logit-adjust actively harmful unseen data." |
| **dedup-marks / legality-repair postproc** | No valid effect | E-18: byte-identical to baseline on all 4 gates; dropped from the pipeline. |

---

## 3. Techniques worth trying (ranked by measured prior, not promise)

Each technique below addresses one or more of L1-L4. "Prior" is what this repo has
already measured; "How to apply" is concrete; "Arm" is the pilot arm that tests it.

### T1 — Char-level MLM pretraining (fixes L3) — FALSIFICATION ARM

**What:** Pretrain a char-level Transformer with masked-language-modeling on a
large Arabic corpus, using the SAME architecture as the diacritizer (64-hidden,
18L), then warm-start the diacritizer from it.

**Prior (measured, negative 3/3):** see §2 "Shape-matched char-LM warm-start."
Warm-start lost at 272h (E-14, E-17) and at 128h (E-21-B). The mechanism is
consistent: the LM warm-start makes the model's OWN text prior better (lower
SFT val_loss) but bidirectional gate behavior worse.

**Why P1 is still run:** (a) the 64-hidden boundary is genuinely new — every
prior was 128h or 272h; (b) it is the only arm that directly tests L3, and L3 is
the one limitation with NO other candidate lever; (c) it is cheap (~50 min
pretrain + ~1 h fine-tune). **P1 is a falsification arm: a negative result closes
L3 for good; a positive result is the first warm-start win in this repo.**

**How to apply:**
1. Pretrain a char-LM with the s7 architecture (18L/64h/256ffn/8H/2KV/ctx128)
   on the v3s10 text via `train_lm.py` (causal, tied head). Config
   `runs/diac/e67_gen/p1_lm64.yaml` (20000 steps, eval/save every 1000).
2. Warm-start the diacritizer from `runs/diac/e67_p1_lm64/final/model.pt` via
   `pretrained_init` in `runs/diac/e67_gen/p1_ft64.yaml` (40000 steps,
   gate_eval_every 2000).
3. **Read-back requirement:** the fine-tune log MUST report `shape_skipped 0`.
   That line proves the warm start is real and not the E-066 no-op.

**Arm:** P1. Control = C0 (E-064 A2, random init, same 40000-step budget).

**RESULT (2026-10-06): PASS (corrected).** The pretrain (job pwsh-130) landed runs/diac/e67_p1_lm64/final/model.pt (20000 steps, best val_loss 1.8944). The fine-tune (job pwsh-131) reported **shape_skipped 0** (tensors_in 165, missing 0, unexpected 0) - the warm start was REAL, the opposite of the E-066 no-op. The best GATE_EXTERNAL mean was **0.4299 @32000 vs C0 0.4577 @38000 = 6.1 percent BETTER** (DER: lower is better), P1 better at all ten matched steps, val_acc 0.9338 vs 0.9260. The first draft recorded this as FAIL because the metric direction was inverted; see §0. Shape-matched char-LM warm-start is negative **3/4** (E-14, E-17, E-21-B) and **positive at 64h (P1)**; **L3 is NOT closed**. Caveat: unseeded (`train.py`), confirmation arm **C0-R** pending. Remaining arms still target L1, L2, and L4, but L3 is a live lever to combine.

### T2 — Variable-length (word+char) representation (fixes L1) — UNTRIED

**What:** Represent words with a dominant diacritization as single tokens
(fully diacritized); keep ambiguous letters as separate char tokens. The model
sees both word-level identity and char-level granularity.

**Prior:** none in this repo. EMNLP 2025 (Advancing Arabic Diacritization):
variable-length repr led to more accurate models than char-only for shorter
context windows; combining char + variable-length repr beat either alone.

**Why it works:** Gives the model word-level identity (L1) without losing
char-level granularity. Reduces token count, so the same context window covers
more of the sentence (partially L2).

**How to apply:**
1. Build a word-dominance map from the v3s10 training corpus: for each word,
   if one diacritized form covers >X% of occurrences, it's "dominant."
2. Tokenize: dominant words -> single token (fully diacritized); ambiguous
   words -> char tokens with the last letter (case ending) separate.
3. Train the model on this mixed representation.

**Arm:** P2. Control = char-only (C0), same data slice.

### T3 — Hierarchical two-level model (fixes L1 + L2) — PRIOR NEGATIVE (pure BiLSTM)

**What:** A word-level encoder (sees the whole sentence as a sequence of words)
+ a char-level encoder (sees each word as a sequence of chars) + cross-level
attention connecting them.

**Prior:** E-23d tested a pure BiLSTM+attention stack on our I/O contract and
lost by +7.49 mean DER. **But** E-23d was a flat BiLSTM, not a two-level
word+char hierarchy with cross-level attention. The hierarchical construction is
distinct and remains untested. D2/D3 (Deep Diacritization, WANLP 2020) report
WER 5.34%, 30.56% relative error reduction.

**How to apply:**
1. Word-level encoder: BiLSTM/Transformer over word embeddings, one vector
   per word.
2. Char-level encoder: BiLSTM/Transformer over char embeddings, one vector
   per char, conditioned on its parent word.
3. Cross-level attention: each char attends to all word-level encodings (except
   its own word).
4. Classifier: 15-class head on the refined char representations.

**Arm:** P3. Control = flat char-only (C0). **Needs code implementation** —
confirm scope before launch.

### T4 — Multi-task learning (fixes L1) — PRIOR MIXED (machine data)

**What:** Jointly optimize diacritization with auxiliary tasks: word
segmentation, POS tagging, syntactic diacritization.

**Prior:** the related machine-data arms split — E-23b (4-gate 50.23) and
E-23c (51.22) hurt the saturated gold (46.37), but **E-23a (QCRI on 30M gold)
WINS on all four gates** (corrected 2026-10-06; the recorded negative used a
wrong gold wn2024 baseline). Multi-task *labels* (not machine data) are a
different mechanism and remain untested. ACL 2020: joint models significantly
outperform baselines.

**How to apply:**
1. Add auxiliary heads: segmentation (IOB per char), POS (per word), syntactic
   diacritization (case ending per word).
2. Shared encoder, separate heads. Joint loss = diacritization + aux weights.
3. Build aux labels from the corpus or a morphological analyzer.

**Arm:** P4. Control = diacritization-only (C0). **Needs code + label
construction** — confirm scope before launch.

### T5 — Word-level statistical corrector (fixes L4) — MEASURED, NOT A PILOT ARM

**Status: already measured. Do not re-run as a pilot arm.**

- E-20: gold + our word cache (140.6M tokens -> 375,923 cached forms) =
  **-0.3..-0.6 DER** (fadel 33.23->32.88, sadeed 45.68->45.13, wn14
  48.99->48.65, abdou 41.46->40.90).
- E-21: micro + cache = **-0.003..-0.03 DER** (zero).
- Verdict: "cache-lookup alone will NOT close the Z-Mahmood gap; it is a cheap
  deployment add-on, not a teaching substitute."

**Residual gap vs literature:** the literature corrector (DER 4.00->2.80) is a
multi-stage pipeline (trigram -> bigram -> word edit-distance -> pattern
edit-distance) that also handles words NOT in the cache. E-20 tested only the
exact-match cache swap. A *stronger* corrector is theoretically distinct, but
the measured prior is weak and the mechanism (majority-vote word prior) is
something any model of this class already learns.

**Disposition:** keep as a **combine-phase add-on only** (cheap, +0.3-0.6 on
gold). Do not spend a pilot arm on it.

### T6 — Sliding-window voting (fixes L4, inference) — UNTRIED

**What:** Split sentences into overlapping windows, diacritize each window
separately, then vote on the final diacritized form for each word (majority
vote; ties favor the middle context).

**Prior:** none in this repo. Mubarak et al. (N19-1248, NAACL 2019): sliding window + voting
-> WER 4.49%, 63.3% lower than prior SOTA. D2/D3 also use voting.

**Why it works:** This is the "review, re edit, re use" mechanism at inference.
The model sees each word in multiple contexts and the consensus is more robust
than any single pass. It is the cheapest L4 lever (no retraining).

**How to apply:**
1. At inference, split each sentence into overlapping windows (e.g., 10-20
   words, stride 1-2).
2. Diacritize each window.
3. For each word, collect its predictions from all windows containing it;
   majority vote; ties favor the middle context.

**Arm:** P6. Inference-time on the C0 checkpoint. Control = single pass.
CPU-only (exempt from the one-GPU-job rule).

### T7 — Sentence dropout (fixes L1, regularization) — UNTRIED

**What:** Randomly dropout 20% of words in the word-level encoder during
training (replace their embeddings with zeros, preserve positions).

**Prior:** none in this repo. D2/D3 (WANLP 2020): sentence dropout led to better
generalization. Forces the model to not over-rely on any single word.

**How to apply:**
1. In the word-level encoder (T3) or as a char-level approximation: randomly
   mask 20% of input positions during training.
2. Config-only if the model supports it; otherwise a small code change.

**Arm:** P7. Control = no sentence dropout (C0), same data slice.

### T8 — Noisy-Student / data augmentation (fixes L1 + L3) — PRIOR NEGATIVE (distillation)

**What:** Use the best model to pseudo-label a large unlabeled Arabic corpus,
then retrain on the combined (gold + pseudo-labeled) data.

**Prior:** E-23c (ZM distillation, +315k windows) = **+4.85 worse** (4-gate). The
distillation mechanism (teacher labels on the same data) is measured and
negative. Noisy-student (SELF-training with noise, on UNSEEN unlabeled data) is
a different mechanism and remains untested, but the prior is cautionary.

**How to apply:**
1. Run the best checkpoint on a large unlabeled Arabic corpus.
2. Filter pseudo-labels by confidence.
3. Retrain on gold + pseudo-labeled data.

**Arm:** P8. Control = gold-only (C0). **Needs an unlabeled corpus** — confirm
scope before launch.

---

## 4. Pilot path — smoke, pilot, combine

The protocol is the E-063 trial-bed pattern: each technique ALONE on a small
data slice, then combine the winners, then full data.

### Phase 0 — Smoke (hard rule, every arm)

Every config has a `*_smoke.yaml` twin: total_steps 60, eval_every 60,
save_every 60. Launch the smoke FIRST; only if it completes end-to-end does the
real pilot launch. A smoke verdict is valid only while the boundary it exercised
is unchanged.

### Phase 1 — Singles (each technique alone vs control)

All arms use the SAME data slice for fair comparison. The slice is **v3s10**
(243,343 train / 9,613 val windows = 10% of v3), already built at
`data/diac/v3s10/tokens`. For techniques that need different data formats
(T2 variable-length, T4 multi-task), the underlying text is the same v3s10
slice; only the tokenization/labels differ.

| Arm | Technique | Type | Prior | Control |
|---|---|---|---|---|
| C0 | s7 winner recipe on v3s10 (E-064 A2) | config | — (baseline, already run) | — |
| C0-R | C0 re-run, same recipe, fresh draw | config | **DONE 2026-10-06: best 0.4539 @32000 vs C0 0.4577 = init-draw spread only 0.82 percent** | C0 |
| P1 | + char-MLM pretraining (64h) | code+data | **WIN: +6.1% vs C0, +5.31% vs C0-R; CONFIRMED 2026-10-06** | C0 |
| P2 | + variable-length repr | code+data | untried | C0 |
| P3 | + hierarchical model | code | negative (flat BiLSTM) | C0 |
| P4 | + multi-task learning | code+data | mixed — E-23a corrects to a WIN on 4/4 gates; E-23b/c hurt | C0 |
| P6 | + sliding-window voting | inference | **BUG FOUND 2026-10-06**: all-False attn_mask in `e67_p6_vote.py` collapsed the vote to class 0 (fatha); scored vote der_anyalt 0.7554 / macro_f1 0.0953 vs single 0.3712 / 0.6422 on fadel_test. Mask fixed to `torch.ones`; re-run queued behind C0-R; wn2014/wn2024 not yet produced | C0 checkpoint |
| P7 | + sentence dropout | config/code | untried | C0 |
| P8 | + Noisy-Student | code+data | negative (distillation) | C0 |
| P9 | + ctx 256 (L2 direct test) | data+config | **PREPARED 2026-10-06** (no longer blocked): the ctx-256 trial bed `data/diac/v3s10ctx256` is being built on CPU by `tokenize_data.py --phase v3s10ctx256 --ctx 256 --max-train 243343 --prepared-dir data/diac/prepared_rebuilt` (a new phase name was added so `data/diac/v3/tokens` is never overwritten), and `runs/diac/e67_gen/p9_ctx256.yaml` is written; launched 2026-10-06 as **pwsh-142** and **CLOSED — PASS: ctx 256 WINS**. Best 3-gate
0.4285 @36000 vs C0 0.4577 = **6.39% better** (4.02% on the 10 shared probe steps; 5.62% vs the
matched 20-probe C0-R replicate) with val_acc up **+2.1 points** (0.9470 vs 0.9261). 42.2% of windows
exceed 128 chars, so L2 is confirmed as a **second independent lever, not a null result** | C0 |

**P5 (statistical corrector) is NOT a pilot arm** — measured (E-20/E-21).

**Matched budget for training arms:** batch 32, accum 1, **40000 steps**
(A2-matched), eval_every 1000, gate_eval_every 2000, ctx 128, fp16 autocast,
identical minibatch stream (same slice length => fixed rng(777) stream draws
IDENTICAL batches across arms). At ~0.08-0.15 s/step this is ~1-1.7 h per arm.

**Selection bands (pre-registered):** an arm is a WINNER if its best
GATE_EXTERNAL mean improves >= 2% relative vs C0 AND val_acc does not fall
below C0 - 0.005. Ties -> prefer the simpler arm. Placebos (within noise of
C0) are dropped without combination. **Single-step wins are not evidence** —
the C0 gate curve moves +/-0.02 between adjacent steps (45.77 @ 38000 vs
46.14 @ 40000); a winner must hold its margin across the probe band.

### Phase 2 — Combinations (winners only)

Each combination is a NEW arm, pre-registered here before any of them launch:

- **C1** = best of {P1, P2, P3} (representation/arch) + best of {P4, P7, P8}
  (training signal). **CLOSED 2026-10-06 as E-068 - WIN**: the E-065
  F2 full-v3 recipe + the E-067 P1 real 64-hidden char-LM warm start. P1 is the
  only confirmed winning ingredient (P2/P7/P9 untried, P3/P4/P8 negative or
  mixed), so C1 is the winning recipe given the warm start it never had: F2's own
  config sets pretrained_init to the 272-hidden stage1lm, a silent no-op for the
  64-hidden s7 (255 vs 165 keys), so the project-best 0.3974 was in fact reached
  from random init. The C1 launch log confirms tensors_in 165 / shape_skipped 0.
  RESULT: best 3-gate GATE_EXTERNAL mean **0.3825 @90000** vs F2's 0.3974 =
  **3.74% better**, against the pre-registered 2% bar, with the val_acc tiebreaker
  passing on both readings and the win surviving a 30-step matched-range check.
  This is the **new project best** and the Phase-3 deployment candidate
  (`runs/diac/e67_c1_ft64_fullv3/gate_probe/best_gate_weights.pt`). Note the best
  gate step is only 45% into the budget, so the win is an early-trajectory effect
  of the warm start, not a late-training one.
  Label smoothing is deliberately excluded: the full-data LS arm (E-063
  stage3best_ls) scored 0.5810 vs S4's 0.3996.
- **C2** = C1 + best of {P6} (inference-time review) + T5 cache add-on
  (combine-only, measured +0.3-0.6)
- **C3** = kitchen sink of every winning ingredient

### Phase 3 — Full data

The best combination is trained on the FULL v3 corpus (2,433,430 train /
160,214 val windows) with the standard full gate_probe (4 gates, DER/WER +
text_preservation CSV) at registered gate steps. This is the candidate to
beat the deployed gold (stage2b2500) and the E-065 F2 best (0.3974).

---

## 5. Evaluation protocol

**Four gates (registered evaluation set - a SELECTION set, not a held-out test):**
fadel_test, sadeed25, wikinews2014, wikinews2024.
The GATE_EXTERNAL mean uses fadel_test + sadeed25 + wikinews2014
(wikinews2024 excluded as contaminated, train.py:39). All four are always
reported; the mean is over three.

**Multi-metrics (per gate):** DER, WER, DER_nocase, text_preservation.
Optionally extended with the metric harness (top-k accuracy, ECE/NLL/Brier,
macro-F1) for the finalists.

**Byte-order condition (mandatory from 2026-10-06):** report `der_canonical`
(the shadda-first canonical DER from `diacritizer/src/eval_metrics.py`) BESIDE
the registered DER for every arm, and name which condition a comparison used.
The registered DER is byte-order sensitive and the correction moved mishkala by
7.25 DER points (MEMORY 2026-10-02). T2/P9-style arms can change mark-run order,
so this is a scoring-correctness requirement, not an extra.

**Selection-bias caveat:** `gate_probe/best_gate_weights.pt` is chosen ON these
three gates, so the gates are a SELECTION set, not a held-out test. Report a
winner's margin with that caveat, and never describe the gates as "never seen"
after a checkpoint was selected on them.

**Noise band (measured 2026-10-06):** C0's own GATE_EXTERNAL mean spans
0.4577-0.4839 across its ten probes (5.72 percent), with a 5.37 percent maximum
adjacent-step jump. The pre-registered >= 2 percent band is therefore *inside*
single-step noise; a positive call requires holding the margin across the
matched-step band (P1 does), and C0-R sizes the run-to-run draw.

**In-training:** val_loss + val_acc on the 9,613-row unseen val slice (for
training arms) or the full val pool (for finalists).

**Fair comparison:** all arms in Phase 1 use the same v3s10 data slice and
the same four gates. The comparison is on the gates, not on training loss
(val_loss is not comparable across arms with different loss functions —
E-066 lesson).

---

## 6. Arm/tier schedule (the clear path)

```
Phase 0: Smoke (every arm, 60 steps, hard gate)
     |
     v
Phase 1: Singles on v3s10 (40000 steps each, same data, same gates)
     C0 (control, E-064 A2) -- C0-R (replicate, DONE: init draw 0.82%) -- P1 (char-MLM, WON +5.3% vs C0-R)
     P2 (var-len) -- P3 (hierarchical) -- P4 (multi-task) -- P6 (voting, partial run)
     P7 (sent-drop) -- P8 (noisy-student) -- P9 (ctx 256, needs re-tokenization)
     |
     v
Phase 2: Combinations (winners only, pre-registered)
     C1 = best repr/arch + best training signal  [CLOSED 2026-10-06 = E-068: WIN 0.3825]
     C2 = C1 + P6 voting + T5 cache add-on
     C3 = kitchen sink
     |
     v
Phase 3: Full v3 (200000 steps, 4 gates, multi-metrics)
     Best combination vs deployed gold (stage2b2500) and E-065 F2 (0.3974)
     >> DONE as E-068 = C1 (F2 recipe + P1 warm start), job pwsh-137, exit 0
     >> VERDICT: WIN 0.3825 @90000 = 3.74% better than F2, 10.3% better than the
        30.1M deployed gold, at 1.08M params
     >> P9 (ctx 256) CLOSED-PASS 2026-10-06: 0.4285 vs C0 0.4577 = 6.39% better,
        val_acc +2.1 pts -> a SECOND confirmed lever
     >> E-069 / C4 (C1's weights at ctx 256 on full v3, job pwsh-144) CLOSED:
        0.3872 @60000 - FAIL on its own best-of-run rule, but -3.4..-4.5% vs C1 at
        MATCHED TOKENS with all 12 probes better, and its best was its last probe
     >> E-070 / C4b (job pwsh-145, 60000 further steps from C4's final weights)
        CLOSED 2026-10-06: 0.3748 @15000 = NEW PROJECT BEST - 2.02% better than C1,
        6.49% at matched tokens, val_acc 0.96573. CAVEAT: the minimum rests on one
        early probe; the curve oscillates in a ~5.7% band with no sustained descent
        after step 15000, and the val-best (step 55000) disagrees with the gate-best.
        Candidate: runs/diac/e70_c4b_ctx256/gate_probe/best_gate_weights.pt
     >> E-071 / P6 voting CLOSED 2026-10-06: WIN - sliding-window majority voting
        (window 256 / stride 128) over those weights: 0.374712 -> 0.369702 = -1.34%
        FREE (inference-only, no retraining, deterministic); EVERY gate improves
        (fadel -1.85, sadeed25 -1.26, wikinews2014 -1.05%), paired McNemar p<=3.2e-11,
        der_canonical improves more (-2.5..-4.0%), non-degenerate. NEW PROJECT BEST
        0.369702. The PROCEDURE ships WITH the weights; weights alone are 0.374811.
        Caveat carried: P6 selects nothing, so E-072's question (is step 15000 really
        the best checkpoint, or is best-of-run picking probe noise?) stays open and
        is now pre-registered against the held-out abdou control.
     >> E-072 CLOSED 2026-10-06: CORROBORATED. On the held-out abdou control (15,091
        sents / 41,378 words, never used for selection) the ordering reproduces the
        gates exactly: 15000 (0.3751) < 40000 (0.3767) < 60000 (0.3945), step 15000
        beating both (McNemar p=4.8e-12 and p<1e-300). E-070's one-probe caveat is
        substantially RETIRED. Nuance recorded: on der_canonical and macro_f1 step
        40000 leads, so the top-two ranking is metric-dependent, not noise.
        Evidence: runs/diac/e70_c4b_ctx256/abdou/verdict.json
```

**Discipline:**
- One GPU arm at a time (single-GPU exclusivity). CPU-only arms (P6) are
  exempt and may run alongside.
- Pre-register as E-067 (and sub-arms) in research/experiments.yaml BEFORE
  any launch (rules/experiments.md).
- Managed pwsh background jobs only.
- The deployed anchor (stage2b2500) stays untouched until a registered
  verdict changes it.
- Honest FAILs recorded as FAILs.

**Open decisions (NEEDS_USER_DECISION before launch):**
1. ~~Confirm v3s10 as the pilot slice.~~ **Confirmed** (default taken).
2. ~~Confirm the arm set.~~ **Revised:** P5 dropped (measured); P1/P3/P4/P8
   flagged with negative priors. Confirm the revised set.
3. ~~Confirm the combination strategy.~~ **Revised:** C1-C3 as above.
4. For T3 (hierarchical) and T4 (multi-task): these need code implementation.
   Confirm scope (pilot first, or defer to a later phase).
5. For T1 (char-MLM): ~~confirm the pretraining corpus and budget.~~
   **Defaulted** to v3s10 text, 20000 pretrain steps. P1 pretrain + fine-tune
   ran (pwsh-130/131); verdict corrected to PASS. C0-R confirmation launched.
