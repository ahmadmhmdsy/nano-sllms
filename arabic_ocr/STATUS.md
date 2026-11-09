# Arabic OCR — honest status (2026-10-01)

## (ab) Routed system, Step 1 PRE-LAUNCH - AMIDDA in, band measured down to 64-112, objective Pareto-only

TASKS 150 / 151 / 155. Not a verdict yet: this records the pre-launch state only.

- **Data**: the 4 licence-verified AMIDDA sub-sets (30,404 lines) are streamed in via
  `fetch_hf_lines.py`. Measured **~118 KB/line** (baybars/train), 2.2x the Muharaf
  precedent, so ~3.6 GB lands on E:. The aggregate `calfa-ai/AMIDDA` is deliberately NOT
  used - its card says "other" and it also contains our own KHATT/Muharaf test splits.
- **Leakage gate**: `arabic_ocr/scripts/audit_amidda_leakage.py` writes
  `data/hf/amidda_audit.json`; the trainer now REFUSES to start without a clean one.
- **Augmentation**: multi-scale height **64-112** (measured; the pre-registered 64-160
  spilled into WDDM shared memory at 9.94 GiB on an 8 GiB card WHILE EXITING 0) plus
  resolution degradation 27 -> 96 px at p=0.5, the direct counter to the E-100 27 px
  ISIPPT regime.
- **Objective**: Pareto over Plan A and Plan B on all four full sets - muharaf_test <= 21.58,
  khatt_teklia_test <= 16.46, khatt_bench <= 25.25, isippt_bench <= 52.68 - delta tuned on
  `muharaf_val` only.

- **Leakage audit executed**: 135 exact + 2 blob + 0 hard hits out of 30,404 lines. They are
  coincidental short-string collisions (median matched length 2 chars; the only 20+ char hits are
  the Basmala), not a shared corpus. All 137 are dropped from training via the audit's drop list,
  applied by `finetune_muharaf.py` and confirmed in the launch log. Dedupe: 1,003 duplicate lines
  across sub-sets (3.3%), 0 duplicate pages. Median ingested image height 123 px.
- **Launched 2026-10-07** (job `pwsh-43`): 56,528 optimizer steps (~3.7 h), loss 1.50 -> 1.28 over
  the first 1,000 steps at 0.22 s/step, 7.3 GiB allocator ceiling armed. Verdict pending.

- **STOPPED 2026-10-07 by explicit user request** at step 6,000 / 56,528 (loss 1.1109).
  No verdict - the Pareto gates are UNTESTED. State preserved: `arabic_ocr/runs/stage06d/ckpt_last.pt`
  (step 5,000, + optimizer state), `ckpt_step001000..005000.pt`, `train_log.jsonl`. GPU released
  (211 MiB used, 0% util).

## (aa) Option (D) VERDICT: FAIL - from scratch never escaped the CTC blank plateau

Stopped at step **4,250 of 52,992** (~1 h 15 m). **G1 FAIL, G2 FAIL, G3 not reached.**

Measured evidence, not impressions:

- **Loss 3.78/char** at step 4250, decaying geometrically (0.76 -> 0.36 -> 0.24 -> 0.16 -> 0.14 per
  1000 steps) toward an asymptote near **3.5 = the training targets' own unigram entropy (3.122
  nats)**. That is the "predict the character marginal, ignore the image" solution.
- **Hypotheses track only sequence LENGTH:** ref(19) `لبنان في ١٧/١٠/١٩٦٠` -> `ا`; ref(27) `حضرة
  محافظ جبل لبنان الافخم` -> `ا`; ref(44) -> `اة ال`; ref(90) -> `اة ال ل.`.
- **Emission rose 5.8% -> 9.4% while CER stayed flat ~92%**, and `CER ~= 100 - emission` held at both
  checkpoints. Train/test gap at step 1000 was **ZERO** (94.53 vs 94.54) - nothing memorised.
- Eval cards flat: muharaf_test 94.17 / 95.56 / 91.70 / 92.61; khatt_teklia_test 93.71 / 93.46 /
  96.67 / 93.21 (steps 1000/2000/3000/4000).

**Not a bug - proven, not assumed.** On the step-4000 checkpoint: the SAME image twice gives mean
|delta log-prob| **0.000000** and 100% argmax agreement; TWO DIFFERENT images give **3.303** and 26.6%
agreement. The image reaches the model; the model never learned to use it. This is a blank-dominated
degenerate optimum.

**Root cause (found in the model directory).** `models/ctc_backbone/training_metrics.json` shows the
published Muharaf backbone is a **chain of warm-started fine-tunes** -
`init_checkpoint` -> `..._v6_antideletion_finetune/model.pt` -> `..._v3_h96_finetune/model.pt` -
with `checkpoint_load_info.missing_keys` = `height_attention.score.*` (the attention head was newly
initialised on an already-reading backbone). Its **epoch-1 train_loss was already 0.524**, final
test_loss **0.768**. Ours was 3.78. The 22,091-line split can fine-tune a reader; it cannot discover
CTC alignment from a random init in 16 epochs.

Our recipe also omitted two of the authors' mechanisms: `augmentation_mode: "light"` (ours is
`"augmentation": "none"`) and `train_blank_logit_penalty: 0.2` (ours has no training-time blank
penalty). Authors: lr 5e-5, OneCycleLR, batch 4 x accum 2, 20 epochs.

The run is **resumable** (`--resume` from `ckpt_last.pt` with the full recipe) if the complete 16
epochs are wanted. The next move should be a WARM START - see HANDOFF (aa).


> **READ THIS FIRST (added 2026-10-06).** This file is the honest *live*
> status home for the OCR line; the durable audit of the whole line is
> [RETROSPECTIVE.md](RETROSPECTIVE.md). **Current state: TASKS 134 done (E-90/E-91);
> the next step (training) remains USER-GATED at TASKS 131, superseded by
> TASKS 137.** **2026-10-06 UPDATE: the user chose option A (the Muharaf-centric
> ladder). Step 0 is now measured on the REAL held-out splits (E-92: Muharaf
> test 22.11 raw, KHATT test 10.99 for FixFips) and the (B1') training run is
> pre-registered and running as TASKS 138.** Stage 1 PASSED; Stage 2 PASSED (hybrid detector, 0.925 line
> recall@IoU0.5); Stage 3 from-scratch FAILED (E-87, 0.9100 CER); Stage 0.5b
> warm-start FAILED (E-88, 71.74 vs the 57.80 control). Stage 0.5c / route (D):
> the two in-domain teachers score 46.93 (FixFips) and 23.64 (Muharaf) on
> `data/hflines/val`, but those headlines are CONTAMINATED -- 9.6% and 79.9% of
> the val lines are in the respective teacher's OWN training data. On the 174
> lines held out from both, FixFips is 61.88 (4.08 pp WORSE than the control)
> and Muharaf 54.06. **The best honest number on the corpus is 46.93%; the
> honest transfer number is ~54-62%.** The <5% target is NOT met for handwriting.
> Sections below are newest-first. Section (d) was a duplicate of (c) and was
> merged 2026-10-06. [PROJECT_REPORT.md](PROJECT_REPORT.md)
> and [COMPARISON.md](COMPARISON.md) are superseded.

## (z) 2026-10-07 -- Option (D) LAUNCHED: 15.3M from scratch at height 128 (pre-registered)

User chose (D). The pre-registration is in TASKS 146, written BEFORE the launch:

- **Architecture** `d_model 384, 8 encoder layers, 8 heads, ffn 1536` = **15,336,743 params,
  measured** with `arch_param_probe.py` (repo rule R5) - 3.61x the 4.25M teacher and the same
  scale as the 15.06M FixFips CRNN. Trained **from scratch** (dims differ from the teacher,
  so no warm start is possible); the charset is still inherited from the published checkpoint
  because it is the label space, not a learned parameter.
- **Height 128.** E-102 proved a height change is only valid at training time; E-100 measured
  that 98%/93% of KHATT lines are downscaled 1.55x at height 96, while the Muharaf source is
  only 60 px and is already upscaled 1.6x at 96. KHATT therefore has real resolution to win
  back and Muharaf has none - 128 is the compromise.
- **Speed 0.72 s/step measured** by `--smoke 30` at h128 -> 3312 steps/epoch -> ~40 min/epoch
  -> **~10.8 h for 16 epochs**. 26,490 training items.
- **P2 is the hypothesis**: beat the best adapted KHATT (21.82) so G1 (<= 21.00) finally
  passes. **P4 is the stated risk**: 26,490 lines is thin for 15.3M from scratch, so Muharaf
  may regress - that would be a genuine trade-off, not a bug. A stop rule is pre-registered:
  if KHATT raw is still > 35 after ~3 h, stop and report.

`finetune_muharaf.py` gained `--target-height/--scratch/--arch-*/--warmup-steps/--resume` and a
`--stage` label; every one defaults to the pre-(D) behaviour, and a regression check confirmed
the teacher still scores the same (G2 true, G1 false at 100 lines/set). `--resume` restores
model + optimizer + scheduler + step, with a deterministic per-epoch batch order so a resumed
run reproduces the same batches. The old `evaluate()` silently scored at height 96 via
`recognise_line`'s default - that would have mis-scored every intermediate eval of this run,
and is now threaded through.

Probe: `arabic_ocr/scripts/arch_param_probe.py`.

**Gate results (both PASS).** `check_ckpt_reload.py` on the step-1000 checkpoint: height 128,
15,336,743 params, d=384/8L/8H/1536, 166 classes, logits [64, 1, 166], optimizer state present.
The resume self-test (CPU-only, isolated out dir) restored epoch 0 / batch 1000 / step 1000 and
continued to step 1003, exit 0 - so a late OOM or session death is survivable.

**First eval at step 1000: muharaf_test raw 94.17, khatt_teklia_test raw 93.71.** Loss 4.97
against ln(166)=5.11, so the model is barely past the predict-the-prior solution: this is the CTC
blank plateau at 1.9% of training, not a result. The 3h stop rule is the decision point.

**Diagnostic at step 1000 (CPU-only emission probe, `--target-height 128`).** Train and test are
indistinguishable - muharaf_train emission 5.8% / raw CER 94.53 vs muharaf_test 6.2% / 94.54 at
delta 0. A **zero train/test gap** means the model has memorised nothing yet, which is the
signature of the CTC blank plateau rather than a broken run. The blank penalty barely helps
(train 94.53 -> 91.62 at delta 1.5) because there is almost nothing to recover. Measured, not
assumed; still not a verdict.

### Closeout plan - exact commands, written BEFORE the run finished

**Resume must repeat the FULL recipe.** `--resume` rebuilds the model from `--arch-*`, so a
mismatched arch raises on `load_state_dict`, and a mismatched `--batch`/`--accum` would
silently mis-index the saved `batch_idx` and replay the wrong batches.

```powershell
# 1. cache logits at the TRAINED height - --height defaults to 96 and MUST be 128 here
& .\.venv\Scripts\python.exe arabic_ocr/scripts/decode_ctc_lm.py `
  --ckpt arabic_ocr/runs/stage06a/ckpt_final.pt --height 128 --tag stage06a --cache `
  --sets muharaf_val,muharaf_test,khatt_teklia_test,isippt_bench,khatt_bench `
  --out arabic_ocr/runs/stage06a --cache-dir arabic_ocr/runs/stage06a/logits

# 2. tune the blank penalty on muharaf_val ONLY - never on a test set
& .\.venv\Scripts\python.exe arabic_ocr/scripts/blank_penalty_tune.py `
  --logits-dir arabic_ocr/runs/stage06a/logits --tags stage06a `
  --ckpt arabic_ocr/runs/stage06a/ckpt_final.pt `
  --out arabic_ocr/runs/stage06a/blank_penalty_tune.json

# 3. re-decode with delta* and report raw AND norm_ar per set, then read the gates
& .\.venv\Scripts\python.exe arabic_ocr/scripts/decode_ctc_lm.py `
  --ckpt arabic_ocr/runs/stage06a/ckpt_final.pt --height 128 --tag stage06a --run `
  --blank-penalty <delta*> --out arabic_ocr/runs/stage06a
```

Gates: **G1** khatt_test raw <= 21.00 · **G2** muharaf_test raw <= 23.11 · **G3** khatt_bench
<= 29.54 and isippt <= 58.34. Compare against the best adapted numbers so far: B2+delta1.00
22.57 / 22.88 / 53.63 / 29.11 and ckpt-11700+delta0.75 23.20 / **21.82** / 55.67 / 28.47.

Integrity gates already added this run: `check_ckpt_reload.py` (the checkpoint must reload
through `load_ctc_backbone` at its stored height) and the `height` stamp in the logits cache
(a cache whose recorded height differs from `--height` is recomputed, not trusted).

## (y) 2026-10-07 -- Steps 2 and 4 report: the resolution knob is dead at inference, the bottleneck is capacity

**Step 2 FAIL (E-102).** The backbone *looks* height-agnostic (height is attention-pooled,
`output_lengths` depends only on width) but it is not. Re-decoding the SAME teacher weights
at a taller input destroys it:

| input height | muharaf_test raw | emission | khatt_test raw | isippt |
|---|---|---|---|---|
| 96 (trained) | **22.11** | 86.7% | **40.79** | **62.05** |
| 128 | 34.91 | 73.6% | 55.14 | 86.95 |
| 160 | 60.54 | 45.9% | 74.13 | 93.70 |
| 192 | 85.27 | 17.1% | 87.27 | 93.76 |

All three pre-registered predictions failed, and P2 failed *backwards* - I predicted KHATT
would gain 1.55x of recovered resolution, and it lost 14.35 pp. BatchNorm running stats, the
learned attention distribution over H and the width-only positional encoding are all bound to
the H=96 training geometry. **The resolution information is real (E-100) but harvesting it
requires TRAINING at the higher height - it is a step-3 variant, not a free inference knob.**

**Step 4 RESOLVES to underfit, not bad labels (E-103).** `trainline_emission_probe.py` (new,
and it carries a self-check that must reproduce the cached-logit path - it does, 22.06 vs
22.11) measures emission on the model's own training split:

| model | split | emit (raw ref) | CER vs the label it was trained on |
|---|---|---|---|
| teacher | muharaf_train | 91.5% | 12.02 |
| teacher | muharaf_test | 86.7% | 16.95 |
| stage05f (ours) | muharaf_train | 91.3% | 13.37 |
| stage05f (ours) | muharaf_test | 86.7% | 18.56 |

The model under-emits on data it was trained on, and the train/test gap is only ~5 pp. That is
the signature of an **underfit, capacity-limited** model - not of label noise (which would also
show low train emission) and not of overfitting (which would show high train fit). The 4.2M
model cannot fit its own training labels.

**Consequence for step 3.** Steps 1, 2 and 4 are now closed: the decoder knob is spent (142),
the resolution knob is dead at inference (143), and the data is not the culprit (144). The one
lever still standing is capacity - exactly the user-deferred step 3.

Probes: `arabic_ocr/scripts/trainline_emission_probe.py`, `image_size_probe.py`.

## (x) 2026-10-07 -- Step 1 shipped + the resolution question is domain-split (E-100/E-101)

**Blank penalty shipped (TASKS 142).** `--blank-penalty` is now wired into
`decode_ctc_lm.py` (applied to the cached logits before greedy/beam, recorded in the
scorecard) and `blank_penalty_tune.py` is generalised (`--logits-dir/--tags/--deltas`,
reports raw + norm_ar + emission%). Re-scored B2 (stage05f), delta* = 1.00 tuned on
`muharaf_val` only:

| model (raw CER) | muharaf_test | khatt_test | isippt | khatt_bench |
|---|---|---|---|---|
| teacher, delta 0 | 22.11 | 40.79 | 62.05 | 41.75 |
| teacher, delta* 1.00 | **21.47** | 41.07 | 59.06 | 41.85 |
| ckpt-11700, delta* 0.75 | 23.20 | **21.82** | 55.67 | 28.47 |
| **stage05f (B2), delta* 1.00** | **22.57** | **22.88** | **53.63** | **29.11** |

B2+penalty passes G2 (22.57 <= 23.11) and G3 (29.11 <= 29.54, 53.63 <= 58.34) and
**fails only G1** (khatt_test <= 21.00). Best available for G1 is ckpt-11700+penalty
at 21.82 - a 0.82 pp miss. Emission rises everywhere with the penalty (muharaf_test
87.0 -> 90.8%, isippt 60.0 -> 70.5%), which is the mechanism E-97 predicted.

**Resolution is domain-split (E-100, `image_size_probe.py`).** The Muharaf source line
images are **60 px tall for 100% of sampled lines** (width 600), so decoding at height
96 already UPSCALES 1.6x - there is no resolution to recover at home. ISIPPT is 27 px.
But `khatt_teklia_test` is 149 px and `khatt_bench` 146 px, so **98%/93% of KHATT lines
are DOWNSCALED 1.55x** at height 96. The acuity limit is therefore a source-data limit
for Muharaf and a genuine preprocessing loss for KHATT.

That is why step 2 targets G1: the only failing gate is a KHATT gate, and KHATT is the
only domain with real resolution to win back. Predictions for the height sweep are
pre-registered in TASKS 143 (P1 muharaf flat +/-0.30 pp, P2 KHATT >= 1.00 pp better,
P3 isippt flat +/-0.50 pp) - recorded BEFORE the run.

Probes: `arabic_ocr/scripts/image_size_probe.py`, `blank_penalty_tune.py`.

## (w) 2026-10-07 -- Error attribution: 29% of our headline CER is evaluation convention (E-99)

Before choosing data / model / strategy, measured how much of raw CER 22.11 is even
model error. Teacher, cached logits, CPU only:

| set | raw CER | norm_ar (published convention) | emit-able ref chars only |
|---|---|---|---|
| muharaf_test | 22.11 | **15.79 (-6.32)** | 21.03 (-1.08) |
| muharaf_val | 21.48 | 17.23 (-4.25) | 20.34 (-1.14) |
| khatt_teklia_test | 40.79 | 38.69 (-2.11) | 40.59 (-0.20) |

- **~6.3 pp of the 22.11 headline (29%) is evaluation convention**, not model error:
  ~1.08 pp is reference characters the 165-symbol charset cannot emit (diacritics are
  dropped 86-89%), the rest is alef/hamza/yeh normalization. Published cards report the
  normalized number, so our line has been compared ~6 pp unfairly against them.
- **The dot-loss hypothesis is REFUTED**: dot-bearing vs dotless letters are dropped at
  9.4% vs 8.5% (muharaf_test) and 23.5% vs 20.7% (KHATT). The real acuity signature is
  glyph HEIGHT/PROFILE - tooth rows and small loops (th 16%, s 15%, b 13%, t 12%, h 12%,
  m 12%, "." 14%) drop far more than tall/distinctive strokes (t. 5%, dh 5%, z 5%,
  ta-marbuta 5%, w 6%, a 6%, l 6%). That points at input resolution, not dot rendering.
- Data-side contribution is small (~1.1 pp) but has two concrete defects: train labels
  were diacritic-stripped while test references keep diacritics, and the union is 82/18
  Muharaf/KHATT with only 4,671 KHATT train lines.
- Probe: `arabic_ocr/scripts/error_attribution.py`.

## (v) 2026-10-07 -- REVISION of (u): the error mode is character UNDER-EMISSION, and a CTC blank penalty is a free gain (E-97, E-98)

`(u)` declared Plan A exhausted. Two CPU probes revise that: the DECODER lever was not
fully exhausted - one inference knob had never been tested.

**E-97 - what kind of errors are these?** Levenshtein backtrace over the cached teacher
logits, muharaf_test: 15,420 edits = **10,066 deletions (65%)** + 4,534 substitutions
(29%) + 820 insertions (5%). The model emits only **86.7%** of reference characters
(KHATT test 80.8%, ISIPPT 52.7%) and drops **29%** of word spaces (whitespace-free CER
19.64 vs 22.11 raw). Two obvious causes are REFUTED: only 1.44% of reference characters
are outside the 165-symbol charset (worth 1.08 pp), and there are 4.3 output frames per
character (ISIPPT has 9 and is the worst). Under-emission scales with domain distance, so
it is a representation failure surfacing as CTC blank bias - which is exactly why the
decoder (E-94), the lexicon (E-96) and adaptation (E-95) all failed downstream of it.

**E-98 - the blank penalty is a real lever (delta tuned on muharaf_val ONLY).**

| model | delta* | muharaf_test | khatt_teklia_test | isippt_bench | khatt_bench |
|---|---|---|---|---|---|
| teacher | 1.00 | 22.11 -> **21.47** | 40.79 -> 41.07 | 62.05 -> **59.06** | 41.75 -> 41.85 |
| ckpt-11700 | 0.75 | 23.71 -> **23.20** | 22.05 -> **21.82** | 57.84 -> **55.67** | 29.04 -> **28.47** |
| trained B2 (no penalty) | - | 23.01 | 22.91 | 56.03 | 29.74 |

- **ckpt-11700 + blank penalty is the best union model so far and beats the trained B2
  rung on 3 of 4 sets at ZERO training cost**; it makes G3 pass (55.67 <= 58.34 and
  28.47 <= 29.54). G2 (Muharaf <= 23.11) is now missed by only 0.09 pp.
- The teacher val-tuned penalty HURTS KHATT (+0.27): the optimum does not transfer
  across domains, so tune it per target domain.
- The acoustic model is still the binding constraint - G1 (KHATT <= 21.00) stays
  unreachable by any decoder - so the 15M question stands. TASKS 142 must land first:
  wire `--blank-penalty` into `decode_ctc_lm.py`, cache stage05f logits, re-score B2.
- Probes: `char_confusion_probe.py`, `charset_coverage_probe.py`, `deletion_probe.py`,
  `blank_penalty_probe.py`, `blank_penalty_tune.py`.

## (u) 2026-10-07 -- Stage 4b lexicon probe: the word-lexicon lever is DEAD in its cheap form (TASKS 140)

Measured BEFORE building (E-94's lesson). Lexicon = 54,084 types / 261,471 tokens from
the Muharaf + KHATT TRAIN labels; greedy decode of the cached teacher logits for
`muharaf_test` reproduced the harness 22.11 exactly (self-check).

| quantity | value |
|---|---|
| reference words OOV in the lexicon | **20.14%** (2,443/12,133) |
| hypothesis words OOV | **45.07%** (5,146/11,417) |
| of the 4,000 most frequent distinct OOV words, share with a distance-1 lexicon neighbour | 75.3% (82.8% of OOV tokens covered) |
| CER after replacing each OOV word by that neighbour | **23.94 (+1.83 pp WORSE)** |

- The rule is net-negative: a train-only lexicon misses 20% of the reference
  vocabulary, so "OOV" is a weak error signal - most OOV predictions are not
  one-character corruptions, and forcing them to the nearest lexicon word adds errors.
- A real word-level decoder would need a much larger, cleaner lexicon plus a word
  n-gram LM. `arabic_ocr/scripts/lexicon_probe.py` reproduces the measurement.
- **Plan A (the 4.25M Muharaf line) is now exhausted**: decoder (E-94, oracle-bounded
  at <= 0.53 pp), adaptation (E-95, capacity-bound Pareto frontier) and lexicon
  post-correction (E-96, net-negative) are all measured. The next real gain needs
  Plan B (a 15M specialist) or a much larger text corpus for a genuine word LM.

## (t) 2026-10-07 -- Stage 0.5f "repair rung" (B2) RESULT: G2 recovered, G1/G3 still fail (TASKS 141)

Warm-starting from ckpt-11700 with a near-zero-LR union pass (lr 3e-6, 2 epochs,
6,624 steps, 1,406 s) moves the model ALONG the Muharaf/KHATT trade-off instead of
pushing it out. Raw CER on the full sets:

| set | teacher | ckpt-11700 (B1') | B2 | B2 gate |
|---|---|---|---|---|
| muharaf_test | 22.11 | 23.71 | **23.01** | G2 <= 23.11 **PASS** |
| khatt_teklia_test | 40.79 | 22.05 | 22.91 | G1 <= 21.00 **FAIL** |
| isippt_bench | 62.05 | 57.84 | 56.03 | G3 ok (improved) |
| khatt_bench | 41.75 | 29.04 | 29.74 | G3 <= 29.54 **FAIL** (+0.70) |

- The 0.70 pp home-set recovery cost 0.86 pp on KHATT and 0.70 pp on KHATT bench: the
  pass re-balanced the model rather than improving it. B1' sits at (23.71, 22.05) and
  B2 at (23.01, 22.91); no point on that line passes both gates.
- **Interpretation**: for this 4.25M backbone the union objective is capacity-bound,
  not optimisation-bound, so further LR/epoch tuning is not worth running.
- Both Plan-A levers (decoder, adaptation) are now exhausted with measured
  negative/partial results. The next real gain needs Plan B (15M KHATT specialist,
  user-deferred) or a decoder that changes the SEARCH SPACE (TASKS 140).
- Artifacts: `arabic_ocr/runs/stage05f/{ckpt_last.pt, ckpt_final.pt, scorecard.json,
  scorecard_eval.json, train_log.jsonl}` (weights local-only).

## (s) 2026-10-07 -- Stage 4 decoder RESULT: FAIL, and now oracle-bounded (TASKS 139 -> 140)

Beam search + char-LM rescoring does **not** beat greedy on this model, and the
failure is bounded rather than merely observed:

| muharaf_test raw CER | greedy | beam | + char LM |
|---|---|---|---|
| summed-probability beam (logsumexp), beam 12 | 22.11 | 24.58 | 23.72 |
| Viterbi (max) beam, beam 32 | 22.11 | 22.11 | 22.11 |

- The Viterbi beam equals greedy **exactly** at beam 1 (21.48 on `muharaf_val`), so
  the best-path ranking is preserved; the summed-probability ranking is *worse* than
  greedy because this CTC model is peaky - prefix probability favours hypotheses the
  best path rejects. The length bonus is not the cause (beam 32 keeps hypothesis
  length at 49.4 vs the 52.7-char reference; a bonus of 1.0 only inflates it to 65.7
  and doubles the error).
- **Oracle CER** (best candidate per line, i.e. a perfect reranker): Viterbi beam 32
  = **21.92** test / **21.27** val; summed beam 32 = **20.98** val; beam 64 = **20.95**
  val. Against greedy 22.11 / 21.48 that is **0.19-0.53 pp** of headroom, and
  doubling the beam buys 0.03 pp.
- Consequence: **G1 (<= 20.61 raw) is unreachable by any reranker over this N-best
  family**, and G2 passes only because the Viterbi decoder cannot lose to greedy.
  Both gates FAIL. The card's published 16.95 -> 13.31 gain is therefore *not*
  reproducible by N-best rescoring; it requires a search space that generates new
  hypotheses (word-constrained/lexicon decoding or a seq2seq rescorer) - TASKS 140.
- The char LM (3.5 MB GRU, val ppl 4.29 on 7.87M train-only chars) contributed at
  most -0.02 pp. The external pool is leakage-free (<= 0.15% verbatim overlap,
  `audit_textpool_leakage.py`) but the oracle bound says a stronger LM cannot matter
  for this decoder.

Artifacts: `arabic_ocr/runs/stage05e/scorecard_teacher.json`,
`adapt/scorecard_adapt11700.json`, `probe_*` sweeps, `nbest/nbest_*.jsonl` (gitignored);
logit caches under `logits/` (gitignored).

## (r) 2026-10-06 -- Stage 4 decoder (TASKS 139) STARTED: beam + char-LM rescoring

Training-free CER, pre-registered in TASKS 139 before any decode. Rationale: the
teacher card shows its own greedy 16.95 -> 13.31 with a decoder/lexicon (~21%
relative) and this applies to every checkpoint we own, including the Stage 0.5d
artifact.

Built and running now:

- `arabic_ocr/scripts/char_lm_common.py` -- char LM + charset mapping (the LM lives
  in the 165-char acoustic charset, so its scores are directly comparable to CTC).
- `arabic_ocr/scripts/train_char_lm.py` -- GRU char LM over TRAIN-ONLY text
  (muharaf train + khatt teklia train + hflines train). `--pool` adds
  `data/ocr_textpool/texts.jsonl` but is opt-in and needs an overlap audit first.
- `arabic_ocr/scripts/decode_ctc_lm.py` -- `--cache` stores fp16 log-softmax logits
  per line (46 s for all 4,141 dev+test lines) so the sweeps never touch the GPU;
  `--run` does CTC prefix beam search (beam 12, top-k 6, N-best 12) then rescores
  with `score = acoustic + beta*lm + gamma*len`, tuning on `muharaf_val` only.

Gates: **G1** `muharaf_test` raw CER <= 20.61 (>= 1.5 pp better than the 22.11
greedy teacher); **G2** no eval set worse than its own greedy baseline by more than
+0.50 pp; **G3** the ckpt-11700 artifact decoded with the same decoder, reported
only. Cached logits: `arabic_ocr/runs/stage05e/logits/teacher/`.

Decoder bugs caught by the first beta=0 smoke pass, before any gate was read:
greedy-from-cache must call `greedy_ctc_decode` (a bare argmax over cached logits
scores 44.55 on `muharaf_test` where the verified harness scores 22.11),
`beam_search` must return mutable candidate records because the LM score is
appended in place, and the greedy hypothesis is now always injected into the
N-best so the decoder cannot lose the greedy answer by construction.

Leakage audit before using the external pool (`data/ocr_textpool/texts.jsonl`,
10,400 distinct texts / 5.26M normalised chars): verbatim line overlap with the
eval splits is 0/1,069 Muharaf validation, **2/1,334** Muharaf test, 0/1,038 KHATT
teklia test, 1/500 ISIPPT, 0/200 KHATT bench (<= 0.15%), so `--pool` is safe as an
external corpus. `arabic_ocr/scripts/audit_textpool_leakage.py` reproduces it.

## (q) 2026-10-06 -- Stage 0.5d (B1') RESULT: both pre-registered gates FAIL

The (p) run trained but was **cut by the session boundary at step 12,250 of
13,437 (91%)**, before it could write its own final scorecard; the step-11,700
checkpoint (the last in-loop eval) was scored through the trainer's new
`--eval-only` path. In-loop free-running greedy raw CER on the two real test
splits:

| step | 900 | 2700 | 4500 | 7200 | 9900 | 11700 |
|---|---|---|---|---|---|---|
| KHATT teklia test | 30.90 | 27.67 | 24.91 | 22.85 | 22.20 | **22.05** |
| Muharaf test | 23.89 | 23.96 | 23.65 | 24.10 | 23.91 | **23.71** |

**G1 FAIL** -- KHATT teklia test raw **22.05** vs the pre-registered <= 20.00
(normalized 20.7, median 15.94). Still a large real gain:
-18.74 pp from the 40.79 zero-shot baseline, a 46% relative reduction.
**G2 FAIL** -- Muharaf test raw **23.71** vs the pre-registered <= 23.11
(normalized 18.54), i.e. +1.60 pp of in-domain forgetting, above
the 1 pp allowance. The same eval-only scorecard was also run against the ZERO-SHOT
teacher, so every number below has a measured baseline:

| set | zero-shot teacher | adapted ckpt-11700 | delta |
|---|---|---|---|
| muharaf_test | 22.11 | 23.71 | +1.60 |
| khatt_teklia_test | 40.79 | 22.05 | -18.74 |
| isippt_bench | 62.05 | 57.84 | -4.21 |
| khatt_bench | 41.75 | 29.04 | -12.71 |

The other three sets all IMPROVE -- ISIPPT -4.21 pp, KHATT bench -12.71 pp -- so the
union recipe helped printed text as well as KHATT; only Muharaf test regressed.
FixFips is still the better printed-text model (46.92 ISIPPT, 12.03 KHATT bench), so the
honest reading is: our single 4.25M model now beats its teacher on **3 of the 4 sets**
and loses 1.60 pp only on the teacher home set. Full data:
`runs/stage05c/eval_muharaf_teacher_all4.json` vs
`arabic_ocr/runs/stage05d/scorecard_eval.json`.

The KHATT trajectory was still creeping down (22.20 -> 22.05 over the last
1,800 steps) while the cosine lr approached zero, so finishing the remaining 9%
would not have closed a 2 pp gap; the run was not resumed.

**Verdict (E-93): the union recipe beats the teacher on 3 of the 4 sets, but buys that
with 1.60 pp on the teacher home set** -- Muharaf 23.71 (vs the 22.11 teacher), KHATT
22.05 (vs 40.79), ISIPPT 57.84 (vs 62.05), KHATT bench 29.04 (vs 41.75) -- and it is
still far from the in-domain specialists (FixFips 10.99 KHATT, 46.92 ISIPPT). The
cross-domain gain is real and costs ~35 min of GPU, but the teacher stays the
best Muharaf model and FixFips the best KHATT model. The artifact is
`arabic_ocr/runs/stage05d/ckpt_last.pt` (weights local-only). Next: the Stage 4
decoder (TASKS 123), which is training-free and was worth 16.95 -> 13.31 on the
teacher's own card.

## (p) 2026-10-06 -- Stage 0.5d (B1') pre-registration + Step 0 on the REAL held-out splits

**Step 0 (E-92) is measured** -- both teachers, zero-shot, on genuinely held-out
splits, our E-89 raw protocol (greedy CTC, no LM, `sum(edit_distance)/sum(len(ref))`):

| model | params | Muharaf test (1,334) raw / norm | KHATT teklia test (1,038) raw / norm |
|---|---|---|---|
| `sdkv2/muharaf-arabic-ocr` CNN-Transformer-CTC | 4,245,671 | **22.11** / 16.83 | 40.79 / 39.35 |
| `FixFips/arabicocr-khatt` CRNN | 15,060,427 | 53.09 / 48.99 | **10.99** / 9.77 |
| published card | -- | 16.95 (normalized) | 6.09 (normalized, KHATT *val*) |

Muharaf detail: median 16.67, `cer_lt10` 24.6%, exact 5.4%. FixFips on KHATT:
median 3.33, `cer_lt10` 82.2%, exact 16.09%. Both directions of cross-domain
transfer are weak (Muharaf->KHATT 40.79 vs FixFips->Muharaf 53.09), so the two
specialists are real and correctly specialised.

**The card numbers are NORMALIZED CER; ours is RAW -- that is the entire ~5 pp
gap.** Muharaf's card `metrics_test_greedy.cer = 0.16954` against our 16.83
normalized (matches to 0.12 pp) and our 22.11 raw. Their `text_raw`/`text_norm`
and our `labels.json` transcripts are byte-identical on every row checked, so
this is a metric convention, not a data or checkpoint difference. Quote any
third-party card as "normalized; ~5 pp lower than our raw measurement".

**Step 1 as originally worded was unreachable and is corrected here.** The
backbone's own `training_metrics.json` (20 epochs, lr 5e-5, effective batch 8,
train 22,091) shows it was fine-tuned on exactly `muharaf_train.csv`, so
fine-tuning it *on Muharaf train again* cannot beat its own 22.11. The measurable
gap is cross-domain, so (B1') trains on Muharaf train + KHATT teklia train and is
scored on both real test splits. FAIL remains a reportable outcome.

**Pre-registered gates (decided before the run):** G1 KHATT teklia test raw CER
<= 20.00 (from 40.79, the point of the run); G2 Muharaf test raw CER <= 23.11
(from 22.11, no more than 1 pp of forgetting); G3 printed ISIPPT/KHATT benches
reported, not a gate on this rung.

**Recipe:** warm-start `models/ctc_backbone/model.pt` (sha256 head
`bba3b6f5301822fc`, 4,245,671 params), 3 epochs, AdamW lr 3e-5 cosine, wd 1e-4,
clip 5.0, batch 8 width-bucketed (size-1 tail batches dropped for BatchNorm),
fp32, no augmentation, image height 96 (identical to the inference path), CTC
blank 0, no LM/reranker. **Label repair (found at launch, before any gate was
read):** the checkpoint charset is 165 characters and contains NO diacritics
(verified), so a diacritized label is charset-OOV -- and dropping such lines
discarded 7,472 of the 22,091 Muharaf train lines (34%). The trainer now strips
U+064B-U+065F/U+0670 and keeps the line, recovering 21,819 Muharaf (oov=0,
too_short=272) + 14,013 KHATT = 35,832 items, 4,479 steps/epoch. The first
launch (`pwsh-10`) used the dropping version and was killed at step 600
(6.4 steps/s, loss 1.02); no gate was read from it. `data/reallines` stays
EXCLUDED on this rung -- it is outside the Muharaf/KHATT pair under test.
Trainer `arabic_ocr/scripts/finetune_muharaf.py`; artifacts under
`arabic_ocr/runs/stage05d/` (weights local-only via the new .gitignore rule,
JSON/logs tracked). Step 2 afterwards = the Stage 4 decoder (TASKS 123).

## (o) 2026-10-06 -- Stage 0.5c (b): the SECOND in-domain teacher eval, and the contamination audit that revises E-89

**TASKS 134 / E-90 -- `sdkv2/muharaf-arabic-ocr` zero-shot (inference only).**
New harness `arabic_ocr/scripts/eval_muharaf_hflines.py` mirrors E-89 exactly
(greedy CTC over the model's own logits, NO language model, no reranker,
free-running, dynamic width) and scores the author's `ctc_backbone/model.pt`
(4,245,671 params, sha256 head `bba3b6f5301822fc`) on all 1,654 lines of
`data/hflines/val`: **CER 23.64% vs the 57.80% control (+34.16 pp)**, median line
CER 13.16%, cer_lt10/20/50 = 38.4 / 66.5 / 91.2%, 11.55% exact lines, 17.3 s /
95.3 img/s. It looked like a rout.

**E-91 -- it is NOT a transfer number, and measuring that is the result.**
`data/hflines` was built from `Teklia/KHATT_v1.0` + `aamijar/muharaf-public` +
`data/reallines` (`build_hf_lines.py`), and this teacher was trained on Muharaf.
Matching normalized transcripts against the author's OWN published training
manifests (22,091 lines) shows **79.9% of the val lines are in its training set**,
where it decodes at 15.69% CER; the 3.4% also in its val/test splits decode at
20.24%, so only **16.7-20.1% is genuinely held out**.

**The same audit re-opens E-89.** FixFips trained on KHATT, and 9.6% of the val
lines are KHATT lines it trained on (CER **1.82%** there). On the **174 val lines
held out from BOTH candidate teachers** -- the only directly comparable number --
it scores **61.88%, i.e. 4.08 pp WORSE than the control**; Muharaf scores
**54.06% (+3.74 pp)**. The sample is small (n=174) but the medians agree
(66.67 vs 50.45) and the ranking is stable on the strict 100-line variant that
also excludes both teachers' val splits (75.26 vs 68.80).

| teacher | headline | in own train | own held-out | shared clean (174) |
|---|---|---|---|---|
| `FixFips/arabicocr-khatt` | 46.93 | 9.6% / CER 1.82 | 52.79 | **61.88** |
| `sdkv2/muharaf-arabic-ocr` | 23.64 | 79.9% / CER 15.69 | 46.30 | **54.06** |
| control (un-mirrored zero-shot) | 57.80 | -- | -- | 57.80 |

**Consequence.** E-89's gate literally passed and its measurement stands, but its
interpretation (route (D) transfers) was **overstated**: route (D) transfers
*inside a teacher's own training domain*, not to unseen handwriting. Muharaf is
still the better base (54.06 vs 61.88, 4.25M vs 15.06M params, 95 vs 51 img/s),
but the honest headroom over the incumbent control is ~3.7 pp, so the <5% target
is NOT reachable by distilling either teacher. TASKS 137 supersedes the TASKS 131
decision and is USER-GATED. Evidence: `runs/stage05c/{teacher_overlap_stratification,teacher_clean_comparison,overlap_muharaf_hflines}.json`.

## 2026-10-06 (n) - OCR Stage 0.5c / E-89: the in-domain teacher TRANSFERS zero-shot (+10.87 pp) - route (D) supported, but it is a weak teacher

`FixFips/arabicocr-khatt` (CRNN = CNN+BiLSTM+CTC, 15.06M params, KHATT 11,375
lines, MIT) was evaluated ZERO-SHOT - no fine-tuning at all, greedy CTC, no
language model - on all 1,654 lines of `data/hflines/val`, using the author's own
preprocessing (grayscale -> binarize -> h=96 -> pad W=1536) and the author's
reversed-label protocol.

| metric | FixFips | control (v6-small) |
|---|---|---|
| CER (raw, summed) | **46.93** | 57.80 |
| CER (author-normalised) | 42.53 | - |
| WER | 81.89 (norm 80.56) | - |
| exact line | 4.66% | - |
| median line CER | **51.02%** | - |
| cer_lt10 / lt20 / lt50 | 11.2 / 14.9 / 48.1% | - |
| speed / params | 51.1 img/s (32.4 s) | 15.06M vs 3.10M |

**GATE PASS: +10.87 pp better than the 57.80 control.** Side-checks: KHATT 200 =
**12.03** vs control 44.41 (+32.38, but KHATT is this model's own training corpus
- NOT independent transfer evidence); printed ISIPPT 500 = **46.92** vs 13.75
(-33.17, expected out-of-domain). Greedy/no-LM 12.03 on KHATT against the card's
6.09 (beam + bigram LM) is the right order of magnitude, which validates the
architecture reconstruction, the checkpoint load and the preprocessing. The
reversed-label protocol was confirmed empirically (flip 46.93 vs no-flip 81.99).

**Honest reading: a real but WEAK win.** Only ~15% of lines decode under 20%
CER and half are worse than 50%. This is the best zero-shot base found so far,
not a usable recognizer. Fine-tuning/distilling from here must still budget far
more than the 2k lines E-77 showed to be harmful, and per E-88 use a much lower
lr with the backbone frozen. **Nothing was trained: TASKS 133 is inference-only,
and the next training step remains USER-GATED at TASKS 131.**

Environment note: the author's `preprocess.py` needs cv2, which `.venv` does not
have; the harness borrows the extension from the sibling `.venv-paddle`
site-packages via `sys.path` instead of installing anything. A real fix is a
user-gated `pip install opencv-python`. The harness and the author's unmodified
code live under the gitignored `arabic_ocr/models/fixfips_khatt/`
(`author_code/` = the author's own `model.py` / `preprocess.py` / `metrics.py` /
`eval_val.py`, re-fetchable from github.com/FixFips/ArabicOCR_KHATT; that GitHub
repo ships no LICENSE file, so it is cached locally rather than vendored into
git).

## 2026-10-04 (m) - OCR Stage 2: stride 1 FAILED at 59.86%; CenterNet-style running

Run A (stride-1 CRAFT, `pwsh-6`) **FAILED the gate at 59.86%** (epoch 29, P 67.62 /
R 53.70) - +2.52 over craft3's stride-2 57.34%, still far under 90%, and still
climbing at the last epoch. Three levers (capacity, dead head, resolution) have
now each bought 2-3 points; the residual is the optimization, not the architecture.

Run B (CenterNet-style, `pwsh-7`, `runs/stage2_paw_cn/`) is training. Its
perfect-target ceiling is **98.96%** and its decode has neither a level set nor a
component, so neither measured CRAFT failure mode can occur.

| run | design | best H | ceiling | verdict |
|---|---|---|---|---|
| `runs/stage2_paw_craft2` | CRAFT stride 2, 2.56M | 54.89% | 92.27% | FAIL |
| `runs/stage2_paw_craft3` | CRAFT stride 2, 4.94M | 57.34% | 92.27% | FAIL |
| `runs/stage2_paw_s1` | CRAFT **stride 1**, 5.01M | 59.86% | 96.91% | FAIL |
| `runs/stage2_paw_cn` | CenterNet stride 2, 4.94M | 46.97% | **98.96%** | FAIL |

## 2026-10-04 (l) - OCR Stage 2: both representation levers built and running

Stage 2's gate (PAW Hmean >= 90% on held-out synthetic) has failed three times on
the CRAFT stride-2 representation (54.89% -> 57.34%). The cause is structural,
not capacity: the perfect-map ceiling at stride 2 is 92.27%, and at the decode
thresholds where merging is controllable it is below the gate.

Two replacements are implemented and running sequentially (single GPU):

| run | design | perfect-map ceiling | job |
|---|---|---|---|
| `runs/stage2_paw_s1` | CRAFT region score at **stride 1** (sigma 1.7 -> 3.4 px) | 96.91% (k=0.15) | `pwsh-4` |
| `runs/stage2_paw_cn` | **CenterNet-style** centre heatmap + w/h regression | **98.96%** | next |

Both ceilings clear the 90% gate, so the open question is how much of the
57 -> 90 gap each recovers, not whether the gate is reachable.

## TL;DR

Every CER this project reported before 2026-10-01 was **teacher-forced** and does
not measure OCR. Under honest free-running evaluation the best checkpoint emits
the **same string for every image** and gets **0% of words right**.

Targets were: CER < 5%, model < 100 MB, inference > 50 img/s.

| Target | Result | Verdict |
|---|---|---|
| CER < 5% | ISIPPT 144.8%, KHATT 97.8% (free-running) | **FAIL** |
| Model < 100 MB | 15.7 MB fp16 | **PASS** |
| > 50 img/s | 33.7 / 38.0 img/s (48x256, batch 32) | **FAIL (close)** |

## The measurement bug

`arabic_ocr/scripts/benchmark.py:46` does:

```python
output = model(images, text=texts)   # ground-truth text passed in
predictions = decode_greedy(logits, idx_to_char)
```

`CompactViTDecoder.forward` with `text` sets `input_tokens = text[:, :-1]`, so at
every decoding step the decoder is handed the ground-truth prefix. A character
language model conditioned on the correct prefix scores well while never reading
the image. The reported numbers were therefore a **language-model score, not OCR**.

`Trainer.evaluate()` had the same flaw, so the `best_cer` (0.19-0.93) stored in
every checkpoint in `runs/` is teacher-forced too.

## Evidence (same checkpoint, `runs/vit_real/best.pt`, 7.85M params)

| Evaluation | ISIPPT CER | KHATT CER | KHATT WER |
|---|---|---|---|
| Teacher-forced (`benchmark.py`) | 34.4% | 16.6% | 24.0% |
| **Free-running (`eval_benchmark.py`)** | **144.8%** | **97.8%** | **100.0%** |

Free-running output for *every* image in both benchmarks:

```
متبلصامتبلكوةبلكوةبلكوةبلكوةبلكوةبلكوةبلصر؟بلصر؟بلكوةبلكوةبلكوة
```

Identical output for different images, and WER 100.00%, means the decoder ignores
the encoder entirely. The historical (teacher-forced) numbers are preserved in
`runs/benchmark_results_TF_historical_backup.json`.

## Why the models do not read

1. **The seq2seq decoder takes a character-LM shortcut.** Given the true prefix,
   cross-attention is unnecessary, so gradient descent never makes it useful.
   Teacher forcing hid this completely.
2. **CTC was tried as the fix** (`arabic_ocr/src/models/crnn.py`, rewritten as a
   line-level CRNN+CTC). It has no prefix to condition on, so it cannot cheat --
   but it collapsed to the all-blank solution (99.7% blank frames) and could not
   overfit even 64 lines in 250 epochs.
3. **The loss and gradients are not the bug.** `F.ctc_loss` was verified
   separately on synthetic linearly-separable features (loss 12.8 -> 0.80, 100%
   frame accuracy in 300 steps), and gradients reach every module
   (conv0 0.199, conv10 1.36, lstm 2.97, head 6.72). The encoder simply does not
   produce input-dependent features.
4. **Every prior number is suspect**, including model-selection decisions.
   `runs/vit_real/best.pt` looked like the best model (best_cer 0.1887) and is in
   fact one of the worst -- it ignores the image completely.

## What was added/fixed on 2026-10-01

- `arabic_ocr/scripts/eval_benchmark.py` (new) -- free-running benchmark eval;
  reports CER/WER/params/size/img-per-s and writes `runs/benchmark_results.json`.
- `arabic_ocr/src/models/vit_decoder.py` -- added the missing causal mask, PAD
  loss weighting, and a correct `generate()`.
- `arabic_ocr/src/models/crnn.py` -- rewritten as a line-level CRNN+CTC
  (height collapsed to 1 row, width /8 => T=128 frames).
- `arabic_ocr/scripts/fetch_hf_lines.py`, `build_hf_lines.py` (new) -- built
  `data/hflines/{train,val}`: 25,921 / 1,654 real line images from
  `Teklia/KHATT_v1.0` + `aamijar/muharaf-public` + `data/reallines`, all text
  unique, with an md5 guard proving **zero benchmark overlap**.
- `arabic_ocr/configs/{hflines,hflines_v3,crnn_ctc}.yaml` (new).

## Path forward (honest, 2026-10-01 -- SUPERSEDED 2026-10-04, see the last section)

From-scratch training on 26k lines with a <25M model on one 8 GB Turing GPU will
not reach CER < 5%. Realistic options, in order of expected value:

1. **Fine-tune a pretrained Arabic OCR model** (TrOCR-style or an Arabic VLM).
   This is the only path with a credible shot at < 5%.
2. If training from scratch is required, fix the encoder first: the current
   encoder yields near input-independent features. Diagnose with the
   synthetic-separability probe before spending GPU hours.
3. **Never report a teacher-forced CER again.** Use
   `eval_benchmark.py` (free-running) for every claim.

Reproduce the honest numbers:

```powershell
& .\.venv\Scripts\python.exe arabic_ocr\scripts\eval_benchmark.py \
    --config arabic_ocr\configs\vit_real.yaml \
    --checkpoint runs/vit_real/best.pt --device cuda
```

## 2026-10-03 -- Path 1 tried: fine-tuning a pretrained Arabic OCR model -> FAIL

The one path this document called "the only path with a credible shot" was run.
Verdict: **it does not work, and the published base model is itself broken.**

### 1. The pretrained base cannot read Arabic (zero-shot, clean input)

`RayR1/trocr-base-arabic-handwritten` (537,475,072 params) on **rendered, clean,
Arial** Arabic lines -- no scan noise, no handwriting:

| True text | Base output |
|---|---|
| `في الحج` | `''` |
| `انني سقيم` | `'.'` |
| `في الحج` | `'فيهاه'` |
| `ذهب مظفر ضرغام ...` | `'ذهب مظ ضر بصح رؤ بنؤ'` |

This is a defect of the checkpoint, not of our data. The model card shows
lr 5.6e-06 over 15 epochs, and the author's own `training_log.txt` is dominated
by `Empty label for index N` lines. **This base is not usable and should not be
fine-tuned further.**

### 2. Fine-tuning it on 500 real lines did not rescue it

`arabic_ocr/scripts/finetune_trocr_v2.py`, 500 train / 100 val real lines from
`data/hflines`, 3 epochs, 189 steps, 73 min, fp16, bs 2 x accum 4, lr 5e-5,
images padded to square, targets UTF-8.

| Metric | Value |
|---|---|
| eval_loss (epoch 1/2/3) | 6.875 / 6.450 / 6.374 |
| **free-running CER** | **98.0%** |
| **free-running WER** | **107.1%** |
| empty predictions | **50 / 100** |
| distinct predictions | **17 / 100** |
| mean prediction length | 4.3 chars (target 53.9) |
| speed | 1.8 img/s |

Most frequent outputs: `''` x50, `في` x25, `١٥` x6, `جورج` x4. The model
learned a handful of high-frequency training tokens and still ignores the image
(WER > 100% = it deletes most of the target and inserts wrong words).
Full per-sample evidence: `runs/trocr_finetune_v2/free_running_eval.json`.

### 3. Two pipeline bugs found while doing this

- **Encoding**: `data/hflines/*/labels.json` is **UTF-8**, but the OCR scripts
  open it as **cp1256**. Every Arabic target becomes mojibake (`في الحج` ->
  `ظپظٹ ط§ظ„ط¬ط¬`), ~2x the characters, discarding the pretrained decoder's
  Arabic prior. Verified at byte level: `في` = `d9 81 d9 8a`.
  `finetune_trocr_v2.py` now reads `utf-8`; the other scripts still need fixing.
- **Aspect ratio**: `TrOCRProcessor` resizes to a fixed 384x384 square, so the
  raw line images (e.g. 1959x128) were squashed into an unreadable strip. Fixed
  with `pad_to_square()` -- necessary, and (see above) not sufficient.

### 4. What this means for the path forward

Fine-tuning is only as good as its base. Before spending GPU hours on any
pretrained Arabic OCR/VLM base, **validate the base zero-shot on clean rendered
Arabic** -- a 5-minute test that would have saved this whole run. The remaining
credible options are a genuinely validated Arabic OCR checkpoint, or a
vision-language model (Qwen2-VL / GOT-OCR class) rather than a TrOCR derivative.

## 2026-10-04 -- User-approved direction: two-stage detection + recognition (docs only; NOT started)

Path 1 (fine-tune a pretrained recognizer) was tried and failed on 2026-10-03.
The user proposed a different decomposition and approved it: *detect the
smallest region that contains characters, recognize that region, and combine
the pieces later into a bigger model if needed - target 20-50M params.*

**Accepted, with one correction: the unit is the PAW (Part of Arabic Word) /
connected component, not the character.** In cursive Arabic 2-6 letters share
one baseline stroke, and dots plus diacritics are detached components, so a
"one character" crop is not well defined without already knowing the word
(PAW word-spotting and VML-HD's sub-word annotations exist for this reason).
Recognizers should read a **sequence** over the detected region (CTC/NRTR),
never classify an isolated crop.

**Why this is the right structural fix:** a detector loss is per-pixel and
spatial. Unlike the seq2seq decoder it cannot condition on the ground-truth
prefix, so it cannot take the character-LM shortcut that produced 144.8% CER
and the identical string for every image (see "Why the models do not read").

**The 20-50M target is empirically right:** PP-OCRv6 (arXiv 2606.13108) is
"From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks".
Reference points: `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` (Apache-2.0)
= 81.27% strict Arabic line accuracy; `medyas/arabic_PP-OCRv6_small_rec`
= CER 10.2% real printed / 13.5% ISIPPT held-out, WER 47.3%, from 800k
synthetic lines over 26 base + 12 display fonts.

The staged ladder, gates, disk budget and risks are in
`docs/plans/2026-09-29-arabic-ocr-nano-model.md` (V2 section); the live
task rows are TASKS.md 119-125; the Stage 0 bars are pre-registered as E-76 in
`research/EXPERIMENTS.md`.

**Status: nothing started.** The user asked for documentation only and gives the
go per stage. Stage 0 is a ~1 h zero-shot bake-off of the two PaddleOCR Arabic
recognizers on `data/hflines/val` with the free-running harness - it
needs the 2.15 GB duplicate checkpoint reclaimed first (E: has 8.0 GB free).

### Corrections to the sections above

- The 2026-10-01 "Path forward" list is superseded by this section: option 1
  (fine-tune a pretrained Arabic OCR model) was tried and failed on 2026-10-03.
- The two pipeline bugs found on 2026-10-03 (`labels.json` is UTF-8, not
  CP1256; `TrOCRProcessor` squashes to a 384x384 square) were fixed in 7
  scripts; the `diag_encoding.py` / `test_encoding_fix.py` hits are
  intentional (they test both encodings).

## 2026-10-04 (b) -- Disk cleanup: what this plan needs, and the harvested text pool

Another agent audited the disk (TIER 0-5, ~21 GB) and the user asked which of it
this plan still needs **before** anything is deleted. Review, per tier:

- **TIER 0** (`runs/trocr_finetune_v2/checkpoint-189/`, 2051 MB): the audit called
  it "provably lossless" because `model.safetensors` is SHA256-identical to the
  top-level file. That was incomplete -- the checkpoint also held this run's
  **only** `trainer_state.json`. Copied out first to
  `runs/trocr_finetune_v2/trainer_state.json` (4,574 bytes, SHA256 verified),
  so the directory is now safe to delete (TASKS row 124).
- **TIER 1** (`runs/trocr_finetune/`, 3138 MB): delete `checkpoint-189/`
  (2050 MB model + the 1086.71 MB `optimizer.pt` left by the crashed v1 write),
  not the parent -- the parent's small logs are the machine evidence for the
  cp1256 and padding findings.
- **TIER 2** (6.4 GB): not needed. Stage 0's control arm is the recorded number,
  and its evidence (`runs/vit_real/best.pt`, `runs/benchmark_results.json`,
  `runs/benchmark_results_TF_historical_backup.json`) is present and outside every
  listed directory.
- **TIER 3** (2.2 GB): not this line; all the `final/` directories exist. Guard
  `runs/pilot/final` (frozen KD teacher) and `runs/u1_fused/final` explicitly.
- **TIER 4** (5.2 GB): the images are worthless, the labels are not -- harvested
  first, see below. `data/hflines` and `data/reallines` stay untouched.
- **TIER 5** (1.9 GB, laya `B_last`): not this line; its owner must confirm.

**Harvested (CPU only, nothing trained, nothing deleted).** New
`arabic_ocr/scripts/harvest_text_pool.py` copied every label file out of the
TIER 4 mixes into `data/ocr_textpool/`: 21 files, **101,954 rows, 10,466
distinct texts**, 10,400 kept (66 boilerplate prompts dropped), 8,528,526 chars,
**78.6% in Arabic Unicode ranges** (no cp1256 mojibake). `texts.jsonl` is 29.65
MB, verbatim copies live in `raw/`, and the pool totals 76.81 MB. This is the
bare-text pool Stage 1 needs.

Two caveats: the texts are **article-length** (median 705 chars), so a line
renderer must segment them first; and `extra/labels_qari_v03.json` is
diacritized with `h1`/`u`/`i` markup, so use the `plain` field.
**`data/ocr_textpool/` is now the only surviving copy of those labels -- never
delete it.**

Status unchanged: nothing OCR has started; Stage 0 is still user-gated.

## 2026-10-04 (c) -- Stage 0 bake-off RAN (E-76 closed): printed is solved by tiny models, handwriting is not

The user approved Stage 0 and it is complete. Report: `runs/ocr_bakeoff/REPORT.md`.

**Free-running, un-mirrored CER** (greedy CTC over the model's own logits; the image is the
only input, so no teacher forcing is possible):

| eval set | what it is | v5 mobile (1.98M) | v6 small (3.10M) | our from-scratch (recorded) |
|---|---|---|---|---|
| `data/benchmark/isippt` (500) | PRINTED | **8.70%** | 13.75% | 144.8% |
| `data/benchmark/khatt` (200) | HANDWRITTEN | 62.61% | 44.41% | 97.8% |
| `data/hflines/val` (1,654) | our in-domain | 71.24% | 57.80% | not run |

The harness reproduces the v6 model card on the same ISI-PPT subset (13.75% vs 13.5% CER;
45.25% vs 47.3% WER), which is what makes the rest of the table credible.

**Verdicts.** E-76 bars 1 and 2 (<=20% / <=10% on `data/hflines/val`) **FAIL**; bar 3
(>=50% in both arms => the gap is our domain) **PASSES**, so Stage 1 must lead with handwriting.
**The 20-50M nano thesis is CONFIRMED for printed Arabic** - a 1.98M-param model is 16.6x better
than our from-scratch control and 87.4% of its lines are under 20% CER.

**Provenance correction.** `data/hflines` is a **handwriting** corpus (KHATT + Muharaf +
reallines), not the "mixed" set the docs called it: only 1.5% of its val lines are read under
20% CER by both printed-trained arms, and its pixels are colour manuscript scans. The E-76
bar was therefore a handwriting bar.

**Traps solved** (recorded in MEMORY.md): both models decode in **visual (mirrored) order** -
un-mirror before scoring or printed Arabic looks like CER 80%; the CTC alphabet is
`['blank'] + dict + [' ']` = 749 classes; `paddle.jit.load` reads the PIR `inference.json`
directly, so `paddleocr`/`paddlex` was never installed.

**Next (user-gated, Stage 1 gate): Stage 0.5** - fine-tune the pretrained recognizer on the
25,921 real handwriting lines in `data/hflines/train`. It already beats our from-scratch
model 2.2x on handwriting zero-shot, so adapting it is the cheapest path to a usable model.

## 2026-10-04 (d) -- Stage 0 RESULT (E-76) - MERGED INTO (c) AND REMOVED 2026-10-06

This section duplicated section (c) above (same E-76 result, same report path,
same table values). It was written before (c) and never deleted, so the file
reported the same Stage 0 result twice. Content removed; see (c) and
[RETROSPECTIVE.md](RETROSPECTIVE.md) section 5.
## 2026-10-04 (e) -- Stage 0.5 pilot (E-77): fine-tuning a printed recognizer on 2k handwriting lines HURTS

PRE-REGISTERED before any number was seen. FREE-RUNNING un-mirrored CER on `hflines_val`: control
**57.80%** / lr 1e-4 **72.09%** / lr 1e-5 **64.04%** - every fine-tuned arm loses to doing nothing,
progressively from epoch 1. The framework's own teacher-forced `norm_edit_dis` IMPROVED
(0.168 -> 0.140) while free-running CER worsened, and `best_accuracy.pdparams` was 14 pp worse
than the start. **FAIL / NO-GO as configured.** 2k handwriting lines is not enough.

## 2026-10-04 (f) -- Stage 0.5 follow-up (E-78): the width lever is REAL but SMALL

Post-hoc diagnostic (not pre-registered). The train/eval width mismatch was a genuine confound:
the SAME zero-shot checkpoint scores **57.80%** at dynamic width but **73.24%** at fixed width 320
- a **15.44 pp** swing with no training at all. Matching the eval width bought 1.28 pp at lr 1e-5
(62.76% vs 64.04%) and LOST 6.25 pp at lr 1e-4 (78.34% vs 72.09%). **FAIL as a rescue; E-77
stands.** Carry: always measure at the deployment width, and a sudden 5-10x batch-cost spike is a
WDDM shared-memory spill, not thermal.

## 2026-10-04 (g) -- Stage 1 GATE PASS (synthetic engine) + Stage 2 detector built and launched

Stage 1 is closed. The renderer `arabic_ocr/src/data/synthetic.py` shapes Arabic with **HarfBuzz**
(`uharfbuzz`) and rasterizes with **FreeType** - PIL on this machine is built **without Raqm**
(`ImageFont.core.HAVE_RAQM is False`), so PIL alone renders disconnected left-to-right Arabic and
cannot be used. PAW boxes are the **connected components of the rendered ink**, merged when two
components share a HarfBuzz cluster, so a floating haraka or a detached dot stays with its letter.
That makes the boxes exact rather than threshold-guessed.

**Gate evidence (mechanical, `arabic_ocr/scripts/stage1_verify.py` -> `runs/stage1_gate/verify.json`):**
200 pages / 1600 lines, 57 Arabic fonts, 28-44 px, with degradations.

| check | result |
|---|---|
| coverage_fail_lines | 0 |
| paw_box_mismatch_lines | 0 |
| orphan_glyphs | 0 |
| empty_render_lines | 0 |
| transcript_mismatch_lines | 0 |
| ink_uncovered_px | 0 (of 5,227,637 clean ink px) |
| ink_multi_covered_px | 0 |
| ink_in_foreign_line_box_px | 0 |
| **GATE** | **PASS** |

Two real bugs were found by the gate and fixed: `render_page` recorded the **pre-`fit_line`** text,
so 25 of 1600 lines were mislabelled against their own pixels; and 30 polarity-inverted pages made
the degraded-ink metric read 0.0065 until the image was flipped when the record carries `invert`.
A naive bbox-overlap check first reported 6881 cross-line overlaps - a **false alarm**: axis-aligned
PAW bounding boxes legitimately overlap, and the meaningful invariants are the ink-level ones above.
`verify_samples.txt` is the UTF-8 artefact for human review (the console cannot show Arabic).

**Stage 2 (PAW detector) started on the user go-ahead.** DBNet-style differentiable binarization,
`PAWDetector` **2,563,866 params**, output at **stride 2** - not the DBNet stride of 4, because the
median PAW is 15x24 px native and only 7x12 px at stride 2 (at stride 4 it would be under 6 px,
below any usable localisation). PAWs whose min side is < 6 px input (~28-30% of them: 3 px harakat
and dots) are dropped from the target and counted. Targets follow the exact offset
`d = area * (1 - r^2) / perimeter` with `shrink_ratio = 0.4`. 2000 train / 300 val pages rendered to
`runs/stage2_synth/`. Unit suite **26/26 PASS**; VRAM probe peak **1.84 GiB at batch 8** against
6.97 GiB free. Full run launched 2026-10-04 (job `pwsh-25`). **Gate: PAW Hmean >= 90% on held-out
synthetic.**

Note: STATUS.md had not been updated for (d)-(f); those sections above are reconstructed from
`HANDOFF.md` 2026-10-04 (d)-(f) and `research/EXPERIMENTS.md` E-76/E-77/E-78.

## 2026-10-04 (h) -- Stage 2: the eval metric was wrong, and DBNet cannot reach the gate at stride 2

The first Stage 2 run (`pwsh-25`, DBNet targets) reported Hmean 35-42%. **Those
numbers were meaningless** - the evaluator had two independent bugs:

1. it decoded connected components from the **shrunk** probability region and matched
   them against **full** ground-truth boxes (IoU ~0.33, far under the 0.5 threshold),
   and applied the `min_side` filter to the shrunk boxes instead of the unclipped ones;
2. it binarized with the **learned threshold map** rather than a fixed threshold.

Fixed by decoding at `min_side=1`, unclipping exactly (the quadratic inverse of the
shrink, not DBNet's approximate fixed 1.5), filtering after the unclip, and using
`bin_thresh=0.3`. Corrected re-score of the old `best.pt` (epoch 3): **H = 0.20-0.24**.

**Why the design had to change.** Decoding *perfect* target maps - the intrinsic
ceiling of each design, 40 val pages / 2860 PAWs, IoU 0.5, min_side 6:

| target design | stride 1 | stride 2 |
|---|---|---|
| DBNet shrunk box + component decode (as built) | 91.2% | **78.6%** |
| CRAFT-style Gaussian, k=0.15 | **96.9%** | 90.6% |
| CRAFT-style Gaussian, k=0.20 | 95.4% | **92.5%** |

DBNet's 78.6% is below the 90% gate, so no amount of training reaches it. The cause
is structural: **50.87% of PAWs share bounding-box area with another PAW** (22.55%
share *shrunk*-box area), so component labelling merges distinct PAWs - 2746
components for 2860 PAWs, and 281 components hold two or more PAW centres. PAW
**centres** are essentially unique (0.03% collide), which is why a centre-based
Gaussian design works where a component-based one cannot.

**Adopted: CRAFT-style region score, stride 2, k=0.20.** No architecture change - the
probability head becomes the region score and the threshold head is kept as an
auxiliary task, so the pre-registered "probability map + threshold map" output
contract is preserved. A per-PAW Gaussian with `sigma = k * size` decodes back
through a constant: at threshold 0.5 the blob extent is `2*sqrt(2 ln 2) * sigma`.
`CRAFTLoss` is a **weighted MSE** (`pos_weight=10` on `t>0.5`) - a focal loss would
drive every core pixel to exactly 1.0, flatten the peak and inflate every decoded
box. Verified by `scratch/stage2_craft_check.py`: analytic round-trip PASS, ceiling
on 40 val pages **P 98.72 / R 86.61 / H 92.27**, loss backprop PASS.

Relaunched 2026-10-04 (job `pwsh-28`, `runs/stage2_paw_craft/`). The DBNet baseline
artifacts in `runs/stage2_paw/` are kept intact for comparison. **Gate: PAW Hmean
>= 90% on held-out synthetic - PENDING `pwsh-28`.**

## 2026-10-04 (i) -- Stage 2: a THIRD eval bug (the CRAFT decode threshold), and the run it killed was at 46%

Continues (h). (h) replaced the DBNet target with a CRAFT region score and launched
`pwsh-28`. Its metric was still wrong -- a third bug -- and the run was killed on the
strength of it. This section corrects the record.

**1. The CRAFT blob constant was applied at the wrong threshold.** `CRAFT_BLOB_SCALE`
= `2*sqrt(2 ln 2)` is the extent of a Gaussian's level set at **0.5**, but
`evaluate_detector` binarized at `bin_thresh=0.3` and divided by the 0.5 constant.
Measured on *perfect* target maps (40 val pages, k=0.20, stride 2, min_side 6):

| threshold | decode constant | H | P | R |
|---|---|---|---|---|
| 0.3 | naive 2.3548 | 74.45 | 80.66 | 69.13 |
| 0.3 | correct 3.1035 | 90.27 | 97.80 | 83.81 |
| 0.5 | 2.3548 (= correct) | **92.27** | 98.72 | 86.61 |

Fixed: `craft_blob_scale(thresh)` + a `thresh=` argument on `decode_craft_boxes`, and
the craft branch of `evaluate_detector` now binarizes at `CRAFT_DECODE_THRESH = 0.5`
instead of `bin_thresh`.

**2. The threshold, not the constant, is what hid the model.** Re-scoring the killed
run's `best.pt` (epoch 5) with the fixed decode, 30 pages each:

| split | H | P | R | tp | fp | fn | fg>0.5 | w50 |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 46.36 | 56.63 | 39.24 | 700 | 536 | 1084 | 0.01249 | 29.7 |
| VAL | **45.90** | 57.81 | 38.06 | 873 | 637 | 1421 | 0.01731 | 34.0 |

against the **3.14%** the run printed. At tau=0.3 the model's soft halo bridges
neighbouring PAWs, so whole words fuse into one component (decoded w50 59.5 px at the
old setting vs GT 16 px).

**3. Train H == val H => the model was UNDERfitting.** 46.36 vs 45.90. The lever is
epochs/capacity, not regularisation.

**4. The architecture is not the wall, and AMP is innocent.** Overfitting 2 pages for
500 steps reaches **H 89-91%** (fg>0.5 0.0058 vs target 0.0106). `pos_weight` sweep on
that task: 10 -> 91.0%, 100 -> 83.6%, 1000 -> 74.2% (keep 10). fp32 91.0% vs
`autocast` 89.8% -- AMP does not cause the diffuseness.

**5. Relaunched.** `pwsh-28` was killed at epoch 7 on a metric that understated it
~15x; its artifacts are kept in `runs/stage2_paw_craft/` as evidence. Relaunched as
**`pwsh-29`** into `runs/stage2_paw_craft2/` with **30 epochs** (was 20), since the run
was still underfitting.

**Gate: PAW Hmean >= 90% on held-out synthetic - PENDING `pwsh-29`.**

## 2026-10-04 (j) -- Stage 2 GATE FAIL (54.89% vs 90%): the model cannot fit even TRAIN, and 98% of the loss was an unused head

Continues (i). `pwsh-29` ran all 30 epochs and **FAILED the pre-registered gate**:
best Hmean **54.89%** (epoch 25), final 54.43 (P 64.86 / R 46.88).

**The logged loss is prob-only.** `stage2_train.py` line 138 logs `running["prob"]`;
the thresh term is never printed. Measured: TRAIN `{prob: 0.00103, thresh:
0.06306}`, VAL `{prob: 0.00559, thresh: 0.11499}` -- the thresh term is **21-61x**
the prob term, and in craft mode the thresh head is never read at eval.

**The decode is not the problem.** Threshold sweep, model vs PERFECT target map,
identical code path, 30 val pages / 2294 PAWs:

| tau | model H | ceiling H |
|---|---|---|
| 0.5 | 55.50 | **91.97** |
| 0.6 | 60.36 | 89.66 |
| 0.7 | **64.71** | 86.28 |

**Not a size bias -- merging.** Decode-scale sweep `k_eff` 0.2 -> 1.0: 0.2 best
everywhere, 0.3 only +2.4 at tau 0.5, >= 0.4 collapses. Components span several
PAWs (0.72 components per PAW) and w50 is 29.7 px vs GT 16.0; no global constant
undoes that. Best possible with this checkpoint: **64.98%**.

**Underfitting, not a generalization gap.** 30 pages each, identical decode:
TRAIN H 58.44 / 64.23 / **68.34** vs VAL H 55.50 / 60.36 / **64.71** at tau
0.5 / 0.6 / 0.7. The training fit is the wall.

**Cause and next.** A 2.56M backbone shared with a thresh head that is unused at
eval and carries 21-61x the loss. Next (inside the pre-registered envelope):
zero `thresh_weight` in craft mode, and spend the freed capacity toward the
plan's 5M ceiling. **NEEDS_USER_DECISION - not started.**

**Gate: PAW Hmean >= 90% on held-out synthetic - FAILED at 54.89% (`pwsh-29`).**

## 2026-10-04 (k) -- Stage 2 GATE FAIL again (57.34%): capacity was not the bottleneck, the blob is the wrong SHAPE, and the 90% gate sits ON the representation's ceiling

Continues (j). `pwsh-52` ran all 30 epochs with `thresh_weight: 0.0` and 4.94M
params: **GATE FAIL, best Hmean 57.34%** (epoch 27), final 57.02 (P 65.42 / R
50.53). craft2 was 54.89%, so option A bought **+2.45 points** -- **capacity and
the unused head were NOT the bottleneck.** The objective fix is proven working
(`loss` == `loss_total` every step).

**Still underfitting.** TRAIN H 71.69 vs VAL H 67.45 at tau 0.7 (24 pages each).

**Detection is good; the gap is EXTENT.** Centre-in-component coverage on val is
**91.20%** (craft2: 87.94%) while H is 57.52% -- and H is almost exactly the
singleton coverage (985/1807 = 54.5%). PAWs covered by one component match; PAWs
in a merged component do not.

**The blob is the wrong SHAPE.** Singletons decode to **25.5 x 25.5 (square)**
against GT PAWs of 20.4 x 22.1 (tall): w 1.249x, h 1.154x. Merged components are
2.644x wide. One isotropic `k` cannot fix a shape error -- hence the k-sweep
collapse.

**FREE WIN: anisotropic decode, +12 points, no retraining.** Separate kx/ky:
tau 0.5 kx0.30/ky0.25 -> 67.38; **tau 0.7 kx0.25/ky0.20 -> 69.55** (isotropic 57.52).

**Merging vs threshold:** PAWs inside merged components 36.8% / 25.3% / **10.1%**
at tau 0.5 / 0.6 / 0.7. The merge is in the MASK, before the decode.

**STRUCTURAL: the gate sits on the ceiling.** Perfect-map ceiling at stride 2:
tau 0.5 **91.97-92.27**, 0.6 89.66, 0.7 **86.28**. The gate is 90%, so the only
threshold whose ceiling clears it leaves **2.3 points** -- and that is where 36.8%
of PAWs merge. At tau 0.7 the ceiling (86.28) is **below the gate**. **A perfect
model cannot pass at any threshold with manageable merging.**

Also: `CRAFTLoss.forward` never reads `target["ignore"]` (lines 88-106) -- inert
today (craft zeroes it) but a trap.

**Next - NEEDS_USER_DECISION:** adopt the anisotropic decode regardless, then
(a) stride-1 output (CRAFT k=0.15 ceiling 96.91%), (b) CenterNet-style centre
heatmap + w/h regression (centres are 99.97% unique, 91% detectable), or
(c) re-scope the gate.

**Gate: PAW Hmean >= 90% on held-out synthetic - FAILED at 57.34% (`pwsh-52`).**


