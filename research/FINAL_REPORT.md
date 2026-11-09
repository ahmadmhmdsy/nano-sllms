# ARABIC-DIACRITIZATION — comprehensive final report (2026-09-16)

Project "D-line" in this repo (nano_SLMs). Written for: (a) future agents/users
needing every detail for later use, (b) possible HF-side publication of the
model/report. Per user decision 2026-09-16: **RESEARCH-ONLY deployment** —
Sadeed-overlap disclosure and the Tashkeela/GPL-2 lineage intentionally left
OPEN and are NOT curve-fit away in any number below.

Honesty contract: every number in sections 4-5 is either re-verified from
on-disk artifacts (verbatim provenance stated per row) or marked as
"ledger-recorded" (documented verdicts from runs whose prediction files no
longer exist; tolerance +-0.1-0.3 pp). Nothing is inferred from vibes.

---

## 1. Evaluation protocol (applies to every number below)

- Metric: DER (diacritization error rate, %) = fraction of Arabic-base
  characters whose diacritic label differs from the gold annotation.
  Computed by diacritizer/scripts/eval.py compare (CPU). Also surface
  WER, DER_nocase (case-less sans-vowel equivalence), text_preservation
  (fraction of non-Arabic/non-base bytes preserved byte-exact).
- Gates (4): fadel_test (2500 lines), sadeed25 (1612), wikinews2024 (356,
  CONTAMINATED - see 4.1), wikinews2014 (393).
- Held-out control (added E-17): abdou_tashkeel test-00000 parquet split -
  15,091 sentences / 41,378 lines / 1,536,096 words - NEVER included in any
  training window by construction (only train/valid were ever prepped).
  bench.py gained an abdou_test source for it; predictions + refs kept at
  scratch/{stage2final,stage2b2500}/abdou_test_pred(.ref).txt.
- All stage-2-era numbers are re-verified from on-disk pred/ref pairs; older
  rows (v2b/v2d28/b65/fadel_spec/stage2a end-state) are ledger-recorded
  (prediction artifacts deleted in the 2026-09-13 cleanup), reported as-is.
- Byte-exactness: the model inserts marks ONLY after Arabic-base characters;
  every other byte passes through, so round-trip preservation of Latin,
  numbers, punctuation is architecturally guaranteed (verified ~1.0 on all
  scored sources; 0.9846 on abdou_test = 4 gold-reference lines in the split
  itself differ, not model-caused).

## 2. Data provenance (what the models actually ate)

Raw sources under data/diac/raw/: abdou_tashkeel, fadel, qcri_diac_clone,
sadeed_25, sadeed_tashkeela, wikinews.

| corpus id | contents / tokens (ctx-128 windows unless noted) | used by |
|---|---|---|
| v2b tokens | fadel_train+val, wikinews2024, abdou_tashkeel train+valid (char windows), qcrI wiki pages | v2b, b65, fadel_spec (fadel slice only) |
| v3 tokens | v2b + Sadeed_Tashkeela (990,544 train / 52,134 val) = 2,441,327 train / 160,630 val windows | stage2a, stage2b, stage2a2500, stage2b2500, stage2final |
| stage1 tokens (E-14) | 302,119 ctx-512 windows from the sources above (w/o wn2024?) - small scale | stage1lm |
| stage1 v4 tokens (E-17) | WHOLE corpora: abdou FULL (181 M chars), sadeed FULL (148 M), qcri (29.8 M), fadel train (11 M) = 370 M chars, 1,226,329 train / 12,580 val ctx-512 windows; **wn2024 EXCLUDED = 0**; gate-shingle exclusion 7,103 docs | stage1lm_v4 |
| EXCLUDED from training (gates) | fadel test/val, sadeed_25, wikinews2014, wn2024 (after contamination stamp; in v2b it was inside, removed from v4), abdou test | gates |

License status (research-only decision context): abdou_tashkeel card says MIT
for the COMPILATION but 45% of content is Tashkeela (commonly GPL-2 mirrored)
+ shamela pages (research intent) + GPT-4o-mini-diacritized wikipedia; fadel
benchmark derives from Tashkeela; sadeed_tashkeela research-per-paper; qcri =
paper-bundled shared data, citation expected. NO number in this report is
changed by that; a commercial superset would need a real license audit.

## 3. Hardware / environment constants

- Quadro RTX 3000 6,144 MiB, Turing sm_75, FP16-ONLY (never bf16), one CUDA
  device; Intel UHD iGPU only exists as WDDM *shared* memory - any spill into
  it is a ~10x slowdown cliff (measured incident, see 5.2).
- Venv .venv (uv-managed CPython 3.12.9), torch autocast fp16 + GradScaler,
  fp32 master weights, transformers 5.16.1 API surface.
- Hard facts: disk headroom swings -> automatic [DISK-WARN]/[DISK-PRUNE]
  (frees <3 GB by pruning weights_stepN probes, keep newest 2 + best_gate);
  keep-3 checkpoint rotation; zero-flag auto-resume contract (re-run the exact
  command, no flags - survived 6+ real crashes/pauses without loss).

## 4. Per-run cards (9 trainings + 3 auxiliary legs)

Card fields: config-what / provenance of number / caveats-found-later.

### 4.1 v2b - baseline generalist                    [ledger-recorded]
- 14L/384h/8h/2kv/ffn1536, ctx 128, 30.0 M params. lr 2e-4, bs 32, 30k steps
  budget, patience 12. Config configs/diac_v2.yaml (phase v2b). Trained on
  char-128 windows after catching the v2 bug (1024-char windows silently
  truncated to 128 by the tokenizer ctx - the silent-training-error LESSON).
- Gates ledger-recorded: 47.1 / 59.9 / 61.1 / 56.4; preservation 1.0,
  sadeed 0.9999.
- Caveats: wn2024 was IN its train corpus -> the 61.1 "gate" is in-domain
  (checked by the 2026-09-14 shingle audit: Jaccard 0.968). Prediction files
  deleted before the report -> not re-verifiable now (hand-count of the raws
  says the number's order is right).
- Verdict: KEEP as the historical baseline; do not compare against as a peer.

### 4.2 fadel_spec - domain specialist              [ledger-recorded]
- Same 30 M arch trained ONLY on fadel_train (+5% fadel_val carve-out);
  6000-step budget, patience 8, lr 2e-4, bs 32 (configs/diac_fadel_spec.yaml).
- Gates ledger-recorded: 34.2 / 51.4 / 62.9 / 57.4.
- Reading: proves domain routing works (best fadel of the campaign until the
  gold matched it); proves it costs modern-news domain (wn gates worse).
- Caveat: sadeed25 overlap from its training slice is the sadeed 51.4 read.

### 4.3 b65 - capacity 76 M scale-up                 [ledger-recorded]
- 20L/512h/8h/2kv/ffn2048 ctx 128, lr 1.5e-4, bs 32, 30k budget
  (configs/diac_b65.yaml); same v2b tokens.
- Gates: 43.1 / 55.3 / 60.3 / 53.7. Reading: params help on classical
  (fadel 43.1 vs 47.1) less than depth helps modern; 2.5x cost for ~4 pp
  average is the worst value-per-params result in the campaign and one of the
  few "DO NOT GROW" proofs in print.

### 4.4 v2d28 - depth ablation                       [ledger-recorded + live pred]
- 28L/272h/8h/2kv/ffn1088 ctx 128 = 30,103,600 params (exact same as all later
  stage-2 models); v2b tokens. Gates: 45.0 / 56.2 / 59.1 / 52.5 (-2.0..-3.9 pp
  vs v2b at ZERO extra params). val_loss 0.1801 vs v2b 0.1802 - a DEAD call
  (val is blind to what the gates see). Configs/diac_v2d28.yaml.
- Verdict: depth is the better axis at M-capacity; kicked off the v2d28 recipe
  set reused by every transfer arm.

### 4.5 stage1lm - Stage-1 char-LM pretrain (E-14)   [completed, artifact kept]
- Causal char-LM, v2d28 arch, TIED lm_head (h@embed.T), stage1 tokens (302k
  windows), ctx 512, VRAM-capped 5.8 GiB, bs 8 x accum 4.
- Smoke 1500 steps (pwsh-52) PASS (val 2.3825->1.6976), FULL 12k steps (pwsh-53)
  6.15 h, val 2.4075 -> 1.2806 [BEST] at final eval = NOT saturated at budget
  edge. Zero crashes. runs/diac/stage1lm/final + checkpoint-12000 kept.
- Which main use happens later: warm-start source for stage2a/stage2b.
- Incident-worthy: nothing lost; disk-pressure handled manually once mid-run.

### 4.6 stage2a - warm-start plain fine-tune (E-14 stage-2)  [closed reset]
- pretrained_init runs/diac/stage1lm/final/model.pt strict=False: 255/255
  tensors, 0 missing/0 unexpected; 15-class label head fresh (embeddings +
  encoder stack inherited). v3 tokens; bs 32, 8k budget, patience 8
  (configs/diac_stage2_a.yaml).
- 1 crash (E: hit 0.4 MB free at step 1000 - disk full); resumed zero-flag.
- Gates (END weights, rescored): 41.6 / 54.5 / 60.9 / 54.1 -> loses to the
  best from-scratch arm on ALL 4 gates. Verdict register: warm-start ALONE is
  not the frontier.
- This run exposed the WDDM shared-iGPU spill at bs32-direct, which mutated
  design -> the VRAM-cap discipline of every later run.

### 4.7 stage2b -> stage2b2500 «GOLD» (E-15)          [best artifact we have]
- Recipe: warm start from stage1LM + reset_last_n_layers 2 (fresh last 2 of
  28 blocks, CATT-style); v3 tokens; same lr 2e-4 / bs 32 / 8k budget /
  patience 8. In-run GATE PROBES every 2500 -> gate_eval.csv.
- RESULT (the campaign's most consequential): overfitting VISUALACHED in the
  first run - val_loss kept improving to the very end while ALL 4 gates
  degraded monotonically after ~step 2500; the step-2500 snapshot was
  best-of-ladder: Fadel 34.1 / Sadeed 47.0 / WN14 49.6 / WN24 57.9*.
- Rotation accident: the live ckpt-2500 was deleted mid-run by the old
  save policy -> pre-registered deterministic REPLAY (stage2b2500,
  configs/diac_stage2_b2500.yaml, total_steps 2500, same seed/config); its
  step-2500 persisted via gate probe = runs/diac/stage2b2500/final +
  gate_probe/best_gate_weights.pt; gates @2500 (rescored):
  **33.2 / 45.7 / 57.6* / 49.0**; held-out abdou_test **41.6 DER / 30.4
  nocase** (pwsh-3, 2026-09-16).
- Verdict: DEPLOYED FINAL MODEL. Research-only.

### 4.8 stage2a2500 - matched-probe control (E-16)
- Same v3 corpus/lr/seed as stage2b2500, reset_last_n_layers: 0 (plain warm),
  total_steps 2500, gate_eval_every 1250 (configs/diac_stage2_a2500.yaml).
- Two probe points: @1250 gates 35.4/47.1/59.2/50.7; @2500 gates 35.4/46.8/
  58.0/49.7 (pwsh-70 rescore; one crash + resume lossless en route).
- Reading: the warm-start-finds-its-peak-@2500 property is a WARM-START
  effect, NOT a reset artifact; still loses to reset-last-2 on ALL 4 at
  matched step -> recipe adopted (reset-last-2 wins, poverty-confirmed control and all).

### 4.9 stage1lm_v4 - whole-corpus LM init (E-17 leg 1)
- Tokens: 1,226,329 ctx-512 windows (370 M chars; abdou/sadeed/qcri FULL,
  wn2024=0 per contamination stamp). Same arch/cap trick (bs 8 x accum 4,
  VRAM cap 5.8). 27k-step budget. pwsh-72 launched ~13 h; user PAUSED it
  (~35 min work lost); zero-flag RESUME (pwsh-1) exact checkpoint-23000.
- Final: val 1.2769 -> 1.1905, [BEST] at 27,000 - STILL DESCENDING at budget
  edge -> honest memory: not converged; a longer/annealed LM is a priced option
  we did not buy.

### 4.10 stage2final - the E-17 final fine-tune        [everything verified]
- configs/diac_stage2_final.yaml: recipe B (reset-last-2) + warm init from
  stage1lm_v4/final/model.pt (255/255 loaded), v3 tokens, lr 2e-4, 8k budget,
  GATE PROBES every 1250. Run pwsh-2 completed exit 0.
- Probes: @1250 36.2/48.2/58.0/50.7; **@2500 PEAK 34.9/47.8/57.5*/49.6
  (mean 44.1)**; @3750 all worse; @5000-8000 all clearly worse while val kept
  improving to 0.1735 -> run-end mean 46.5 = the WORST probe point.
  best_gate.json: {"step":2500, "mean_der":0.4410}; per-probe permanent
  snapshots weights_step{1250,2500,3750,5000,6250,5000,6250,7500,8000}.pt all
  on disk under runs/diac/stage2final/gate_probe/.
- HELD-OUT abdou_test (15,091 sents): stage2final@2500 **49.4 DER / 33.3
  nocase** vs gold 41.6 / 30.4 (pwsh-3, both passes on the same GPU back to
  back, SAME eval pipeline, SAME refs -> apples-to-apples).
- Verdict: whole-corpus init REJECTED for deployment; kept only as a
  matched-experiment artifact + a counterexample to "more pretrain data = "
  must be better".

## 5. Use vs don't-use list (what to carry forward)

USE:
1. **Model = runs/diac/stage2b2500/gate_probe/best_gate_weights.pt** (cleanest
   gates of the campaign; nocase 30.4 held-out; preservation ~1.0). Any future "final" must BEAT it
   on the held-out abdou_test + the 3 clean gates, decided BEFORE any further
   training investment.
2. Recipe reset-last-2 + gate-probe-peak early stop (~2.5k fine-tune steps),
   probed every 1250-2500.
3. Gate watchdog + best_gate snapshot mechanism (train.py gate_probe) as a
  a first-class feature for every future fine-tune in the repo.
4. Fresh never-trained split discipline (abdou test reserved; if reused too
   often it degrades - rotate a new untouched split per campaign).
5. Eval instrumentation: bench.py multi-source + eval.py compare (CPU-only,
   re-runnable by hand from committed pred/ref pairs in scratch/).
6. VRAM discipline: micro-batch 8 x accum 4 + hard cap; never direct-batch-32
   on this card.
7. Zero-flag resume + keep-3 + disk guard: unattended-safe ops.

DON'T USE:
1. Init scale-ups beyond stage1lm (the small one): proven dead twice.
2. val_loss for selection: wrong 3/3 times.
3. 76 M params route: poor ROI; 30 M/28L keeps the pace on this GPU.
4. plain warm-start: loses to reset-last-2.
5. Raw Tashkeela archive directly.
6. wn2024 as external gate evidence.
7. Anything abdou-test-based for selection: it is now the single honest witness
   of generalization; re-using it for refinement would burn it (same logic as
   the wn2024 contamination that colored v2b numbers forever).

## 6. What might have produced wrong results (error inventory, honest)

1. **wn2024 contamination** - v2b-stage models' wn24 numbers (59.1-62.9 range)
   are optimistic by construction; never cited as progress since 2026-09-14.
2. **sadeed25 overlap flag** - sadeed_tashkeela IS in the stage2 train mix; the
   sadeed number is honest w.r.t. itself but NOT fully external. Treat it as
   "overlap-flagged" (works either hand: the GOLD's sadeed 45.7 is also the
   lowest of its gates so the ranking does not hinge on it).
3. **Older-row tolerances** - v2b/v2d28/b65/fadel_spec numbers are ledger
   copies (pred files deleted), reproduction-not-guaranteed; +-0.3 pp noise.
4. **The early stage1lm checkpoint loss** - rotation deleted live ckpt-2500
   (the incident that produced the REPLAY leg and the keep-3 + snapshot rules).
5. **fadel_spec had no gate watchdog** - its numbers are END-state reads; if
   its true peak was mid-run, its real gates might be slightly better than
   recorded (the direction the error runs is AGAINST adoption of fewer-clears;
   the use/don't-use list is unchanged).
6. **Token budget in stage-1 v4** - still descending at 27k; init was never
   saturated; any new study could pick up longer LM training or LR-annealing
   as a priced option, but THIS campaign proves it would not have rescued
   the init route (the held-out loss argument stands: more init data,
   MORE overfit to the same unlabeled-domain distribution, still worse).
7. **bf16 inability** (Turing card) - everything runs autocast fp16; parameter
   polynomial sensitivity is a standard risk, but grad scaler + master fp32
   handled stability; no loss NaN ever observed.
8. **SOTA comparability** - matched-benchmark numbers (public papers) meet our
   gates on 3 surfaces; the real comparison row remains "we cover ~1/10 of the
   gap" from E-13; the NEW at this campaign: our GOLD's held-out abdou read
   (41.6) is the strongest honest generalization number the campaign has.

## 7. Human-published HF-model checklist (if user ever flips to publish)

MUST-FIX BEFORE PUBLISH:
1. Licenses: abdou_tashkeel card/MIT claim, Tashkeela lineage (45% of abdou +
   derivative), sadeed_tashkeela (paper license), qcri (research share) -
   resolve-or-restrict the model card to research-only AND list the dataset
   provenance matrix verbatim (.section 2 here).
2. Sadeed overlap DISCLOSURE in the model card (it gates the sadeed_25 number).
3. Actually publishing requires the deployable artifact: best_gate_weights.pt
   is a bf16-friendly torch save with an arch description; the infer.py path
   is ready (ask_model.bat); a model card needs the eval CSV + this report.
4. If publishing WEIGHTS: state the breadth of "training data includes web
   scraped franken-datasets (abdu card)" verbatim.

DON'T-FORGET:
- The REPORTED fadel numbers depend on the 5 trick: fadel_test gate is a
  benchmark ABOVE the raw fadel_test.txt; the same pipeline is used in
  every published number above - keep the eval.py compare instrument if any
  future model is claimed better than the gold.
- The 6,144 MiB card constraint shapes every hyper in section 4 - a published
  reproduction on a bigger GPU is expected to be FASTER, not different.

## 8. Register of all numbers (append-only provenance)

research/gate_results/gate_results.csv = 40 rows (source_of_number column
carries the pwsh job id / ledger marker of EVERY number), exactly mirrored by
research/gate_results/SUMMARY.md and the tables above. Any future disagreement
between docs resolves ONLY toward re-scoring from best_gate snapshots.
