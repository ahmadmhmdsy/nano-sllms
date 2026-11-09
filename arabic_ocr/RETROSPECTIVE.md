# Arabic OCR line - retrospective audit (2026-10-06)

> **WHAT THIS IS.** An evidence-based review of everything the Arabic OCR line
> (`arabic_ocr/`, TASKS rows 119-133) has done, written by a fresh agent
> after auditing the durable docs + on-disk artifacts. It records *what was
> done right*, *what was done wrong*, and *what is still at risk*. Companion
> docs: [STATUS.md](STATUS.md) (honest live status), [README.md](README.md),
> the plan [../docs/plans/2026-09-29-arabic-ocr-nano-model.md](../docs/plans/2026-09-29-arabic-ocr-nano-model.md),
> ledger rows E-24/E-76..E-89 in [../research/EXPERIMENTS.md](../research/EXPERIMENTS.md).
>
> **STATUS IS BLOCKED (TASKS 131, user-gated).** No run is active. The two
> legacy claim documents, [PROJECT_REPORT.md](PROJECT_REPORT.md) and
> [COMPARISON.md](COMPARISON.md), are **superseded and their numbers are not
> measurements** - banners were added on 2026-10-06.

## 1) One-paragraph verdict

The line has produced a **trustworthy measurement and evaluation stack**, a
**pixel-exact synthetic renderer (Stage 1 PASS)**, a **working hybrid detector
(Stage 2 PASS, 0.925 line recall@IoU0.5)**, and the strongest available
**negative results** on from-scratch recognition and out-of-domain fine-tuning
(E-24, E-77, E-78, E-80..E-88). It has **not** produced a usable handwriting
recognizer: the best honest number is the 57.80% zero-shot control, against a
<5% target. Most of the lost effort traces to a single class of bug - **a
metric that measured something other than OCR** - which recurred at least five
times before the team built controls that catch it.

## 2) Timeline

| when | what | outcome |
|---|---|---|
| ~2026-09-29..10-01 | V1 four-architecture sweep (CRNN/ViT/VLM/distilled) on synthetic then mixed data, model selection on "best_cer" | **All numbers teacher-forced; model selection invalid** |
| 2026-10-01 | Self-audit: `benchmark.py` passed ground-truth text into the decoder; free-running = one string for every image | **Measurement bug found and documented**; free-running harness built |
| 2026-10-03 | Path 1: fine-tune `RayR1/trocr-base-arabic-handwritten` | **FAIL** - base emits `''`/`'.'` on clean Arabic; 73 min wasted |
| 2026-10-04 | Two-stage plan approved (detect region -> recognize sequence); PAW not character | Direction set |
| 2026-10-04 | Stage 0 bake-off (E-76): pretrained tiny recognizers, free-running | **PASS for printed** (8.70% CER from 1.98M params); handwriting 44-72% |
| 2026-10-04 | Stage 0.5 pilot (E-77/E-78): fine-tune printed recognizer on 2k handwriting lines | **FAIL** - hurts at every lr; width confound found |
| 2026-10-04 | Stage 1 renderer (HarfBuzz + FreeType, exact PAW boxes) | **GATE PASS** - 0/1600 lines fail at pixel level |
| 2026-10-04..05 | Stage 2 from-scratch detector: DBNet, CRAFT width, capacity, stride-1, CenterNet (E-80..E-84) | **4 GATE FAILs** (46.97-59.86% vs 90%); gate sat on the representation ceiling |
| 2026-10-05 | Stage 2 hybrid: pretrained PP-OCRv5_mobile_det + measured post-processing (E-85/E-86) | **GATE PASS** - 0.925 line recall |
| 2026-10-05..06 | Stage 3 from-scratch 7.9M vit_hybrid (E-87) | **GATE FAIL** - 0.9100 CER; blank image scores 0.9026 |
| 2026-10-06 | Stage 0.5b full-corpus warm-start (E-88) | **FAIL** - 71.74% vs 57.80 control; catastrophic forgetting |
| 2026-10-06 | Route (D): search for an in-domain handwriting teacher | Two MIT PyTorch models found; eval not yet run (E-89) |

## 3) Done right

1. **The 2026-10-01 self-audit.** Finding that a seq2seq decoder was handed the
   ground-truth prefix (`model(images, text=texts)`) invalidated ~a month of
   results but stopped the project building on a character-LM score. It is the
   single most valuable act in the line. `STATUS.md` was rewritten honestly and
   `README.md` got a warning banner.
2. **Pre-registration.** E-76..E-89 fix arms and gates *before* numbers are seen;
   E-88 even falsified its own predecessor's carry note ("just get more data").
3. **Cheap falsification before GPU spend.** Perfect-target-map ceilings killed
   DBNet at stride 2 for free (E-80); the **blank-image control** named the
   Stage-3 failure in one 2-minute pass (E-87: 0.9026 blank vs 0.9173 free); the
   auxiliary CTC head (0.812) localised the failure to the visual half.
4. **Licensing a rebuilt external stack.** After the PaddleOCR checkout vanished,
   `stage05_eval.py` reproduced the E-77 control to the digit (CER 57.80,
   422/422 tensors) *before* training (MEMORY 2026-10-06).
5. **Gate over convenience.** PaddleOCR's teacher-forced `best_accuracy.pdparams`
   was rejected on principle and every epoch free-run individually (E-77/E-88).
6. **Data preservation.** 10,400 distinct Arabic texts harvested to
   `data/ocr_textpool/` before their images were deleted; `trainer_state.json`
   copied out before a checkpoint deletion; TIER-5 deletion refused; deletions
   verified against an ancestor guard (18.60 GB reclaimed).
7. **The right unit.** Accepting PAW/connected-component regions (not characters)
   and sequence decoders (never isolated-crop classification) was correct for
   cursive Arabic.
8. **Invariant tests where they mattered.** Stage 1 pixel-level verifier (0
   failures), 26/26 detector tests, 18/18 crop tests, crop labels assigned by
   geometry not index.
9. **Choosing the hybrid detector** once four from-scratch attempts failed, and
   re-measuring the pretrained model's stock post-processing (+0.51 recall).

## 4) Done wrong (and what each cost)

1. **The teacher-forcing era.** Architectures compared and "selected" on a
   metric that scored a character language model. Cost: the whole V1 phase and
   every `best_cer` in `runs/`.
2. **Metric bugs recurred ~5 times.** DBNet decode scored shrunk boxes against
   full GT (IoU 0.33) and binarised with the learned threshold map; the CRAFT
   blob constant was applied at the wrong level set (0.3 vs 0.5). A run was
   **killed at 46% because the metric read 3.14%**, and relaunched. Lesson only
   learned after 3 runs: "a metric proven in a scratch script is not proven in
   the module".
3. **Gate designed on top of the representation's ceiling.** At stride 2 the
   CRAFT ceiling is 92.27% only where 36.8% of PAWs merge, and 86.28% where they
   do not - against a 90% gate. Four detector runs were spent before anyone
   checked a *perfect* model could not pass.
4. **An unused head owned 98% of the loss.** `thresh` was 21-61x the `prob` term
   and never read at eval, for a full 30-epoch run; the log printed only `prob`.
5. **Out-of-domain fine-tuning attempted three times.** Broken TrOCR base
   (73 min; a 5-minute zero-shot check would have vetoed it), then E-77 (2k
   lines), then E-88 (25.9k lines, *worse*). A forgetting check on the model's
   own domain only appeared at E-88 (printed 13.75 -> 46.75).
6. **Width confound.** The E-77 control was scored at the training width 320
   (73.24%) instead of deployment width (57.80%) - a 15.44 pp self-inflicted
   swing (E-78).
7. **Strategy reversal.** E-76 had already measured a pretrained model reading
   our handwriting 2.2x better than from-scratch zero-shot, and the plan's
   original text recommended warm-start; a 7.9M from-scratch recognizer was
   still trained two-phase (E-87) before returning to warm-start, which then
   also failed (E-88). GPU-days.
8. **Un-annotated legacy claim docs.** `PROJECT_REPORT.md` ("Successfully built
   a complete Arabic OCR pipeline", "CER=0.189") and `COMPARISON.md`
   (ISIPPT 34.41% / KHATT 16.56%) carried no warning until 2026-10-06 - long
   after the bugs were known.
9. **Smaller engineering debts:** cp1256-vs-UTF-8 label bug across 7 scripts;
   TrOCR's 384x384 aspect squash; a killed pwsh job orphaning its python child
   so two trainers wrote one `history.json`; a dynamic-width OOM (10,199 MB on
   an 8 GB card).

## 5) Repo-hygiene findings (the live risk)

- **The whole line was untracked in git.** `arabic_ocr/` (122 files of code,
  configs, scripts, tests, `STATUS.md`) and
  `docs/plans/2026-09-29-arabic-ocr-nano-model.md` were untracked; 516 OCR
  evidence files under `runs/` had 0 tracked. `.gitignore` explicitly says
  "Code, results and reports stay tracked" for this line and only ignores
  `arabic_ocr/models/` - the intent was right and unexecuted. A `git clean` or
  a machine move would have destroyed the entire line's evidence. **Fixed
  2026-10-06** (see section 7).
- **AGENTS.md did not know this line existed.** The §2 layout and §1 document map
  had zero mentions of `arabic_ocr`. By the repo's own rule (§7, "a doc outside
  the map does not exist for the next agent") `STATUS.md` - the honest-status
  home - did not exist. That is how the fabricated legacy reports survived.
  **Fixed 2026-10-06.**
- **Status sprawl.** `STATUS.md` / `HANDOFF.md` / `README.md` /
  `PROJECT_REPORT.md` all claimed to be the status; `STATUS.md` duplicated the
  same Stage-0 result in sections (c) and (d). **Deduplicated 2026-10-06.**
- **Weights must stay out of git.** OCR runs hold ~9.2 GB (33 `.pdopt` x ~100 MB,
  16 `.pt`, 21,139 `.png`), and `.gitattributes` LFS-tracks only
  `*.safetensors` - so a careless `git add runs/` would create >100 MB
  non-LFS objects. **Ignore rules added 2026-10-06**; only textual evidence is
  committed.

## 6) Current state

- **Blocked** at TASKS 131 (user-gated): after E-88's FAIL, the routes are
  (B1') freeze the PPLCNetV4 backbone and train neck+heads - PaddleOCR has no
  `freeze_backbone` option, so it needs a custom loop; (B2') distil the 57.80%
  teacher - hard-capped by the weak teacher; (D) find an in-domain teacher;
  (C) re-scope Stage 3.
- Route (D) found two candidates: `FixFips/arabicocr-khatt` (CRNN+CTC, KHATT
  11,375 lines, claimed 6.09% CER) and `sdkv2/muharaf-arabic-ocr` (CER 13.31%).
  **The zero-shot eval on `data/hflines/val` vs the 57.80 control (E-89) was
  the agreed next action and had never been run.** It RAN on 2026-10-06 as part
  of this remediation: **PASS - 46.93% vs 57.80%, +10.87 pp**, so route (D)
  transfers zero-shot; but it is a weak teacher (median line CER 51.02%, only
  14.9% of lines under 20%). The best honest number on `data/hflines/val` is
  therefore **46.93%**, still far from the <5% target. The step that follows is
  training, and it remains user-gated at TASKS 131 (now superseded by TASKS 137).

  **CORRECTION (E-90/E-91, 2026-10-06, added after the audit below):** that
  46.93 compared the teacher against a corpus that contains its OWN training
  lines - 9.6% of the val lines are KHATT lines it trained on, where it scores
  CER 1.82%. On the 174 val lines held out from both candidate teachers it
  scores **61.88%, worse than the 57.80 control**; the Muharaf teacher scores
  54.06%. Route (D) transfers *inside* a teacher own training domain, not to
  unseen handwriting. See STATUS (o) and TASKS 137.
- No python/GPU OCR process is running.

## 7) Remediation log (2026-10-06)

| # | action | status |
|---|---|---|
| R1 | This retrospective written and registered in AGENTS.md | done |
| R2 | `arabic_ocr/` + plan + textual run evidence + text pool committed | done |
| R3 | Warning banners on `PROJECT_REPORT.md` and `COMPARISON.md` | done |
| R4 | `STATUS.md` deduplicated + current-state header | done |
| R5 | `AGENTS.md` §1/§2 register the OCR line | done |
| R6 | OCR ignore rules added (weights, renders, raw pool) | done |
| R7 | TASKS/HANDOFF/MEMORY updated; E-89 pre-registered | done |
| R8 | E-89 zero-shot eval of `FixFips/arabicocr-khatt` on `data/hflines/val` | **done - PASS: 46.93 vs 57.80 (+10.87 pp); weak teacher (median line CER 51%)** |
| R9 | E-89 harness + results committed; STATUS/HANDOFF/TASKS/MEMORY/EXPERIMENTS/WHAT_WORKS/plan updated with the result | done |
| R10 | Teacher-contamination audit (TASKS 134 / E-90 / E-91): both in-domain teacher numbers stratified against the teachers OWN training corpora, per-line JSONL added to both harnesses, E-89 interpretation corrected | **done - 79.9% of the Muharaf score and 9.6% of the FixFips score were in-train; on the 174 lines held out from BOTH, FixFips 61.88 (worse than the 57.80 control) vs Muharaf 54.06** |

## 8) Guardrails for the next agent

1. **Never gate on teacher-forced numbers.** Free-running only; run
   `diag_stage3.py`-style blank-image control before blaming data/capacity.
2. **Measure the decode's perfect-map ceiling before training any detector**, and
   check the gate against that ceiling.
3. **Unit-test the metric in the module that reports it** - a scratch script
   proving it is not enough.
4. **Before any fine-tune, evaluate the checkpoint on a benchmark from its
   ORIGINAL domain** to detect forgetting (E-88: +33 pp on printed).
5. **Measure at the deployment width** (E-78: 15.44 pp).
6. **Validate a pretrained base zero-shot on clean inputs before fine-tuning it**
   (E-24: 5 minutes would have saved 73).
7. **Search for an in-domain teacher before adapting an out-of-domain one** - the
   domain gap decides whether fine-tuning helps or destroys.
8. **Commit code + textual evidence as you go**; never leave a line untracked.
9. **A teacher score on a corpus that shares its training data is NOT transfer
   evidence.** Before trusting any teacher number, match the eval transcripts
   against the teacher OWN published training manifests and report the held-out
   stratum (`arabic_ocr/scripts/stratify_teacher_overlap.py` is the worked
   example). Compare candidates on the SAME held-out lines - two numbers both
   labelled "held-out" can be different lines and reverse the ranking.
