# Arabic OCR Nano Model — Plan

> **REVISION 2026-10-04 (V2) - USER-APPROVED DIRECTION, NOT STARTED.**
> The original V1 plan is preserved below the divider as historical record.
> Two V1 assumptions are dead: (1) V1 Phase 1 planned around
> `medyas/arabic-ocr-printed-500k`, but that local cache entry holds
> `.no_exist` placeholders and ~7 KB of metadata only - the dataset is
> **NOT on disk**; (2) the V1 "train 4 architectures and compare" sweep is the
> process that produced the teacher-forced-CER era (`arabic_ocr/STATUS.md`).
> **Status: documentation only.** No run, install, download or GPU job has been
> started; the user gives the go per stage and has said explicitly not to start.
>
> **2026-10-06 UPDATE.** Stages 0-3 have since RUN (E-76..E-88): Stage 1 PASS,
> Stage 2 hybrid detector PASS (0.925 line recall), Stage 3 from-scratch FAIL
> (E-87), Stage 0.5b warm-start FAIL (E-88). **Stage 0.5c is user-gated at
> TASKS 131** and the in-domain-teacher eval is E-89. Live status:
> [../../arabic_ocr/STATUS.md](../../arabic_ocr/STATUS.md); full audit:
> [../../arabic_ocr/RETROSPECTIVE.md](../../arabic_ocr/RETROSPECTIVE.md).
>
> **2026-10-06 (b) E-89 RESULT.** The in-domain-teacher eval RAN: `FixFips/arabicocr-khatt` zero-shot = **46.93%** vs the 57.80% control on `data/hflines/val` (**+10.87 pp**, free-running, no LM), so route (D) transfers - but it is a weak teacher (median line CER 51.02%, only 14.9% of lines under 20%). The next step is training and remains user-gated (TASKS 131).

# V2 - two-stage detection + recognition (approved 2026-10-04)

## AMENDMENT 2026-10-05 (Stage 2 CenterNet FAIL + honest two-track target)

**Stage 2 gate has now failed FOUR times.** The CenterNet-style run (ceiling 98.96%)
scored only **46.97%** - worse than CRAFT's 59.86%. All four runs are
**underfitting** (train H == val H, still climbing at epoch 28). The bottleneck is
**optimization/capacity, not representation**: a 98.96% ceiling with 46.97% achieved
means the model cannot fit 2000 synthetic pages in 30 epochs.

| run | design | best H | ceiling | verdict |
|---|---|---|---|---|
| `stage2_paw_craft2` | CRAFT stride 2, 2.56M | 54.89% | 92.27% | FAIL |
| `stage2_paw_craft3` | CRAFT stride 2, 4.94M | 57.34% | 92.27% | FAIL |
| `stage2_paw_s1` | CRAFT stride 1, 5.01M | 59.86% | 96.91% | FAIL |
| `stage2_paw_cn` | CenterNet stride 2, 4.94M | 46.97% | 98.96% | FAIL |

**Honest two-track CER target** (replaces the single < 5% bar):
- **Printed: CER < 5%** - achievable. 1.98M model already at 8.70% (E-76); the
  synthetic print engine + a from-scratch recognizer should close this.
- **Handwriting: CER ~16-30%** - likely the realistic ceiling for a 20-50M
  from-scratch model on 25.9k lines. Literature: small CRNN ~16% on KHATT;
  2B VLM LoRA ~4-16%. **< 5% on handwriting is likely unreachable from scratch
  at this size.**

**Strategic question for the user (gates Stage 3):** is "20-50M from scratch" a
hard requirement or a preference? If it can relax to "pretrained base + LoRA",
handwriting < 5% is achievable today (Fanar-2 + QLoRA = 3.77% on KHATT) but
breaks the < 100 MB / > 50 img/s deployment targets.

## Decision record

- **User insight (accepted):** train a traditional OCR that first *detects* the
  smallest region containing characters and then *recognizes* that region,
  instead of asking one black-box seq2seq model to do everything. Accepted
  because a detection loss is per-pixel and spatial - it **cannot** take the
  character-language-model shortcut that destroyed every previous from-scratch
  run (see STATUS.md, "Why the models do not read").
- **User target (unchanged):** 20-50M params, CER < 5%, < 100 MB, > 50 img/s.
- **Correction accepted:** the detection unit is the **PAW (Part of Arabic
  Word)** / connected component, **not** the character. In cursive Arabic 2-6
  letters share one baseline stroke, so cutting at true character boundaries
  requires already knowing the word; dots (ب/ت/ث/ن/ي) and diacritics are
  *detached* components. VML-HD annotates at sub-word level for this reason.
  Recognizers therefore read a **sequence** over a detected region (CTC/NRTR),
  never classify an isolated crop.
- **Evidence that 20-50M is the right size:** PP-OCRv6 (arXiv 2606.13108) is
  titled "From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR
  Tasks". Reference points: `PaddlePaddle/arabic_PP-OCRv5_mobile_rec`
  (Apache-2.0, official) reports **81.27% strict line accuracy** on Arabic;
  `medyas/arabic_PP-OCRv6_small_rec` reports **CER 10.2%** (real
  printed scan) / **13.5%** (ISIPPT held-out) with **WER 47.3%**, trained on
  **800k synthetic lines** over 26 base + 12 display fonts, MultiHead CTC+NRTR,
  ~1050 character classes. The CER-to-WER gap is the decoder/lexicon gap, which
  is why Stage 4 exists.
- **User directive on data volume (2026-10-04):** do not materialize 500k
  images. Render only a chunk that fits the free disk, train on it, delete it,
  move on. Preferred implementation is rendering **inside dataloader workers**
  (near-zero disk), with chunked materialization as the fallback.

## Stages, gates, and costs

Every stage is separately user-gated; a failed gate stops the ladder.

### Stage 0 - zero-shot bake-off of existing small Arabic recognizers (~1 h, inference only)
- Arms: `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` and
  `medyas/arabic_PP-OCRv6_small_rec`, zero-shot, **free-running**
  harness on `data/hflines/val` (1,654 real lines), CER reported
  **split by source** (KHATT/Muharaf handwritten vs printed).
- Why first: it tests "can a 20-50M model read OUR data" with someone else's
  weights, before a GPU day is spent building one.
- Bars pre-registered as **E-76** in `research/EXPERIMENTS.md`.
- Disk: needs the 2.15 GB duplicate-checkpoint reclaim (row 124) + ~1 GB paddle
  runtime.
- Deliverable: `runs/ocr_bakeoff/REPORT.md` + per-sample JSON.

### Stage 1 - synthetic engine with exact glyph/PAW boxes (CPU, ~1 day)
- Replace `arabic_ocr/src/data/synthetic.py`. The current file is a
  smoke-test toy: it uses **Latin** fonts (arial/times/calibri) and draws raw
  Python strings with no shaping and no bidi, so it renders disconnected,
  left-to-right Arabic. Replace, do not extend.
- Shaping via `uharfbuzz` (absent) with
  `arabic_reshaper`+`python-bidi` as fallback. **HarfBuzz
  per-cluster advances give glyph boxes for free** - the enabler for Stage 2.
- 20+ real Arabic fonts are already on disk (Amiri, AmiriQuran, Noto
  Naskh/Kufi/Sans Arabic, Scheherazade, ReemKufi, tradbdo, simpo, Candarab,
  GARABD, arabtype).
- Text: the 25,921 real `data/hflines` transcripts (real word
  statistics) plus `data/ocr_textpool/texts.jsonl` - **10,400 distinct Arabic
  texts harvested 2026-10-04** from the superseded OCR mixes before their images
  were deleted (101,954 rows -> 10,466 distinct, 78.6% Arabic chars; provenance
  in `data/ocr_textpool/README.md`). Caveat: those texts are article-length
  (median 705 chars), so the renderer must segment them into lines. Degradation
  as in the PP-OCRv6 Arabic recipe (blur, noise, JPEG, rotation, texture,
  polarity inversion).
- Output contract: page image + PAW boxes + glyph boxes + transcript + font and
  degradation metadata.
- Gate: 200 rendered pages visually verified (write UTF-8 to a file - never
  judge Arabic from the console) + box-coverage assertion (every rendered glyph
  lies inside exactly one PAW box).
- Disk policy: `--chunk N` materialization (default ~20k pages, ~1 GB)
  plus `--on-the-fly` rendering in workers that writes nothing.

### Stage 2 - text-region detector (HYBRID: pretrained, user decision C 2026-10-05)
- **Decision (C):** use a validated pretrained Arabic detector (PaddleOCR class)
  instead of training from scratch. The from-scratch PAW detector failed 4x
  (all underfitting; CenterNet 46.97% vs 98.96% ceiling). The detector is a
  mature pretrained field; the from-scratch nano contribution is the recognizer.
- The detector provides text line/word boxes; the from-scratch recognizer
  (Stage 3) reads each region as a sequence (CTC/NRTR).
- Gate: the pretrained detector achieves >= 90% line recall on held-out
  synthetic pages, then measured on real `data/hflines`.
- The from-scratch PAW detector work is preserved as evidence
  (`runs/stage2_paw_*`) but is no longer the critical path.

### Stage 3 - PAW-sequence recognizer (GPU, ~4-8 h)
- 5-20M params, CTC + NRTR-style attention head over each detected region -
  **not** isolated-crop classification. ~250 character classes (28 letters x
  positional forms + lam-alef ligature + marks), or the ~1050-class scheme the
  shipped Arabic PP-OCRv6 model uses.
- Train on generated lines; fine-tune on the 25,921 real lines.
- Gate: free-running CER on `data/hflines/val`; must beat the Stage 0
  baseline.

### Stage 4 - decoder / lexicon (CPU or small GPU, ~1 day)
- Character n-gram or a 5-20M transformer LM over the recognized sequence plus
  lexicon beam search. This closes the CER-to-WER gap (the reference small model
  has CER 10.9-13.5% but WER 47.3%).
- Gate: WER reduction vs greedy at equal CER.

## Disk budget (measured 2026-10-04: **8.0 GB free on E:**)

| Item | Size | Note |
|---|---|---|
| `runs/trocr_finetune_v2/checkpoint-189/` | 2.15 GB | byte-duplicate of the top-level weights; reclaim candidate, user-gated (row 124) |
| paddle runtime + 2 models | ~1 GB | Stage 0 |
| synthetic chunk (default) | ~1 GB | deleted after each training chunk |
| on-the-fly rendering | ~0 | preferred mode |

## Stage 0 outcome (2026-10-04, E-76 closed - see runs/ocr_bakeoff/REPORT.md)

Free-running un-mirrored CER, zero-shot, CPU: **printed** `data/benchmark/isippt` (500)
v5-mobile **8.70%** / v6-small 13.75% (our from-scratch control 144.8% - a **16.6x** win from
**1.98M params**); **handwritten** `data/benchmark/khatt` (200) 62.61% / 44.41% (control
97.8%); `data/hflines/val` (1,654) 71.24% / 57.80%. Both arms >=50% on hflines, so **E-76
branch 3 fires: the gap is our domain, and that domain is HANDWRITING.** Consequences for
this plan: (a) the 20-50M thesis is **confirmed for printed** - Stage 1's synthetic print
engine is the right engineering step and must beat 8.70% CER; (b) synthetic **print** cannot
fix handwriting, so Stage 1 must also cover the handwriting path; (c) **Stage 3 trains
the recognizer FROM SCRATCH** (user decision C, 2026-10-05) - the nano contribution is the
recognizer, so it is NOT warm-started from a pretrained Arabic recognizer; the pretrained
detector (Stage 2) is the only borrowed component; (d) Stage 2 is
unaffected (neither arm is a detector).

## Risks (recorded before starting)

1. **Domain mismatch (CORRECTED 2026-10-04, after Stage 0).** `data/hflines` is NOT
   "mixed": it is a **handwriting corpus** (KHATT + Muharaf + reallines, and the first two
   are handwritten). Stage 0 measured it - only **1.5%** of its val lines are read at <20%
   CER by both printed-trained recognizers, and its pixels are colour manuscript scans.
   KHATT and Muharaf are
   *handwritten*, ISIPPT is printed. Synthetic printed data will not fix
   handwriting, and the 25,921 real lines are the only handwriting asset.
2. **Cascade compounding.** det recall x rec accuracy is the ceiling, and a
   missed PAW is an unrecoverable deletion.
3. **No page-level real data.** `data/hflines` is line-level only
   (widths 60-2457 median 600; heights 26-153 median 60), so Stage 2 can only
   be fine-tuned on real data by treating lines as pages, or by sourcing
   page-level images with boxes (VML-HD is sub-word annotated).
4. **Disk.** 8.0 GB free is the binding constraint, and the
   `.no_exist` mirage means the 500k-line datasets we believed were
   local are not.
5. **The 5% CER target may be unreachable on handwriting** with 25.9k lines;
   printed-only may be reachable.

## Out of scope (V2)

- Any training, install or download before the user gives the per-stage go.
- VLMs and billion-scale teachers.
- Page layout analysis beyond text regions.

---

# V1 ARCHIVE - original plan (2026-09-29), SUPERSEDED 2026-10-04

## Objective (V1)
Build a nano Arabic OCR model (20-50M params) with near-SOTA performance, trainable on a single 8GB GPU.

## Constraints
- Single Turing GPU (sm_75, 8GB) — fp16 only
- Nano size: 20-50M params (small: 50-100M later)
- Must smoke test on tiny data before full training
- All architecture options must be tried

## Phase 0: Setup & Smoke Test (tiny data) — DAY 1
**Goal**: Verify the entire pipeline works on 100-1000 images before scaling up.

### Tasks
1. Create tiny Arabic OCR dataset (100-1000 images)
   - Use synthetic data: render Arabic text with PIL + Arabic fonts
   - Or subset of medyas/arabic-ocr-printed-500k
2. Implement data pipeline (image loading, text tokenization, augmentation)
3. Smoke test ALL 4 architectures on tiny data:
   - Option A: Compact ViT + Transformer Decoder
   - Option B: CNN-Transformer Hybrid (CRNN-style)
   - Option C: Vision-Language Model (scaled down)
   - Option D: Distilled VLM
4. Verify training loop, loss computation, backward pass, checkpoint saving
5. Run eval on tiny val set — verify CER/WER computation

**Smoke test criteria**: Each architecture must complete 10 training steps + 1 eval step without errors.

## Phase 1: Data Pipeline — DAY 2-3
**Goal**: Build a robust data pipeline for Arabic OCR.

### Tasks
1. Download/prepare datasets:
   - medyas/arabic-ocr-printed-500k (printed Arabic)
   - KHATT (handwritten, if time permits)
   - Synthetic data generation (render Arabic text with various fonts)
2. Data augmentation:
   - Random crops, rotations, scaling
   - Noise injection, blur, contrast
   - Font variation, size variation
3. Text processing:
   - Arabic text normalization (remove tatweel, normalize alef forms)
   - Diacritics handling (keep or strip based on target)
   - Tokenization (character-level or subword)
4. Dataset class + DataLoader with proper batching

## Phase 2: Architecture Experiments — DAY 3-5
**Goal**: Train all 4 architectures on small data (~10k images), compare.

### Option A: Compact ViT + Transformer Decoder
- ViT-Tiny: patch 16, 4 layers, hidden 256, 4 heads (~5M params)
- Projector: linear layer
- Transformer Decoder: 4 layers, hidden 256, 4 heads (~10M params)
- Total: ~15-20M params
- Loss: Cross-entropy (teacher forcing)

### Option B: CNN-Transformer Hybrid (CRNN-style)
- CNN backbone: 4-6 conv blocks (~3M params)
- BiLSTM or Transformer encoder: 2 layers (~5M params)
- CTC loss or attention decoder
- Total: ~10-15M params
- Loss: CTC (no alignment needed)

### Option C: Vision-Language Model (scaled down)
- Small ViT: patch 16, 6 layers, hidden 384 (~8M params)
- Small LLM: Qwen2.5-0.5B or similar (~500M params — TOO BIG)
- Alternative: Use a tiny custom LM (~10M params)
- Total: ~20M params (with tiny LM)
- Loss: Cross-entropy

### Option D: Distilled VLM
- Same architecture as Option A or C
- Teacher: QARI-OCR or Baseer (pre-trained)
- Distillation loss: KL divergence on output distributions
- Total: ~20M params
- Loss: CE + alpha * KL(teacher, student)

**Comparison metrics**: CER, WER, inference speed, model size, training time

## Phase 3: Efficiency Techniques — DAY 5-6
**Goal**: Apply efficiency techniques to the best architecture from Phase 2.

### Techniques to try
1. **Knowledge Distillation** (if not already in Option D)
   - Teacher: QARI-OCR or Baseer
   - Student: Best architecture from Phase 2
   - Loss: CE + alpha * KL(teacher, student)
2. **Quantization (INT8/INT4)**
   - Post-training quantization (PTQ)
   - Quantization-aware training (QAT)
3. **Pruning**
   - Structured pruning (remove entire heads/layers)
   - Unstructured pruning (magnitude-based)
4. **Patch merging**
   - Reduce number of patches before transformer
   - E.g., merge 2x2 patches into 1

**Target**: < 100 MB model size, > 50 img/s inference, < 5% CER

## Phase 4: Full Training & Evaluation — DAY 6-8
**Goal**: Train the best model on full dataset, evaluate on benchmarks.

### Tasks
1. Train on full dataset (~500k images)
2. Evaluate on:
   - medyas/arabic-ocr-printed-500k test set
   - KHATT (if available)
   - Custom test set with diacritics
3. Compare with SOTA:
   - QARI-OCR (large)
   - Baseer (medium)
   - Arabic-Nougat (small)
   - Nanonets OCR-s (small)
4. Ablation studies:
   - With/without distillation
   - With/without quantization
   - Different data mixes

## Phase 5: Deployment — DAY 8-9
**Goal**: Optimize for inference, create demo.

### Tasks
1. ONNX export
2. INT8 quantization for deployment
3. Inference benchmark (latency, throughput)
4. Demo script (image → text)
5. Comparison with existing models

## Monitoring & Checkpoints
- [ ] Phase 0: Smoke test PASS on tiny data (all 4 architectures)
- [ ] Phase 1: Data pipeline ready (500k images)
- [ ] Phase 2: Architecture comparison complete (best architecture selected)
- [ ] Phase 3: Efficiency techniques applied (model < 100 MB)
- [ ] Phase 4: Full training complete (CER < 5%)
- [ ] Phase 5: Deployment ready (ONNX, INT8, demo)

## Risk Mitigation
- **Start tiny**: Always smoke test on 100-1000 images before scaling
- **Monitor VRAM**: Use vram_cap_gib to prevent OOM
- **Checkpoint frequently**: Save every N steps for auto-resume
- **Validate early**: Run eval on small val set after each phase
- **Compare baselines**: Always compare with existing models (arabic-small-nougat, Nanonets OCR-s)

## File Structure
```
arabic_ocr/
├── configs/
│   ├── smoke.yaml          # Tiny data smoke test
│   ├── nano_vit.yaml       # Option A
│   ├── nano_crnn.yaml      # Option B
│   ├── nano_vlm.yaml       # Option C
│   ├── nano_distilled.yaml # Option D
│   └── full_train.yaml     # Full training
├── src/
│   ├── models/
│   │   ├── vit_decoder.py     # Option A
│   │   ├── crnn.py             # Option B
│   │   ├── vlm.py              # Option C
│   │   └── distilled.py        # Option D
│   ├── data/
│   │   ├── dataset.py          # Dataset class
│   │   ├── augmentation.py      # Augmentation pipeline
│   │   └── synthetic.py        # Synthetic data generation
│   └── training/
│       ├── trainer.py          # Training loop
│       ├── losses.py           # CTC, CE, distillation
│       └── eval.py             # CER/WER computation
├── scripts/
│   ├── prepare_data.py         # Data preparation
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   └── infer.py                # Inference script
├── data/
│   ├── tiny/                   # Tiny dataset for smoke test
│   ├── train/                  # Training data
│   └── val/                    # Validation data
└── runs/
    └── arabic_ocr/             # Training runs
```

### Addendum 2026-10-06 -- route (D) teacher-selection finding (TASKS 134 / E-90 / E-91)

The route (D) step of this plan evaluated in-domain teachers zero-shot on
`data/hflines/val`. Both candidates were subsequently found to share training data with
that corpus: 9.6% of the val lines are in the KHATT training set of
`FixFips/arabicocr-khatt` and 79.9% are in the Muharaf training manifests of
`sdkv2/muharaf-arabic-ocr`. On the 174 val lines held out from both, FixFips scores
61.88% CER and Muharaf 54.06% against the 57.80% control, so route (D) transfers
inside a teacher own domain but only ~parity outside it. The gate text above is
unchanged; the expectation attached to it is corrected by TASKS 137. Details:
`arabic_ocr/STATUS.md` section (o).
