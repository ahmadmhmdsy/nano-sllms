# Small-model options for Arabic OCR (50-100M params, and below)

Status: **approved direction, 2026-10-01**. Companion to `arabic_ocr/STATUS.md`
(which records the teacher-forcing measurement bug and the honest baseline).

## 0. Objective (user-approved 2026-10-01)

Maximize accuracy at the **lowest possible parameter count / size**. Training
from scratch is **not required** — fine-tuning, LoRA, distillation and other
transfer routes are all in scope. The original < 100 MB model target stands
unless explicitly relaxed.

## 1. Size arithmetic (the first conflict to resolve)

| Params | fp16 weights | fp32 weights | int8 |
|---|---|---|---|
| 25M | 50 MB | 100 MB | 25 MB |
| 50M | 100 MB | 200 MB | 50 MB |
| 75M | 150 MB | 300 MB | 75 MB |
| 100M | 200 MB | 400 MB | 100 MB |

50-100M params in fp16 is 100-200 MB and therefore **breaks the < 100 MB target**.
Three ways out: cap at ~45M in fp16; use int8 (50-100M -> 50-100 MB); or relax
the target to < 200 MB. Given the 'lowest possible size' objective, prefer the
smallest student that still hits the accuracy bar.

Training VRAM note (8 GB Turing, fp16 only): AdamW state is 8 bytes/param plus
~4 bytes/param fp32 master weights, so a 60M model needs ~0.8 GB of optimizer
state before activations. Batch 8-16 at 64x1024 is the practical ceiling.

## 2. The constraint that decides the architecture: CTC needs T >= L + repeats

Arabic doubles letters constantly. KHATT `textlen_p50 = 64`, max ~130. At 1024 px
wide:

| Width downsampling | T | Verdict for Arabic lines |
|---|---|---|
| /16 (patch 16) | 64 | **too short** for most KHATT lines |
| /8 (current `crnn.py`) | 128 | marginal |
| /4 | 256 | safe |
| /2 | 512 | safe but slow |

**Landmine:** `crnn.py` uses `zero_infinity=True`. When T < L + repeats the CTC
loss is inf and PyTorch silently zeroes **both the loss and the gradient**. A
too-short T does not crash -- it quietly stops learning. With T=128 and
`max_text_len=120`, any line with more than 8 adjacent repeats receives zero
gradient. Fix T before judging the architecture.

## 3. The measurement rule (non-negotiable)

Every CER this project reported before 2026-10-01 was teacher-forced and did not
measure OCR (`runs/vit_real/best.pt`: 34.4%/16.6% teacher-forced vs
144.8%/97.8% free-running, WER 100%, identical output for every image).
**Only free-running CER from `arabic_ocr/scripts/eval_benchmark.py` may be used
to select or report a model.**

## 4. Option space (measured on this repo's code unless marked *published*)

### Family A -- CTC-only (structurally cannot cheat)

| ID | Architecture | Params | fp16 | Notes |
|---|---|---|---|---|
| A1 | CRNN hd512 x4L + CTC | 26.1M | 52 MB | current design, needs /4 |
| A2 | CRNN hd1024 x2L + CTC | 40.6M | 81 MB | |
| A3 | CRNN hd1024 x3L + CTC | 65.8M | 132 MB | |
| A4 | ViT enc h384 x6L + CTC | 11.4M | 23 MB | T too short at patch16 |
| A5 | ViT enc h512 x6L + CTC | 19.8M | 40 MB | |
| A6 | ViT-B/16 (h768 x12L) + CTC | 86.3M | 173 MB | classic ViT-B |
| A7 | ConvNeXt-T + BiLSTM + CTC | ~43M | 86 MB | strong conv prior |
| A8 | ResNet-34 + 6L transformer + CTC | ~52M | 104 MB | |

### Family B -- Attention encoder-decoder (TrOCR-style)

| ID | Architecture | Params | fp16 |
|---|---|---|---|
| B1 | TrOCR-small (DeiT-S + 6L dec) | ~62M *published* | 124 MB |
| B2 | `CompactViTDecoder` h512 enc8/dec8 | 60.1M | 120 MB |
| B3 | `CompactViTDecoder` h512 enc6/dec6 | 45.4M | 91 MB |
| B4 | `CompactViTDecoder` h384 enc6/dec6 | 25.8M | 52 MB |
| B5 | TrOCR-base | ~334M | too big |

Family B **alone reproduces the diagnosed failure** (the decoder takes the
character-LM shortcut). Use it only with pretrained init or a CTC auxiliary loss.

### Family C -- Hybrid CTC + attention (the fix for the diagnosed bug)

| ID | Architecture | Params |
|---|---|---|
| C1 | ViT enc h384 x6L (11.4M) + 4L dec + CTC head | ~25M |
| C2 | ViT enc h512 x6L (19.8M) + 6L dec + CTC | ~55M |
| C3 | CRNN hd512 + 4L decoder + CTC | ~40M |

The CTC loss on the encoder **forces the encoder to be image-informative** --
precisely what is broken today -- while the attention decoder still gets
language modelling. Standard ESPnet/Conformer recipe, and verifiable: CTC loss
falling is direct evidence the encoder reads the image.

### Family D -- Non-autoregressive

| ID | Architecture | Params |
|---|---|---|
| D1 | ViT enc + CTC (= A4-A6) | 20-86M |
| D2 | ViT enc + parallel char head (NRTR) | ~30M |
| D3 | ABINet-style dual vision+LM branch | ~40M *published* |

No exposure bias, no LM shortcut, single forward pass -- easiest route past the
> 50 img/s target.

### Family E -- Pretrained init / distillation (highest accuracy per param)

| ID | Architecture | Params |
|---|---|---|
| E1 | 25-60M student distilled from a large Arabic OCR/VLM teacher | 25-60M |
| E2 | TrOCR-small fine-tuned from an Arabic-pretrained checkpoint | ~62M |
| E3 | Pretrained ViT/DeiT encoder + trained decoder | 25-90M |
| E4 | LoRA/adapters on a frozen pretrained backbone | +0.5-5M on top |

The repo already has `distill.yaml`, `compare_distilled` and a local teacher at
`data/teacher/qwen35_08b/`, so this path is partly built.

## 5. Recommendation, ranked (approved 2026-10-01)

1. **E1 -- distillation into a small student.** Best accuracy per parameter, and
   the existing infrastructure covers part of it. Prefer the smallest student
   (C1 ~25M or smaller) that clears the accuracy bar, given the size objective.
2. **C2 -- hybrid CTC+attention (~55M)** if training mostly from scratch; this is
   the one that fixes the diagnosed bug.
3. **A2 or A7 (~40-43M) -- CTC-only.** Simplest, cannot cheat, but no language
   model.
4. **E2 -- fine-tune TrOCR-small (~62M).** Most direct route to < 5%, needs a
   weight download and disk.

Given the clarified objective (highest accuracy, lowest params), the ordering
should be read as: **E1/E4 first** (transfer into the smallest possible student),
then C2 as the from-scratch fallback.

## 6. Honest caveat

Architecture is the **second** lever. The corpus is 26k lines; TrOCR was trained
on ~684M images. Even a flawless 60M architecture trained on 26k lines lands
around 15-30% CER, not < 5%. Transfer (fine-tune/LoRA/distill) is what makes the
target reachable at small size -- not the parameter budget.

## 7. Anti-patterns (do not repeat)

- Pure attention seq2seq without a CTC auxiliary loss.
- Teacher-forced CER as a selection or reporting metric.
- T < L + repeats with `zero_infinity=True` (silent zero gradient).
- 100M+ params at 64x1024 with batch > 16 on 8 GB.
