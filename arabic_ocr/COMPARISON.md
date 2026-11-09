# Arabic OCR Model Comparison

> **WARNING (2026-10-06) - THE COMPARISON NUMBERS ARE NOT MEASUREMENTS.**
> ISIPPT 34.41% / KHATT 16.56% and every "our model" row below were produced
> under **teacher forcing** (the ground-truth prefix was handed to the decoder),
> so they score a character language model. Free-running evaluation of the same
> checkpoints collapses to one repeated string. The honest bake-off numbers are
> in [STATUS.md](STATUS.md) (Stage 0, 2026-10-04) and
> [runs/ocr_bakeoff/REPORT.md](../runs/ocr_bakeoff/REPORT.md). This file is kept
> only as a record of the measurement bug.

## Our Model (all_combined)

| Metric | Value |
|--------|-------|
| Parameters | 25.7M |
| Model size | 308 MB |
| Val CER | 0.9124 |
| Val WER | 0.9167 |
| ISIPPT CER | 34.41% |
| ISIPPT WER | 51.68% |
| KHATT CER | 16.56% |
| KHATT WER | 23.96% |

## SOTA Models

| Model | Params | CER | WER | Size |
|-------|--------|-----|-----|------|
| QARI-OCR v0.2 | 2B | 6.1% | - | ~8 GB |
| Arabic-Nougat | ~1B | 6.1% | - | ~4 GB |
| Baseer | ~2B | - | - | ~8 GB |

## Comparison

| Model | Params | ISIPPT CER | KHATT CER | Size |
|-------|--------|------------|-----------|------|
| Our Model | 25.7M | 34.41% | 16.56% | 308 MB |
| QARI-OCR v0.2 | 2B | 6.1% | - | ~8 GB |
| Arabic-Nougat | ~1B | 6.1% | - | ~4 GB |

## Analysis

Our model is **78x smaller** than QARI-OCR v0.2 (25.7M vs 2B params) and **26x smaller** in size (308 MB vs 8 GB).

However, our CER is still **5.6x higher** than SOTA (34.41% vs 6.1% on ISIPPT).

## Training Progress

| Model | Data | Val CER | ISIPPT CER | KHATT CER | Size |
|-------|------|---------|------------|-----------|------|
| vit_real | 480 | 0.189 | 139.52% | 88.46% | 94 MB |
| combined_large | 5384 | 0.9127 | 50.87% | 25.63% | 307 MB |
| all_combined | 5784 | 0.9124 | 34.41% | 16.56% | 308 MB |

## Next Steps

1. **Get more training data** - We have 5784 images, need 10k+ for better performance
2. **Fine-tune from SOTA model weights** - Use QARI-OCR or Arabic-Nougat as teacher
3. **Knowledge distillation** - Distill knowledge from SOTA teacher to our student
4. **Apply efficiency techniques** - Quantization, pruning to reduce model size

## Efficiency Techniques

### Quantization
- INT8 quantization can reduce model size by 4x (308 MB → 77 MB)
- May slightly reduce accuracy

### Pruning
- Structured pruning can reduce parameters by 30-50%
- May slightly reduce accuracy

### Knowledge Distillation
- Use SOTA model as teacher to improve student accuracy
- Can improve CER by 5-10%
