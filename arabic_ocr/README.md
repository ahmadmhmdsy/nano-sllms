# Arabic OCR Nano Model

A lightweight Arabic OCR model using deep learning, designed for deployment on edge devices.

> **WARNING (2026-10-04) - THE TABLES BELOW ARE NOT MEASUREMENTS.**
> The status table and the CER/WER comparison table in this file come from the
> era when every reported CER was **teacher-forced** (the decoder was handed the
> ground-truth prefix, so the numbers score a character language model, not OCR).
> Under honest free-running evaluation the same models emit one identical string
> for every image and get 0% of words right - see [STATUS.md](STATUS.md).
> "CRNN 4.2M CER 1.54" is such an artifact.
> The current honest status and the approved two-stage plan (PAW-region
> detection + sequence recognition) live in [STATUS.md](STATUS.md) and
> [../docs/plans/2026-09-29-arabic-ocr-nano-model.md](../docs/plans/2026-09-29-arabic-ocr-nano-model.md)
> (V2 section). The historical numbers below are preserved as a record of the
> measurement bug, not as results. See [RETROSPECTIVE.md](RETROSPECTIVE.md)
> for the 2026-10-06 audit of the whole line.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Smoke test (synthetic data) | ✅ Complete |
| 1 | Architecture comparison (4 models) | ✅ Complete |
| 2 | Efficiency techniques | ✅ Complete |
| 3 | Full training (50k images) | ✅ Complete |
| 4 | Deployment pipeline | ✅ Complete |

## Architecture Comparison

| Model | Params | CER | WER | Best For |
|-------|--------|-----|-----|----------|
| **CRNN** | **4.2M** | **1.54** | 1.0 | **Best CER, smallest** |
| VLM | 18.4M | 9.44 | 0.99 | Most flexible |
| Distilled | 7.8M | 10.09 | 1.0 | Same as ViT |
| ViT Decoder | 7.8M | 12.1 | 1.0 | Most interpretable |

## Full Training Results

- **Dataset**: 50k synthetic Arabic OCR images (40k train, 10k val)
- **Model**: CRNN (4.2M params)
- **Training**: 30 epochs, batch size 32
- **Best CER**: 2.5750 (257.5% error rate)
- **Final CER**: 2.7656
- **WER**: 1.0000 (100% word error rate)

### Analysis

The high CER is expected because:
1. Synthetic data has only 50 unique Arabic phrases
2. Character-level OCR is difficult for Arabic (complex ligatures, diacritics)
3. Model is small (4.2M params) with limited capacity

## Efficiency Techniques

| Technique | Latency | Throughput | Speedup | Notes |
|-----------|---------|------------|---------|-------|
| Original | 3.13 ms | 319.5 img/s | 1.0x | Baseline |
| Quantization | N/A | N/A | N/A | LSTM not supported on CUDA |
| Pruning (30%) | 3.76 ms | 265.7 img/s | 0.83x | Slower (sparse overhead) |

### Key Findings

1. **Quantization fails on CUDA** — LSTM dynamic quantization is CPU-only in PyTorch
2. **Pruning hurts speed** — Unstructured sparsity is not GPU-optimized
3. **Original model is already fast** — 319 img/s exceeds the 50 img/s target by 6x
4. **Model is already small** — 4.2M params = ~16MB fp32, ~4MB fp16

## Deployment

### Model Performance

- **Latency**: 3.41 ms
- **Throughput**: 293.3 img/s
- **Model size**: 4.2M params (~16MB fp32, ~4MB fp16)

### Usage

```python
from src.models import create_model
from src.data.dataset import ArabicOCRDataset
import torch

# Load model
config = {
    'type': 'crnn',
    'vocab_size': 256,
    'image_height': 48,
    'image_width': 256,
    # ... other config
}
model = create_model(config)
checkpoint = torch.load('runs/full/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
# ... see scripts/deploy.py for full example
```

### Files

- `scripts/train.py` — Training script
- `scripts/eval.py` — Evaluation script
- `scripts/deploy.py` — Deployment script
- `scripts/efficiency.py` — Efficiency techniques
- `configs/full.yaml` — Full training config
- `runs/full/best.pt` — Best model checkpoint

## Next Steps

1. **Get real Arabic OCR data** — Download actual image datasets (e.g., Khatt, EvArEST)
2. **Improve model** — Use larger model, better architecture, or transformer-based approach
3. **Add diacritics support** — Arabic diacritics are important for accurate OCR
4. **Optimize for edge** — Use TensorRT, ONNX Runtime, or TFLite for deployment
5. **Create web demo** — Build a simple web interface for testing

## Technical Details

### CRNN Architecture

- **CNN Backbone**: 4 conv blocks (32, 64, 128, 256 channels)
- **BiLSTM**: 2 layers, 256 hidden units
- **CTC Loss**: Connectionist Temporal Classification
- **Input**: 48x256 grayscale images
- **Output**: Character-level predictions

### Training

- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler**: Cosine annealing
- **Batch size**: 32
- **Epochs**: 30
- **Device**: CUDA (single GPU)

## License

MIT
