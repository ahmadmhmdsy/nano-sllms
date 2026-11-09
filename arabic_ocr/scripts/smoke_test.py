"""Smoke test for all 4 architectures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import create_model

configs = [
    {'type': 'vit_decoder', 'vocab_size': 256, 'image_height': 48, 'image_width': 256},
    {'type': 'crnn', 'vocab_size': 256, 'image_height': 48, 'image_width': 256},
    {'type': 'vlm', 'vocab_size': 256, 'image_height': 48, 'image_width': 256},
    {'type': 'distilled', 'vocab_size': 256, 'image_height': 48, 'image_width': 256},
]

print("=== Parameter Counts ===")
for c in configs:
    m = create_model(c)
    print(f"  {c['type']}: {m.count_params()/1e6:.1f}M params")

print("")
print("=== Forward Pass Tests ===")
x = torch.randn(2, 3, 48, 256)
text = torch.randint(0, 256, (2, 10))
text_lens = torch.tensor([10, 10])

# ViT Decoder
m = create_model(configs[0])
out = m(x, text)
print(f"  ViT: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")

# CRNN
m = create_model(configs[1])
out = m(x, text, text_lens=text_lens)
print(f"  CRNN: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")

# VLM
m = create_model(configs[2])
out = m(x, text)
print(f"  VLM: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")

# Distilled
m = create_model(configs[3])
out = m(x, text)
print(f"  Distilled: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")

print("")
print("=== All Architectures PASS ===")
