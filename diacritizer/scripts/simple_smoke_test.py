
"""Simple smoke test for e63s_s7_18l64 - just verify model builds and trains."""
import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

from model import build_from_config
import tokenizer as TK
import yaml
import torch
import torch.nn.functional as F

config_path = REPO / "runs" / "diac" / "e63s_s7_18l64" / "final" / "config.yaml"
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

print("=== Simple Smoke Test for e63s_s7_18l64 ===")
print(f"Config: {cfg['model']}")

# Build model
print("\nBuilding model...")
model = build_from_config(cfg["model"], TK.VOCAB_SIZE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,} ({round(n_params/1e6, 2)}M)")

# Test forward pass
print("\nTesting forward pass...")
x = torch.randint(0, TK.VOCAB_SIZE, (2, 16))
with torch.no_grad():
    out = model(x)
print(f"Output type: {type(out)}")
if isinstance(out, dict):
    print(f"Output keys: {list(out.keys())}")
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

# Test backward pass
print("\nTesting backward pass...")
model.train()
x = torch.randint(0, TK.VOCAB_SIZE, (2, 16))
y = torch.randint(0, 15, (2, 16))
out = model(x)
if isinstance(out, dict):
    logits = out['logits']
else:
    logits = out
loss = F.cross_entropy(logits.view(-1, 15), y.view(-1))
print(f"Loss: {loss.item():.4f}")
loss.backward()
print("Backward pass: OK")

# Test optimizer step
print("\nTesting optimizer step...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
optimizer.step()
print("Optimizer step: OK")

print("\n=== Smoke test PASSED ===")
print("Model builds, forward pass, backward pass, and optimizer step all work correctly.")
