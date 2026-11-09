
"""Verify the e63s_s7_18l64 config is valid."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

from model import build_from_config
import tokenizer as TK
import yaml
import torch

config_path = REPO / "runs" / "diac" / "e63s_s7_18l64" / "final" / "config.yaml"
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

print("Config:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

print("\nBuilding model...")
model = build_from_config(cfg["model"], TK.VOCAB_SIZE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,} ({round(n_params/1e6, 2)}M)")
print(f"Model size: ~{round(n_params * 4 / 1024 / 1024, 1)} MB (fp32)")

# Verify the model can do a forward pass
x = torch.randint(0, TK.VOCAB_SIZE, (1, 16))
with torch.no_grad():
    out = model(x)
print(f"Output type: {type(out)}")
if isinstance(out, dict):
    print(f"Output keys: {list(out.keys())}")
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
print("\nConfig is VALID!")
