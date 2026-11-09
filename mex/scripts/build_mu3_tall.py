"""mex/scripts/build_mu3_tall.py - E-50: 3-layer taller trunk warm-init (layer2 = layer1 clone)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
src = ROOT / "runs/mex/mu3_g4/final"
base_sd = load_file(str(src / "model.safetensors"))
cfg = LlamaConfig.from_pretrained(str(src))
cfg.num_hidden_layers = 3
model = LlamaForCausalLM(cfg)
sd = {k: v for k, v in base_sd.items()}
for key in list(base_sd.keys()):
    if key.startswith("model.layers.1."):
        sd[key.replace("layers.1.", "layers.2.", 1)] = base_sd[key].clone()
missing, unexpected = model.load_state_dict(sd, strict=False)
bad = [m for m in missing if not m.startswith("lm_head") and "embed" not in m]
print("missing-after-fill:", missing[:4])
print("unexpected:", unexpected)
assert not bad, bad
assert not unexpected, unexpected
dst = ROOT / "runs/mex/mu3_tall/init"
dst.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(dst))
print(json.dumps({"saved": str(dst), "layers": 3, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "kv": cfg.num_key_value_heads, "ffn": cfg.intermediate_size}))
