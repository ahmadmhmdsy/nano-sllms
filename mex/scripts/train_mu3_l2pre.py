"""mex/scripts/train_mu3_l2pre.py - E-52 phase A: layer2-only dia pretrain (4000 steps)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
device = "cuda"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
seq = 96
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_tall/init")).to(device)
setted = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final"))
FREEZE_PREFIX = ("model.layers.0.", "model.layers.1.", "model.embed_tokens.weight")
for n2, p in trunk.named_parameters():
    p.requires_grad_(not n2.startswith(FREEZE_PREFIX))
base_sd = {k: v for k, v in load_file(str(ROOT / "runs/mex/mu3_g4/final/model.safetensors")).items()}
with torch.no_grad():
    sd = trunk.state_dict()
    for k in sd:
        if k in base_sd:
            sd[k].copy_(base_sd[k])
trunk.to(device)
trunk.train()
dtrain = PackedDataset(sorted((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin")), 96)
blocks = [dtrain[i]["input_ids"] for i in range(0, len(dtrain), max(1, len(dtrain) // 1000))]
opt = torch.optim.AdamW([p for p in trunk.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.1)
rng = None
import numpy as np
rng = np.random.default_rng(41)
for step in range(1, 4001):
    opt.zero_grad()
    blk = list(blocks[int(rng.integers(0, len(blocks)))])[:seq]
    x = torch.tensor(blk, dtype=torch.long, device=device).unsqueeze(0)
    lg = trunk(input_ids=x).logits[0][:-1].float()
    loss = torch.nn.functional.cross_entropy(lg, x[0, 1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in trunk.parameters() if p.requires_grad], 5.0)
    opt.step()
    if step % 400 == 0:
        print(f"step {step} loss {float(loss):.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_l2pre"
dst.mkdir(parents=True, exist_ok=True)
trunk.save_pretrained(str(dst))
items_val = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][::3]
tot = 0.0; n = 0
with torch.no_grad():
    trunk.eval()
    for it in items_val:
        ids = voc_encode = None
        enc = None
    
        continue
dst2 = dst
tot = None
print("[save done]", dst)

