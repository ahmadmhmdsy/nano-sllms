"""mex/scripts/train_mu3_router.py - E-47: task-router + x3 head mounts (trunk/bridges frozen)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
device = "cuda"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from mex.src.vocab import CharVocab
voc = CharVocab()
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
seq = 96
pad = int(voc.vocab["<pad>"])
train_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
FAMS = ["x1", "x2", "x3", "x4", "dia"]
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(640, 640)
        self.f2 = nn.Linear(640, 6)
    def forward(self, h):
        return self.f2(torch.tanh(self.f1(h)))
class X3Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(640, 2)
    def forward(self, h):
        return self.f(torch.tanh(h))
router = Router().to(device)
x3h = X3Head().to(device)
opt = torch.optim.AdamW(list(router.parameters()) + list(x3h.parameters()), lr=1e-3, weight_decay=0.01)
@torch.no_grad()
def last_hidden(entry):
    B = len(entry)
    x = torch.full((B, seq), pad, dtype=torch.long)
    m = torch.zeros((B, seq), dtype=torch.bool)
    for i, t in enumerate(entry):
        t = t[:seq]
        x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        m[i, :len(t)] = True
    x = x.to(device)
    hh = trunk.model(input_ids=x, output_hidden_states=True).last_hidden_state
    p = m.sum(1) - 1
    return hh[torch.arange(B, device=device), p]
feats = []
for it in train_items:
    f = voc.encode(it["prompt"])
    if len(f) >= 4:
        feats.append((f, FAMS.index(it["task"])))
feats3 = [it for it in train_items if it["task"] == "x3"]
lab3_all = [0 if it["target"] == "bad" else 1 for it in feats3]
rng = np.random.default_rng(7)
for step in range(1, 801):
    b = rng.integers(0, len(feats), 64)
    picked = [feats[int(i)][0] for i in b]
    ys = [feats[int(i)][1] for i in b]
    with torch.no_grad():
        hv0 = last_hidden(picked)
        hv3_0 = last_hidden([voc.encode(feats3[int(i)]["prompt"]) for i in rng.choice(len(feats3), size=min(32, len(feats3)), replace=False)])
    judge = rng.choice(len(feats3), size=min(32, len(feats3)), replace=False)
    hv3 = 0
    hv3 = None
    b3 = rng.choice(len(feats3), size=min(32, len(feats3)), replace=False)
    with torch.no_grad():
        hv3_0 = last_hidden([voc.encode(feats3[int(i)]["prompt"]) for i in b3])
        lab3 = torch.as_tensor([lab3_all[int(i)] for i in b3], dtype=torch.long, device=device)
    hv = hv0.detach().clone()
    hv3 = hv3_0.detach().clone()
    opt.zero_grad()
    loss = F.cross_entropy(router(hv), torch.as_tensor(ys, dtype=torch.long, device=device)) + F.cross_entropy(x3h(hv3), lab3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {float(loss):.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_router"
dst.mkdir(parents=True, exist_ok=True)
torch.save(router.state_dict(), str(dst / "router.pt"))
torch.save(x3h.state_dict(), str(dst / "x3_head.pt"))
print("[saved] men 800 steps joint router+head")
print("done")
