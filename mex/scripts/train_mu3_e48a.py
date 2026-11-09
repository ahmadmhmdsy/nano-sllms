"""mex/scripts/train_mu3_e48a.py - E-48a: per-family heads trained on TOWERS-ARMED hidden."""
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
from mex.scripts.train_mu2_g5_bridge import Bridge
voc = CharVocab()
seq = 96
pad = int(voc.vocab["<pad>"])
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bm._hold_kv = None
    bx._strength = 0.0; bx._hold_kv = None
    towers.append((bm, bx))
class DualWrapped(torch.nn.Module):
    def __init__(self, base, bm, bx):
        super().__init__()
        self.base = base; self.bm = bm; self.bx = bx
    def forward(self, *a, **kw):
        out = self.base(*a, **kw)
        y = out[0] if isinstance(out, tuple) else out
        hm = self.bm._hold_kv
        if hm is not None and self.bm._strength > 0:
            y = self.bm(y, hm.to(y.dtype))
        hx = self.bx._hold_kv
        if hx is not None and self.bx._strength > 0:
            y = self.bx(y, hx.to(y.dtype))
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
for i in range(2):
    trunk.model.layers[i] = DualWrapped(trunk.model.layers[i], towers[i][0], towers[i][1])
class ArrHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f1 = nn.Linear(640, 640)
        self.f2 = nn.Linear(640, n)
    def forward(self, h):
        return self.f2(torch.tanh(self.f1(h)))
ALPH2 = [c for c in voc.vocab if c.isdigit()] + ["+", "="]
ALPH2 = [c for c in ALPH2 if c and c != " "]
D2I = {c: j for j, c in enumerate(ALPH2)}
x3h = ArrHead(2).to(device)
x2h = ArrHead(len(ALPH2)).to(device)
opt = torch.optim.AdamW(list(x3h.parameters()) + list(x2h.parameters()), lr=1e-3, weight_decay=0.01)
train_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
feats3 = [it for it in train_items if it["task"] == "x3"]
lab3 = [(0 if it["target"] == "bad" else 1) for it in feats3]
feats2 = [it for it in train_items if it["task"] == "x2" and it["target"] and it["target"][0] in D2I]
lab2 = [D2I[it["target"][0]] for it in feats2]
@torch.no_grad()
def armed_h(prompts):
    ids_list = [voc.encode(p)[:seq] for p in prompts]
    B = len(ids_list)
    x = torch.full((B, seq), pad, dtype=torch.long)
    m = torch.zeros((B, seq), dtype=torch.bool)
    for i, t in enumerate(ids_list):
        x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        m[i, :len(t)] = True
    x = x.to(device)
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i2 + 1]
    hh = trunk.model(input_ids=x, output_hidden_states=True).last_hidden_state
    pos = m.sum(1) - 1
    g = hh[torch.arange(B, device=device), pos]
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return g
rng = np.random.default_rng(11)
for step in range(1, 601):
    b3 = rng.choice(len(feats3), size=min(32, len(feats3)), replace=False)
    b2 = rng.choice(len(feats2), size=min(32, len(feats2)), replace=False)
    with torch.no_grad():
        h3 = armed_h([feats3[int(i)]["prompt"] for i in b3])
        h2 = armed_h([feats2[int(i)]["prompt"] for i in b2])
    y3 = torch.as_tensor([lab3[int(i)] for i in b3], dtype=torch.long, device=device)
    y2 = torch.as_tensor([lab2[int(i)] for i in b2], dtype=torch.long, device=device)
    loss = F.cross_entropy(x3h(h3.detach().clone()), y3) + F.cross_entropy(x2h(h2.detach().clone()), y2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {float(loss):.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_router"
torch.save(x3h.state_dict(), str(dst / "x3_head_armed.pt"))
torch.save(x2h.state_dict(), str(dst / "x2_head_armed.pt"))
print("saved armed heads")
