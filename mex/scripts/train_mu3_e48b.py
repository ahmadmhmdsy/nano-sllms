"""mex/scripts/train_mu3_e48b.py - E-48b: per-family expert heads, teacher-forced per-position training."""
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
class Expert(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f1 = nn.Linear(640, 640)
        self.f2 = nn.Linear(640, n)
    def forward(self, h):
        return self.f2(torch.tanh(self.f1(h)))
ALPH2 = [c for c in voc.vocab if c.isdigit()] + ["+", "="]
D2I = {int(voc.vocab[c]): j for j, c in enumerate(ALPH2)}
I2D = {j: c for c, j in D2I.items()}
A4 = sorted("abcdefghijklmnopqrstuvwxyz")
D4C = {c: j for j, c in enumerate(A4)}
D4 = {int(voc.vocab[c]): j for c, j in D4C.items()}
D4 = D4
D4C = D4C
ex2 = Expert(len(ALPH2)).to(device)
ex4 = Expert(len(A4)).to(device)
opt = torch.optim.AdamW(list(ex2.parameters()) + list(ex4.parameters()), lr=1e-3, weight_decay=0.01)
train_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
fam2 = [it for it in train_items if it["task"] == "x2" and it["target"] and all(ch in "0123456789+" for ch in it["target"])]
fam4 = [it for it in train_items if it["task"] == "x4" and it["target"] and all(ch in "abcdefghijklmnopqrstuvwxyz" for ch in it["target"])]
print(f"train x2={len(fam2)} x4={len(fam4)}")
@torch.no_grad()
def armed_hiddens(batch):
    rows = []
    for it, tg in batch:
        pi = voc.encode(it["prompt"])[:seq - 1 - len(tg)]
        if len(pi) < 4:
            continue
        rows.append((pi, tg))
    B = len(rows)
    x = torch.full((B, seq), pad, dtype=torch.long)
    m = torch.zeros((B, seq), dtype=torch.bool)
    for i, (pi, tg) in enumerate(rows):
        t = pi + tg
        x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        m[i, :len(t)] = True
    x = x.to(device); m = m.to(device)
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i2 + 1]
    hh = trunk.model(input_ids=x, output_hidden_states=True).last_hidden_state
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return hh, m, rows
rng = np.random.default_rng(13)
for step in range(1, 1001):
    b2 = rng.integers(0, len(fam2), 24)
    b4 = rng.integers(0, len(fam4), 24)
    batch2 = [(fam2[int(i)], voc.encode(fam2[int(i)]["target"])) for i in b2]
    batch4 = [(fam4[int(i)], voc.encode_text_new(i) if False else voc.encode(fam4[int(i)]["target"])) for i in b4]
    hh2, m2, rows2 = armed_hiddens(batch2)
    hh4, m4, rows4 = armed_hiddens(batch4)
    with torch.no_grad():
        pass
    lg2 = ex2(hh2.detach().clone())
    lg4 = ex4(hh4.detach().clone())
    loss2 = 0.0
    loss4 = 0.0
    for i, (pi, tg) in enumerate(rows2):
        s = len(pi) - 1
        e = s + len(tg)
        if e > seq:
            continue
        tgt = torch.as_tensor([D2I[c] for c in tg], dtype=torch.long, device=device)
        loss2 = loss2 + F.cross_entropy(lg2[i, s:e], tgt)
    for i, (pi, tg) in enumerate(rows4):
        s = len(pi) - 1
        e = s + len(tg)
        if e > seq:
            continue
        tgt = torch.as_tensor([D4[c] for c in tg], dtype=torch.long, device=device)
        loss4 = loss4 + F.cross_entropy(lg4[i, s:e], tgt)
    loss = (loss2 + loss4) / max(1, len(rows2) + len(rows4))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step} loss {float(loss):.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_router"
torch.save(ex2.state_dict(), str(dst / "expert_x2.pt"))
torch.save(ex4.state_dict(), str(dst / "expert_x4.pt"))
print("saved experts")
