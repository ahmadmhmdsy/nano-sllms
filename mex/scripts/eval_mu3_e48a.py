"""mex/scripts/eval_mu3_e48a.py - E-48a gates: armed-head readouts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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
D2I = {c: j for j, c in enumerate(ALPH2)}
I2D = {j: c for c, j in D2I.items()}
x3h = ArrHead(2).to(device); x3h.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/x3_head_armed.pt"), weights_only=True))
x2h = ArrHead(len(ALPH2)).to(device); x2h.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/x2_head_armed.pt"), weights_only=True))
for mm in [trunk, x3h, x2h]:
    mm.eval()
    for pp in mm.parameters():
        pp.requires_grad_(False)
val_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
@torch.inference_mode()
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
    pos = (m.sum(1) - 1).to(device)
    g = hh[torch.arange(B, device=device), pos]
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return g
tasks = []
for it in val_items:
    if it["task"] == "x2":
        tasks.append(("x2", it))
for it in val_items:
    if it["task"] == "x3":
        tasks.append(("x3", it))
out = {}
for fam in ["x2", "x3"]:
    sel = [it for f, it in tasks if f == fam]
    ok = 0
    for it in sel:
        with torch.inference_mode():
            h = armed_h([it["prompt"]])
        if fam == "x3":
            k = int(x3h(h).reshape(-1, 2).argmax(-1)[0])
            good = (it["target"] == "bad" and k == 0) or (it["target"] != "bad" and k == 1)
            out.setdefault("x3", [0, 0])
            out["x3"][1] += 1
            if good:
                out["x3"][0] += 1
        else:
            j = int(x2h(h).reshape(-1, len(ALPH2)).argmax(-1)[0])
            pred = I2D[j]
            out.setdefault("x2", [0, 0])
            out["x2"][1] += 1
            if it["target"] and pred == it["target"][0]:
                out["x2"][0] += 1
print(json.dumps(out))
acc_c = out["x2"][0] / max(out["x2"][1], 1)
acc_d = out["x3"][0] / max(out["x3"][1], 1)
gates = {"gate_c_armed": acc_c >= 0.80, "gate_d_armed": acc_d >= 0.85}
print(json.dumps({"x2_acc": acc_c, "x3_acc": acc_d, "gates": gates}))
(ROOT / "runs/mex/mu3_router" / "gates_e48a.json").write_text(json.dumps({"x2_acc": acc_c, "x3_acc": acc_d, "gates": gates}, indent=1), encoding="utf-8")
