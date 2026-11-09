"""mex/scripts/eval_mu3_e48b.py - E-48b gates: multi-step expert exact-match decode."""
from __future__ import annotations
import json, sys
from pathlib import Path
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
    bm._strength = 0.0; bx._hold_kv = None
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
I2D = {j: int(voc.vocab[c]) for j, c in enumerate(ALPH2)}
A4 = sorted("abcdefghijklmnopqrstuvwxyz")
D4C = {c: j for j, c in enumerate(A4)}
I2D4 = {j: int(voc.vocab[c]) for c, j in D4C.items()}
ex2 = Expert(len(ALPH2)).to(device); ex2.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/expert_x2.pt"), weights_only=True))
ex4 = Expert(len(A4)).to(device); ex4.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/expert_x4.pt"), weights_only=True))
for mm in [trunk, ex2, ex4]:
    mm.eval()
    for pp in mm.parameters():
        pp.requires_grad_(False)
val_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
def armed_unit(ids):
    x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i2 + 1]
    hh = trunk.model(input_ids=x, output_hidden_states=True).last_hidden_state
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return hh[0, -1]
def decode(it, which):
    table = I2D if which == "x2" else I2D4
    head = ex2 if which == "x2" else ex4
    ids = list(voc.encode(it["prompt"]))
    out_ids = []
    for step in range(8):
        if len(ids) >= seq:
            break
        with torch.no_grad():
            h = armed_unit(ids)
            lg = head(h)
        j = int(lg.reshape(-1).argmax())
        nid = int(table[j])
        out_ids.append(nid)
        ids.append(nid)
    return voc.decode(out_ids)
res = {}
for fam, n_eval in [("x2", 100), ("x4", 100)]:
    sel = [it for it in val_items if it["task"] == fam][:n_eval]
    ok = 0
    for it in sel:
        pred = decode(it, fam)
        if pred == it["target"]:
            ok += 1
    res[fam] = [ok, len(sel)]
    fam_name = fam
    print(fam, res[fam_name])
acc2 = res["x2"][0] / max(res["x2"][1], 1)
acc4 = res["x4"][0] / max(res["x4"][1], 1)
gates = {"gate_g1_x2_exact": acc2 >= 0.50, "gate_g2_x4_exact": acc4 >= 0.50}
result = {"x2_exact": acc2, "x4_exact": acc4, "counts": res, "gates": gates}
print(json.dumps(result))
(ROOT / "runs/mex/mu3_router" / "gates_e48b.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
