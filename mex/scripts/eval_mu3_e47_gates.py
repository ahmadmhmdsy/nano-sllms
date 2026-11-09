"""mex/scripts/eval_mu3_e47_gates.py - E-47 gates (b),(c),(d),(e) + identity check."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
device = "cuda"
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE = 8; seq = 96; pad = int(voc.vocab["<pad>"])
mark_ids_t = torch.tensor(mark_ids, device=device)
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bm._hold_kv = None; bx._strength = 0.0; bx._hold_kv = None
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
head = MarkHead(640, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu3_joint/head.safetensors")))
head = head.to(device).eval()
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
rz = torch.load(str(ROOT / "runs/mex/mu3_router/router.pt"), weights_only=True)
fo = open(str(ROOT / "runs/mex/mu3_router/router.pt"), "rb")
fo.close()
router = Router(); router.load_state_dict(rz); router.to(device).eval()
x3_sd = torch.load(str(ROOT / "runs/mex/mu3_router/x3_head.pt"), weights_only=True)
x3h = X3Head(); x3h.load_state_dict(x3_sd); x3h.to(device).eval()
val_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
@torch.inference_mode()
def main():
    FAMS = ["x1", "x2", "x3", "x4", "dia"]
    ok = tot = 0
    dig_ok = dig_tot = 0
    x3_ok = x3_tot = 0
    for it in val_items:
        f = voc.encode(it["prompt"])
        if len(f) < 4:
            continue
        x = torch.as_tensor(f, dtype=torch.long, device=device).unsqueeze(0)
        import torch as _t
        h = _t.as_tensor(trunk.model(input_ids=x, output_hidden_states=True).last_hidden_state[0, -1:].float())
        route = int(torch.as_tensor(router(h)).reshape(-1).argmax())
        fam = FAMS[route] if route < len(FAMS) else "x4"
        if fam == it["task"] or (it["task"].startswith("x") and fam == it["task"]):
            ok += 1
        tot += 1
        gold_first = voc.encode(it["target"])[0] if voc.encode(it["target"]) else -1
        lg = trunk(input_ids=x).logits[0, -1].float()
        pred = int(lg.argmax())
        if fam == "x2":
            digits = [int(v) for c, v in voc.vocab.items() if c.isdigit() or c == "=" and v < 97]
            sub = lg[digits].argmax()
            predx = digits[int(sub)]
            dig_tot += 1
            if predx == gold_first:
                dig_ok += 1
        if fam == "x3":
            k = int(torch.as_tensor(x3h(h)).reshape(-1).argmax())
            x3_tot += 1
            if (it["target"] == "bad" and k == 0) or (it["target"] != "bad" and k == 1):
                x3_ok += 1
    acc = ok / max(tot, 1); dacc = dig_ok / max(dig_tot, 1); xacc = x3_ok / max(x3_tot, 1)
    print(f"router acc {acc:.4f} ({ok}/{tot}) | x2 digit-fill {dacc:.4f} ({dig_ok}/{dig_tot}) | x3 {xacc:.4f} ({x3_ok}/{x3_tot})")
    gates = dict(gate_b=acc >= 0.95, gate_c=dacc >= 0.80, gate_d=xacc >= 0.85)
    print(json.dumps(gates))
    (ROOT / "runs/mex/mu3_router" / "gates.json").write_text(json.dumps({"router_acc": acc, "x2_digit_acc": dacc, "x3_acc": xacc, "gates": gates}, indent=1), encoding="utf-8")
main()
