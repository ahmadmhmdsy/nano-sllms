"""mex/scripts/demo_mu3_e49.py - E-49: live composed decode demo for all families (read-only)."""
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
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE = 8; seq = 96; pad = int(voc.vocab["<pad>"])
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
head = MarkHead(640, 9); head.load_state_dict(load_file(str(ROOT / "runs/mex/mu3_joint/head.safetensors"))); head.to(device).eval()
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(640, 640)
        self.f2 = nn.Linear(640, 6)
    def forward(self, h):
        return self.f2(torch.tanh(self.f1(h)))
router = Router(); router.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/router.pt"), weights_only=True)); router.to(device).eval()
class ArrHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f1 = nn.Linear(640, 640)
        self.f2 = nn.Linear(640, n)
    def forward(self, h):
        return self.f2(torch.tanh(self.f1(h)))
ALPH2 = [c for c in voc.vocab if c.isdigit()] + ["+", "="]
I2D = {j: int(voc.vocab[c]) for j, c in enumerate(ALPH2)}
A4 = sorted("abcdefghijklmnopqrstuvwxyz")
I2D4 = {j: int(voc.vocab[c]) for j, c in enumerate(A4)}
def load_exp(n, f):
    ex = ArrHead(n).to(device)
    ex.load_state_dict(torch.load(str(f), weights_only=True))
    return ex
ex2 = load_exp(len(ALPH2), ROOT / "runs/mex/mu3_router/expert_x2.pt")
ex4 = load_exp(len(A4), ROOT / "runs/mex/mu3_router/expert_x4.pt")
x3h = ArrHead(2).to(device); x3h.load_state_dict(torch.load(str(ROOT / "runs/mex/mu3_router/x3_head_armed.pt"), weights_only=True))
for mm in [trunk, head, router, ex2, ex4, x3h]:
    mm.eval()
    for pp in mm.parameters():
        pp.requires_grad_(False)
def disarm():
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
def step_h(ids):
    x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    disarm()
    hs_all = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs_all[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs_all[i2 + 1]
    hh_lm = trunk.model(input_ids=x, output_hidden_states=True)
    hh = hh_lm.last_hidden_state
    lg = trunk(input_ids=x).logits[0, -1].float()
    cls = int(head(hh[:, -1:]).reshape(-1, 9).argmax(-1)[0])
    route = int(router(hh[:, -1:].float()).reshape(-1, 6).argmax(-1)[0])
    disarm()
    return lg, cls, route, hh[:, -1].float()
FAMS = ["x1", "x2", "x3", "x4", "dia"]
def decode_demo(prompt, n_out):
    base = list(voc.encode(prompt))
    ids = list(base)
    lg0, cls, route, hv = step_h(ids)
    fam = FAMS[route] if route < 5 else "x4"
    if fam == "dia":
        for k in range(n_out):
            lg, cls, route, hv = step_h(ids)
            nid = int(mark_ids[cls]) if cls != NONE else int(lg.argmax().item())
            ids.append(nid)
        return fam, voc.decode(ids[len(base):])
    if fam == "x3":
        k = int(x3h(hv).reshape(-1, 2).argmax(-1)[0])
        return fam, "bad" if k == 0 else "ok"
    table = I2D if fam == "x2" else I2D4
    exm = ex2 if fam == "x2" else ex4
    produced = []
    for k in range(min(9, n_out + 1)):
        if len(ids) >= seq:
            break
        lg, cls, route, hv = step_h(ids)
        j = int(exm(hv).reshape(-1).argmax())
        nid = int(table[j])
        produced.append(nid)
        ids.append(nid)
    return fam, voc.decode(produced)
val_items = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
picked = []
seen = set()
for it in val_items:
    if it["task"] not in seen and len(picked) < 5:
        picked.append(it)
        seen.add(it["task"])
out_rows = []
for it in picked:
    fam, out = decode_demo(it["prompt"], len(voc.encode(it["target"])))
    out_rows.append({"task": fam, "prompt": it["prompt"], "gold": it["target"], "model": out})
fam_d, out_d = decode_demo("\u062b\u0646\u064a\u0629\u060c|", 7)
out_rows.append({"task": "dia(forced)", "prompt": "glyph sample", "gold": "glyph sample", "model": out_d})
txt = json.dumps(out_rows, ensure_ascii=False, indent=1)
print(txt)
(ROOT / "runs/mex/mu3_router" / "demo_e49.json").write_text(txt, encoding="utf-8")
