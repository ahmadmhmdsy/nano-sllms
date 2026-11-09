"""mex/scripts/eval_mu3_tall.py - E-50 gates: t1 dia char acc, t2 x2 fill, x3 transfer."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
ROOT = Path("E:/python_projects/nano_SLMs")
ROOT2 = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
seq = 96
import os
BASE = os.environ.get("TRUNK_BASE", "runs/mex/mu3_tall/final")
bdir = ROOT / BASE.replace("\\", "/")
print("base", bdir)
trunk = LlamaForCausalLM.from_pretrained(str(bdir)).to("cuda")
trunk.eval()
mub = torch.load(str(bdir / "bridge_w.pt"), weights_only=True) if (bdir / "bridge_w.pt").exists() else torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(bdir / "xbridge_w.pt"), weights_only=True) if (bdir / "xbridge_w.pt").exists() else torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(1280, 16).cuda(); bm.load_state_dict(mub["bridge" + str(i)]); bm._strength = 0.0; bm._hold_kv = None
    bx = Bridge(1280, 16).cuda(); bx.load_state_dict(xb["bridge" + str(i)]); bx._strength = 0.0; bx._hold_kv = None
    towers.append((bm, bx))
class DW(torch.nn.Module):
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
trunk.model.layers[0] = DW(trunk.model.layers[0], towers[0][0], towers[0][1])
trunk.model.layers[1] = DW(trunk.model.layers[1], towers[1][0], towers[1][1])
head = MarkHead(1280, 9).cuda(); head.eval()
head.load_state_dict(load_file(str((bdir / "head.safetensors") if (bdir / "head.safetensors").exists() else (ROOT / "runs/mex/mu3_joint/head.safetensors"))))
mods = [head] + [b for pr in towers for b in pr]
for m in mods:
    for pp in m.parameters():
        pp.requires_grad_(False)
its = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
dig = torch.tensor(sorted(int(v) for c, v in voc.vocab.items() if c.isdigit() or c in "+="), device="cuda")
@torch.no_grad()
def run(ids):
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i2 + 1]
    lg = trunk(input_ids=x).logits
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return lg, hs
ok1 = 0; n1 = 0
with torch.no_grad():
    for it in [i for i in its if i["task"] == "x1"]:
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        P = len(voc.encode(it["prompt"]))
        lg, hs = run(ids)
        for j in range(P, len(ids)):
            hv = hs[-1][0, j].float()
            cl = int(head(hv).argmax())
            pred = int(lg[0][j - 1].argmax()) if cl == 8 else int(voc.vocab[marks[cl]])
            ok1 += int(pred == ids[j])
            n1 += 1
ok2 = 0; n2 = 0
with torch.no_grad():
    for it in [i for i in its if i["task"] == "x2"]:
        ids = voc.encode(it["prompt"])[:seq]
        gold = int(voc.vocab[it["target"][0]])
        lg, hs = run(ids)
        p = int(dig[lg[0][len(ids) - 1][dig].argmax()])
        n2 += 1
        ok2 += int(p == gold)
ok3 = 0; n3 = 0
x3h = torch.nn.Sequential(torch.nn.Linear(1280, 1280), torch.nn.Tanh(), torch.nn.Linear(1280, 2)).cuda()
x3h.eval()
sd3 = torch.load(str(ROOT / "runs/mex/mu3_router/x3_head_armed.pt"), weights_only=True)
map3 = {"f1.weight": "0.weight", "f1.bias": "0.bias", "f2.weight": "2.weight", "f2.bias": "2.bias"}
w0 = torch.cat([torch.cat([sd3['f1.weight'], ], dim=0)] * 2, dim=0); w0 = torch.cat([w0] * 2, dim=1) * 0.5; b0 = torch.cat([sd3['f1.bias']] * 2, dim=0); w2 = torch.cat([sd3['f2.weight']] * 2, dim=1) * 0.5; x3h.load_state_dict({'0.weight': w0, '0.bias': b0, '2.weight': w2, '2.bias': sd3['f2.bias']})
for pp in x3h.parameters():
    pp.requires_grad_(False)
with torch.no_grad():
    for it in [i for i in its if i["task"] == "x3"]:
        ids = voc.encode(it["prompt"])[:seq]
        lg, hs = run(ids)
        hv = hs[-1][0, len(ids) - 1].float()
        p = int(x3h(hv).reshape(-1, 2).argmax(-1)[0])
        g = 1 if "bad" in it["target"] else 0
        n3 += 1
        ok3 += int(p == g)
sum = {"x1_composed_characc": ok1 / max(n1, 1), "x2_fill": [ok2, n2], "x3_acc": [ok3, n3], "x1_n": n1}
print(json.dumps(sum))
(ROOT / "runs/mex/mu3_tall/final/gates_e50.json").write_text(json.dumps(sum, indent=1), encoding="utf-8")
