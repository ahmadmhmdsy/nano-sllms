"""mex/scripts/demo_mu3_e46.py - E-46: live decode demo, one prompt per task family, live vs off."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(ROOT) if False else "")
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
device = "cuda"
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE = 8
mark_ids_t = torch.tensor(mark_ids, device=device)
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
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
def engage(x_clean):
    for bm, bx in towers:
        bm._strength = 0.0; bm._hold_kv = None
        bx._strength = 0.0; bx._hold_kv = None
    with torch.no_grad():
        hs = trunk.model(input_ids=x_clean, output_hidden_states=True).hidden_states
    for i, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i + 1]
def step(ids, live, ctx=96):
    x = torch.as_tensor(ids[-ctx:], dtype=torch.long, device=device).unsqueeze(0)
    if live:
        engage(x)
    else:
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
    hh = trunk.model(input_ids=x, output_hidden_states=True)[0]
    lg = trunk(input_ids=x).logits[0, -1]
    cls = int(head(hh[:, -1:]).argmax(-1)[0, 0])
    if live and cls != NONE:
        return int(mark_ids[cls]), float(lg.max())
    return int(lg.argmax()), float(lg.max())
def complete(prompt, n_new=14, live=True):
    ids = [int(v) for v in voc.encode(prompt)]
    for _ in range(n_new):
        nid, _c = step(ids, live)
        ids.append(nid)
    return voc.decode(ids)
demos = []
count = {}
items = []
for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines():
    l = l.strip()
    if not l:
        continue
    it = json.loads(l)
    items.append(it)
seen = set()
demos = []
for it in items:
    seen.add(it["task"])
    if sum(1 for d in demos if d["task"] == it["task"]) >= 2:
        continue
    live = complete(it["prompt"], 12, True)
    off = complete(it["prompt"], 12, False)
    demos.append(dict(task=it["task"], prompt=it["prompt"], gold=it["target"], live=live, off=off))
out = json.dumps(demos, indent=1, ensure_ascii=False)
print(out)
(ROOT / "runs/mex/mu3_joint/demo_raw.json").write_text(out, encoding="utf-8")
