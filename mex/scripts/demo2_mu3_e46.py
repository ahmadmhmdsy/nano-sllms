"""mex/scripts/demo2_mu3_e46.py - E-46 (c): per-family fill accuracy at the | position."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
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
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bx._strength = 0.0
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
items = []
for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        items.append(json.loads(l))
@torch.inference_mode()
def fill(live):
    fam = {}
    for it in items:
        ids = voc.encode(it["prompt"])
        gold = voc.encode(it["target"])
        if not gold or len(ids) >= 94:
            continue
        x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
        if live:
            hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
            for i, (bm, bx) in enumerate(towers):
                bm._strength = 1.0; bm._hold_kv = hs[i + 1]
                bx._strength = 1.0; bx._hold_kv = hs[i + 1]
        lg = trunk(input_ids=x).logits[0, -1].float()
        hh = trunk.model(input_ids=x, output_hidden_states=True)[0][:, -1:]
        cls = int(head(hh).argmax(-1)[0, 0])
        pl = int(mark_ids[cls]) if (live and cls != NONE) else int(lg.argmax())
        po = int(lg.argmax())
        d = fam.setdefault(it["task"], [0, 0])
        d[1] += 1
        if pl == gold[0]:
            d[0] += 1
    return fam
f_no = fill(False)
f_yes = fill(True)
print("fill trunk-only :", json.dumps(f_no))
print("fill composed   :", json.dumps(f_yes))
import json as _j
(ROOT / "runs/mex/mu3_joint/e46_fill.json").write_text(_j.dumps({"fill_trunkonly": f_no, "fill_composed": f_yes}, indent=1), encoding="utf-8")
