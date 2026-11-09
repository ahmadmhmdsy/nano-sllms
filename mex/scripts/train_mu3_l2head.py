"""mex/scripts/train_mu3_l2head.py - E-52 phase B: MarkHead on new last hidden."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
device = "cuda"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import save_file
from transformers import LlamaForCausalLM
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
seq = 96
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_l2pre")).to(device)
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)]); bm._hold_kv = None
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)]); bx._hold_kv = None
    towers.append([bm, bx])
class DualWrapped(torch.nn.Module):
    def __init__(self, base, bm, bx):
        super().__init__()
        self.base = base; self.bm = bm; self.bx = bx
    def forward(self, *a, **kw):
        out = self.base(*a, **kw)
        y = out[0] if isinstance(out, tuple) else out
        if self.bm._hold_kv is not None and self.bm._strength > 0:
            y = self.bm(y, self.bm._hold_kv.to(y.dtype))
        if self.bx._hold_kv is not None and self.bx._strength > 0:
            y = self.bx(y, self.bx._hold_kv.to(y.dtype))
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
trunk.model.layers[0] = DualWrapped(trunk.model.layers[0], towers[0][0], towers[0][1])
trunk.model.layers[1] = DualWrapped(trunk.model.layers[1], towers[1][0], towers[1][1])
head = MarkHead(640, 9).to(device)
for n2, p in trunk.named_parameters():
    p.requires_grad_(False)
trunk.eval()
xpool = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
its_dia = [i for i in xpool if i["task"] == "x1"]
ai = torch.tensor([], dtype=torch.long, device=device)
MK = torch.zeros((9,), dtype=torch.float32)
nonex = 8
MAP = torch.tensor([9 if 0 else 0] * 0, device=device) if False else None
opt = None
items_all = its_dia
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.1)
rng = np.random.default_rng(41)
@torch.no_grad()
def disarm():
    for pr in towers:
        for b in pr:
            b._strength = 0.0; b._hold_kv = None
for step in range(1, 1501):
    disarm()
    opt.zero_grad()
    it = items_all[int(rng.integers(0, len(items_all)))]
    ids = voc.encode(it["prompt"] + it["target"])[:seq]
    P = len(voc.encode(it["prompt"]))
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, pr in enumerate(towers):
        pr[0]._strength = 1.0; pr[0]._hold_kv = hs[i2 + 1].detach()
        pr[1]._strength = 1.0; pr[1]._hold_kv = hs[i2 + 1].detach()
    lg = trunk(input_ids=x).logits
    disarm()
    hid = lg[0].float()
    losses = 0.0; cnt = 0
    for j in range(P, len(ids), 1):
    
        hv = hs[-1][0, j].float()
        gold = ids[j]
        cls_target = 8
        if int(gold) in mark_ids:
            if True:
                pass
                cls_target = int(mark_ids.index(int(gold)))
        tgt = torch.tensor(cls_target, device=device, dtype=torch.long)
        losses = losses + F.cross_entropy(head(hv).unsqueeze(0), tgt.unsqueeze(0))
        cnt = cnt + 1
    if cnt == 0:
        continue
    (losses / cnt).backward()
    opt.step()
    if step % 300 == 0:
        print(f"step {step} DCHECK {float(losses / cnt):.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_l2head"
dst.mkdir(parents=True, exist_ok=True)
save_file(head.state_dict(), str(dst / "head.safetensors"))
print("[saved]", dst)
