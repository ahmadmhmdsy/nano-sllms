"""mex/scripts/train_mu3_lora_x2.py - E-52 phase C (x2 family): rank-4 q/v LoRA on layer2."""
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
from transformers import LlamaForCausalLM
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
voc = CharVocab()
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
for n2, p in trunk.named_parameters():
    p.requires_grad_(False)
class LoRA(nn.Module):
    def __init__(self, r, cu):
        super().__init__()
        self.A = nn.Linear(cu, r, bias=False)
        self.B = nn.Linear(r, cu, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
lq = LoRA(4, 640).to(device)
class LoRA2(nn.Module):
    def __init__(self, r, cu, co):
        super().__init__()
        self.A = nn.Linear(cu, r, bias=False)
        self.B = nn.Linear(r, co, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
lv = LoRA2(4, 640, 320).to(device)
orig_q = trunk.model.layers[2].self_attn.q_proj
orig_v = trunk.model.layers[2].self_attn.v_proj
class Loraized(nn.Module):
    def __init__(self, base, lora):
        super().__init__()
        self.base = base; self.lora = lora
    def forward(self, h):
        return self.base(h) + self.lora.B(self.lora.A(h))
trunk.model.layers[2].self_attn.q_proj = Loraized(orig_q, lq)
trunk.model.layers[2].self_attn.v_proj = Loraized(orig_v, lv)
trunk.eval()
xpool = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
its_x2 = [i for i in xpool if i["task"] == "x2"]
items_val = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
dig = torch.tensor(sorted(int(v) for c, v in voc.vocab.items() if c.isdigit() or c in "+="), device=device)
def disarm():
    for pr in towers:
        for b in pr:
            b._strength = 0.0; b._hold_kv = None
def forward_ids(ids, grad=True):
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    disarm()
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, pr in enumerate(towers):
        pr[0]._strength = 1.0; pr[0]._hold_kv = hs[i2 + 1].detach()
        pr[1]._strength = 1.0; pr[1]._hold_kv = hs[i2 + 1].detach()
    lg = trunk(input_ids=x).logits
    disarm()
    return lg, x
opt = torch.optim.AdamW(list(lq.parameters()) + list(lv.parameters()), lr=5e-4, weight_decay=0.1)
rng = np.random.default_rng(41)
for step in range(1, 1201):
    disarm()
    opt.zero_grad()
    it = its_x2[int(rng.integers(0, len(its_x2)))]
    ids = voc.encode(it["prompt"] + it["target"])[:seq]
    lg, x = forward_ids(ids)
    loss = F.cross_entropy(lg[0][:len(ids) - 1].float(), torch.tensor(ids[1:], dtype=torch.long, device=device))
    loss.backward()
    opt.step()
    if step % 300 == 0:
        with torch.no_grad():
            ok = 0; n = 0
            for it2 in items_val:
                if it2["task"] != "x2" or not it2["target"] or it2["target"][0] not in voc.vocab:
                    continue
                ids2 = voc.encode(it2["prompt"])[:seq]
                lg2, _ = forward_ids(ids2)
                p = int(dig[lg2[0][len(ids2) - 1][dig].argmax()])
                ok += int(p == int(voc.vocab[it2["target"][0]]))
                n += 1
        print(f"step {step} loss {float(loss):.4f} x2fill {ok}/{n}", flush=True)
dst = ROOT / "runs/mex/mu3_lora/x2"
dst.mkdir(parents=True, exist_ok=True)
torch.save({"lq": lq.state_dict(), "lv": lv.state_dict()}, str(dst / "lora.pt"))
print("[saved]", dst)
