"""mex/scripts/train_mu3_xfin.py - E-51: taller-trunk x-stream co-train with fresh head (3000 steps)."""
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
from safetensors.torch import load_file, save_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE = 8; seq = 96; pad = int(voc.vocab["<pad>"])
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_tall/init")).to(device)
mub = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._hold_kv = None; bx._hold_kv = None
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
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu3_joint/head.safetensors")))
@torch.no_grad()
def disarm():
    for pr in towers:
        for b in pr:
            b._strength = 0.0; b._hold_kv = None
dtrain = PackedDataset(sorted((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin")), 96)
blocks = [dtrain[i]["input_ids"] for i in range(0, len(dtrain), max(1, len(dtrain) // 800))]
xpool = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
items_val = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][::2]
dig_ids = sorted(int(v) for c, v in voc.vocab.items() if c.isdigit() or c in "+=")
dig = torch.tensor(dig_ids, device=device)
def eva_train(ids):
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    disarm()
    trunk.train()
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, pr in enumerate(towers):
        pr[0]._strength = 1.0; pr[0]._hold_kv = hs[i2 + 1].detach()
        pr[1]._strength = 1.0; pr[1]._hold_kv = hs[i2 + 1].detach()
    lg = trunk(input_ids=x).logits
    disarm()
    return lg, hs
@torch.no_grad()
def eva_eval(ids):  # eval twin
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    disarm()
    trunk.eval()
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, pr in enumerate(towers):
        pr[0]._strength = 1.0; pr[0]._hold_kv = hs[i2 + 1]
        pr[1]._strength = 1.0; pr[1]._hold_kv = hs[i2 + 1]
    lg = trunk(input_ids=x).logits
    disarm()
    return lg, hs
@torch.no_grad()
def x2_fill():
    ok = 0; n = 0
    for it in items_val:
        if it["task"] != "x2" or not it["target"] or it["target"][0] not in voc.vocab:
            continue
        ids = voc.encode(it["prompt"])[:seq]
        gold = int(voc.vocab[it["target"][0]])
        lg, hs = eva_train(ids)
        p = int(dig[lg[0][len(ids) - 1][dig].argmax()])
        ok += int(p == gold); n += 1
    return ok, n
M2I = torch.tensor([0] + [640] * 0, device=device)
trunk.train(); head.train()
opt = torch.optim.AdamW([
    {"params": [p for n2, p in trunk.named_parameters() if not n2.startswith("model.layers.0.") and not n2.startswith("model.layers.1.")], "lr": 5e-5},
    {"params": head.parameters(), "lr": 1e-3},
    {"params": list(towers[0][0].parameters()) + list(towers[0][1].parameters()) + list(towers[1][0].parameters()) + list(towers[1][1].parameters()), "lr": 2e-4}],
    weight_decay=0.1)
rng = np.random.default_rng(41)
dia_ids_set = set(mark_ids)
for step in range(1, 3001):
    disarm()
    opt.zero_grad()
    use_dia = rng.random() < 0.6
    if use_dia:
        blk = list(blocks[int(rng.integers(0, len(blocks)))])[:seq]
        x = torch.tensor(blk, dtype=torch.long, device=device).unsqueeze(0)
        lg = trunk(input_ids=x).logits[0].float()
        loss = F.cross_entropy(lg[:-1], x[0, 1:])
    else:
        it = xpool[int(rng.integers(0, len(xpool)))]
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        lg, hs = eva_train(ids)
        loss = F.cross_entropy(lg[0][:len(ids) - 1].float(), torch.tensor(ids[1:], dtype=torch.long, device=device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(trunk.parameters()) + list(head.parameters()), 5.0)
    opt.step()
    if step % 250 == 0:
        r = x2_fill()
        print(f"step {step} loss {float(loss):.4f} x2fill {r[0]}/{r[1]}", flush=True)
dst = ROOT / "runs/mex/mu3_xfin"
dst.mkdir(parents=True, exist_ok=True)
sum = {}
trunk.save_pretrained(str(dst))
torch.save({"bridge0": towers[0][0].state_dict(), "bridge1": towers[1][0].state_dict()}, str(dst / "bridge_w.pt"))
torch.save({"bridge0": towers[0][1].state_dict(), "bridge1": towers[1][1].state_dict()}, str(dst / "xbridge_w.pt"))
save_file(head.state_dict(), str(dst / "head.safetensors"))
r2 = x2_fill()
lgv = 0.0; nv = 0
with torch.no_grad():
    for it in items_val[::3]:
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        if len(ids) < 4:
            continue
        lgv, hsv = eva_eval(ids)
        lgv += float(F.cross_entropy(lgv[0][:len(ids) - 1].float(), torch.tensor(ids[1:], dtype=torch.long, device=device)))
        nv += 1
sum = {"x2_fill": r2, "mixed_xce": lgv / max(nv, 1)}
print(json.dumps(sum))
(dst / "summary.json").write_text(json.dumps(sum, indent=1), encoding="utf-8")
print("[saved]", dst)
