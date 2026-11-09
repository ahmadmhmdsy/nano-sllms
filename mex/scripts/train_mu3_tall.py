"""mex/scripts/train_mu3_tall.py - E-50: taller-trunk co-train (trunk + remounted bridges + head)."""
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
head = MarkHead(640, 9).to(device)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu3_joint/head.safetensors")))
@torch.no_grad()
def eva(ids):
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    trunk.eval()
    hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i2, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i2 + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i2 + 1]
    lg = trunk(input_ids=x).logits
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    return lg
items_val = [json.loads(l) for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][::2]
@torch.no_grad()
def x2_fill():
    dig = torch.tensor(sorted(int(v) for c, v in voc.vocab.items() if c.isdigit() or c in "+="), device=device)
    ok = 0; n = 0
    for it in items_val:
        if it["task"] != "x2" or not it["target"] or it["target"][0] not in voc.vocab:
            continue
        ids = voc.encode(it["prompt"])[:seq]
        gold = int(voc.vocab[it["target"][0]])
        lg = eva(ids)
        sub = lg[0, len(ids) - 1, dig].argmax()
        if int(dig[sub]) == gold:
            ok += 1
        n += 1
    return ok, n
@torch.no_grad()
def mixed_xce(sub=3):
    tot = 0.0; n = 0
    for it in items_val[::sub]:
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        if len(ids) < 4:
            continue
        lg = eva(ids)
        tot += float(F.cross_entropy(lg[0, :len(ids) - 1], torch.as_tensor(ids[1:], device=device)))
        n += 1
    return tot / max(n, 1)
dtrain = PackedDataset(sorted((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin")), 96)
blocks = []
for i in range(0, len(dtrain), max(1, len(dtrain) // 600)):
    blocks.append(dtrain[i])
blocks = [blk if isinstance(blk, (list, tuple)) else (blk["input_ids"] if isinstance(blk, dict) else list(blk)) for blk in blocks]
xpool = [json.loads(l) for l in (ROOT / "data/mex/mixed/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
trunk.train(); head.train()
opt = torch.optim.AdamW([
     {"params": [p for n2, p in trunk.named_parameters() if not n2.startswith("model.layers.")], "lr": 3e-5},
    {"params": list(towers[0][0].parameters()) + list(towers[0][1].parameters()) + list(towers[1][0].parameters()) + list(towers[1][1].parameters()), "lr": 1e-4},
    {"params": head.parameters(), "lr": 5e-4}],
    weight_decay=0.1)
rng = np.random.default_rng(41)
for step in range(1, 1501):
    for bm, bx in towers:
        bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
    opt.zero_grad()
    use_dia = (rng.random() < 0.5) or (step <= 200)
    if use_dia:
        blk = blocks[int(rng.integers(0, len(blocks)))]
        ids = list(blk)[:seq]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        trunk.train()
        lg = trunk(input_ids=x).logits[0, :-1]
        loss = F.cross_entropy(lg.float(), x[0, 1:])
    # dia uses its own loss above
    else:
        it = xpool[int(rng.integers(0, len(xpool)))]
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        trunk.train()
        for bm, bx in towers:
            bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
        for i2, (bm, bx) in enumerate(towers):
            bm._strength = 1.0; bm._hold_kv = hs[i2 + 1].detach()
            bx._strength = 1.0; bx._hold_kv = hs[i2 + 1].detach()
        lg = trunk(input_ids=x).logits[0].float()
        for bm, bx in towers:
            bm._strength = 0.0; bx._strength = 0.0; bm._hold_kv = None; bx._hold_kv = None
        loss = F.cross_entropy(lg[:len(ids) - 1].float(), torch.tensor(ids[1:], dtype=torch.long, device=device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(trunk.parameters()) + list(head.parameters()), 5.0)
    opt.step()
    if step % 150 == 0:
        res = x2_fill()
        print(f"step {step} loss {float(loss):.4f} x2fill {res[0]}/{res[1]}", flush=True)
dst = ROOT / "runs/mex/mu3_tall/final"
dst.mkdir(parents=True, exist_ok=True)
trunk.save_pretrained(str(dst))
res2 = x2_fill()
summary = {"x2_fill": res2, "mixed_xce": mixed_xce()}
print(json.dumps(summary))
(dst / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
torch.save({"bridge0": towers[0][0].state_dict(), "bridge1": towers[1][0].state_dict()}, str(dst / "bridge_w.pt"))
torch.save({"bridge0": towers[0][1].state_dict(), "bridge1": towers[1][1].state_dict()}, str(dst / "xbridge_w.pt"))
save_file(head.state_dict(), str(dst / "head.safetensors"))
print("[saved]", dst)
