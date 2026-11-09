"""mex/scripts/eval_mu3_e45_gates.py - E-45 gates (b),(c),(d) post joint co-train."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import os
import torch
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE, mid, pad, seq = 8, int(voc.vocab["|"]), int(voc.vocab["<pad>"]), 96
mark_ids_t = torch.tensor(mark_ids, device=device)
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / os.environ.get("DIA2_TRUNK", "runs/mex/dia2_init"))).to(device).eval()
trunk.load_state_dict(load_file(str(ROOT / os.environ.get("DIA2_TRUNK", "runs/mex/dia2_init") / "model.safetensors")), strict=True)
mub = torch.load(str(ROOT / "runs/mex/dia2_init/bridge_w.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/dia2_init/xbridge_w.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(1280, 16).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(1280, 16).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bm._hold_kv = None
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
head = MarkHead(1280, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/dia2_init/head.safetensors")))
head = head.to(device).eval()
for m in [trunk, head] + [b for pr in towers for b in pr]:
    m.eval()
    for pp in m.parameters():
        pp.requires_grad = False
dval = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)
def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H:
            continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out
@torch.inference_mode()
def engaged():
    out = trunk.model(input_ids=torch.zeros((1, seq), dtype=torch.long, device=device), output_hidden_states=True).hidden_states
    for i, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = out[i + 1]
        bx._strength = 1.0; bx._hold_kv = out[i + 1]
@torch.inference_mode()
def composed(ds):
    ok = {"all": [0, 0], "markpos": [0, 0]}
    for b in range(0, len(ds) - 1, 8):
        k = min(8, len(ds) - b)
        clean = np.stack([ds[b + j]["input_ids"] for j in range(k)])
        corr = corrupt(clean)
        clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
        corr_t = torch.as_tensor(corr, dtype=torch.long, device=device)
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
        hs = trunk.model(input_ids=clean_t, output_hidden_states=True).hidden_states
        for i, (bm, bx) in enumerate(towers):
            bm._strength = 1.0; bm._hold_kv = hs[i + 1]
            bx._strength = 1.0; bx._hold_kv = hs[i + 1]
        hh = trunk.model(input_ids=corr_t, output_hidden_states=True)[0]
        tl = trunk(input_ids=corr_t).logits[:, :-1]
        cls = head(hh[:, :-1]).argmax(-1)
        pred = torch.where(cls != NONE, mark_ids_t[cls.clamp(max=7)], tl.argmax(-1))
        truth = clean_t[:, 1:]
        marc = torch.isin(truth, mark_ids_t)
        ok["all"][0] += int((pred == truth).sum()); ok["all"][1] += int(pred.numel())
        ok["markpos"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos"][1] += int(marc.sum())
    return {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
@torch.inference_mode()
def mixed_ce():
    xval = []
    for l in (ROOT / "data/mex/mixed/val.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        it = json.loads(l)
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        if len(ids) >= 8:
            xval.append(ids)
    tot = n = 0
    for b in range(0, len(xval), 32):
        chunk = xval[b:b + 32]
        x = torch.full((len(chunk), seq), pad, dtype=torch.long)
        m = torch.zeros((len(chunk), seq), dtype=torch.bool)
        for i, t in enumerate(chunk):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long); m[i, :len(t)] = True
        x = x.to(device); m = m.to(device)
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
        for i, (bm, bx) in enumerate(towers):
            bm._strength = 1.0; bm._hold_kv = hs[i + 1]
            bx._strength = 1.0; bx._hold_kv = hs[i + 1]
        logits = trunk(input_ids=x).logits
        nxt = m[:, 1:]
        tot += F.cross_entropy(logits[:, :-1][nxt], x[:, 1:][nxt], reduction="sum").item()
        n += int(nxt.sum())
    return round(tot / n, 4)
r = composed(dval)
ce = mixed_ce()
gates = dict(gate_b=r["markpos"] >= 0.78, gate_c=ce <= 3.49, gate_d=r["all"] >= 0.6202)
print(json.dumps({"gates": gates, "composed": r, "mixed_ce": ce}))
(ROOT / "runs/mex/mu3_joint" / "gates.json").write_text(json.dumps({"gates": gates, "composed": r, "mixed_ce": ce}, indent=2), encoding="utf-8")
