"""mex/scripts/eval_mu3_e44_cohab.py - E-44 gate (c): mu2 composed readout with BOTH towers live."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g4b_head import MarkHead
from mex.scripts.train_mu2_g5_bridge import Bridge
device = "cuda" if torch.cuda.is_available() else "cpu"
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE, mid = 8, int(voc.vocab["|"])
mark_ids_t = torch.tensor(mark_ids, device=device)

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


mu2sd = [torch.load(str(ROOT / ("runs/mex/mu3_g4/final/bridge_w" + str(i) + ".pt")), weights_only=True) for i in range(2)]
xtwsd = torch.load(str(ROOT / "runs/mex/mu3_xtower/xtower.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(640, 8).to(device); bm.load_state_dict(mu2sd[i]["bridge"]); bm._strength = 0.0; bm._hold_kv = None
    bx = Bridge(640, 8).to(device); bx.load_state_dict(xtwsd["bridge" + str(i)]); bx._strength = 0.0; bx._hold_kv = None
    towers.append((bm, bx))
    trunk.model.layers[i] = DualWrapped(trunk.model.layers[i], bm, bx)
head = MarkHead(640, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu3_g4/final/head_wide.safetensors")))
head = head.to(device).eval()
seq = 96
val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)
def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H:
            continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out
@torch.inference_mode()
def run(ds, tag, x_on, m_on):
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
        for bm, bx in towers:
            bm._strength = 1.0 if m_on else 0.0; bm._hold_kv = hs[id(towers)] if False else None
            bx._strength = 1.0 if x_on else 0.0; bx._hold_kv = None
        hs = trunk.model(input_ids=clean_t, output_hidden_states=True).hidden_states
        for i, (bm, bx) in enumerate(towers):
            bm._strength = 1.0 if m_on else 0.0; bm._hold_kv = hs[i + 1] if m_on else None
            bx._strength = 1.0 if x_on else 0.0; bx._hold_kv = hs[i + 1] if x_on else None
        hh = trunk.model(input_ids=corr_t, output_hidden_states=True)[0]
        tl = trunk(input_ids=corr_t).logits[:, :-1]
        cls = head(hh[:, :-1]).argmax(-1)
        pred = torch.where(cls != NONE, mark_ids_t[cls.clamp(max=7)], tl.argmax(-1))
        truth = clean_t[:, 1:]
        marc = torch.isin(truth, mark_ids_t)
        ok["all"][0] += int((pred == truth).sum()); ok["all"][1] += int(pred.numel())
        ok["markpos"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos"][1] += int(marc.sum())
    res = {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
    print(tag + " " + json.dumps(res))
    return res
r_m = run(val, "mu2-tower-only composed", False, True)
r_b = run(val, "BOTH towers live composed", True, True)
gate = "PASS" if r_b["markpos"] >= 0.78 else "FAIL"
print(json.dumps({"gate_c_cohab": gate, "mu2_only": r_m, "both_live": r_b,
                  "anchor_E43": {"all": 0.6564, "markpos": 0.8012}}))
