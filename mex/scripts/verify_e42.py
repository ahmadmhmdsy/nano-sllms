"""mex/scripts/verify_e42.py - E-41 gate (c): mu2 stack reproduction (x tower separate).

E-42 gate (b): widened trunk (runs/mex/mu2_g4_init) + widened E-37a bridges
(runs/mex/mu2_g4_init/bridge_w{0,1}.pt) + widened E-39a head (head_wide.safetensors)
must reproduce the E-39a composed readout EXACTLY: 0.6533 all-pos / 0.7984 mark-pos.
"""
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
from mex.scripts.train_mu2_g5_bridge import Bridge, WrappedLayer

device = "cuda" if torch.cuda.is_available() else "cpu"
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g4_init")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu2_g4_init") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]

NONE, mid = 8, int(voc.vocab["|"])
mark_ids_t = torch.tensor(mark_ids, device=device)

bridges = [Bridge(640, 8).to(device) for _ in range(2)]
for i, br in enumerate(bridges):
    sw = torch.load(str(ROOT / ("runs/mex/mu2_g4_init/bridge_w" + str(i) + ".pt")), weights_only=True)
    br.load_state_dict(sw["bridge"]); br._strength = 0.0; br._hold_kv = None
    trunk.model.layers[i] = WrappedLayer(trunk.model.layers[i], br)

head = MarkHead(640, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_g4_init/head_wide.safetensors")))
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
def run(ds, tag, bridged):
    ok = {"all": [0, 0], "markpos": [0, 0]}
    for b in range(0, len(ds) - 1, 8):
        k = min(8, len(ds) - b)
        clean = np.stack([ds[b + j]["input_ids"] for j in range(k)])
        corr = corrupt(clean)
        clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
        corr_t = torch.as_tensor(corr, dtype=torch.long, device=device)
        for br in bridges:
            br._strength = 0.0; br._hold_kv = None
        hs = trunk.model(input_ids=clean_t, output_hidden_states=True).hidden_states
        for i, br in enumerate(bridges):
            br._hold_kv = hs[i + 1] if bridged else None
            br._strength = 1.0 if bridged else 0.0
        ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        hh = trunk.model(input_ids=ids, output_hidden_states=True)[0]
        tl = trunk(input_ids=ids).logits[:, :-1]
        cls = head(hh[:, :-1]).argmax(-1)
        pred = torch.where(cls != NONE, mark_ids_t[cls.clamp(max=7)], tl.argmax(-1))
        truth = clean_t[:, 1:]
        marc = torch.isin(truth, mark_ids_t)
        ok["all"][0] += int((pred == truth).sum()); ok["all"][1] += int(pred.numel())
        ok["markpos"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos"][1] += int(marc.sum())
    res = {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
    print(tag + " " + json.dumps(res))
    return res

r = run(val, "E-42 widened remount", True)
gates = {
  "hygiene_allpos_eq_0.6533": "PASS" if abs(r["all"] - 0.6533) < 1e-4 else "FAIL",
  "hygiene_markpos_eq_0.7984": "PASS" if abs(r["markpos"] - 0.7984) < 1e-4 else "FAIL",
}
(ROOT / "runs/mex/mu2_g4_init/verify.json").write_text(json.dumps({**r, **gates}), encoding="utf-8")
print(json.dumps(gates))
