"""mex/scripts/hygiene_e41.py - E-41 gate (c): mu2 stack reproduction (x tower separate).

Re-runs the EXACT E-39a composed readout (E-37a bridges + E-39a head) on fresh
instances. The E-41 x tower lives only in a saved file: if mu2 reproduction is
exact, the towers are genuinely federated.
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
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g3/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu2_g3/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]

NONE, mid = 8, int(voc.vocab["|"])
mark_ids_t = torch.tensor(mark_ids, device=device)

bridges = [Bridge(320, 4).to(device) for _ in range(2)]
sd = torch.load(str(ROOT / "runs/mex/mu2_g37a/bridge.pt"), weights_only=True)
for i, br in enumerate(bridges):
    br.load_state_dict(sd["bridge" + str(i)]); br._strength = 0.0; br._hold_kv = None
    trunk.model.layers[i] = WrappedLayer(trunk.model.layers[i], br)

head = MarkHead(320, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_e39a/head.safetensors")))
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

r = run(val, "mu2-stack reproduction", True)
gates = {
  "hygiene_allpos_eq_0.6533": "PASS" if abs(r["all"] - 0.6533) < 1e-4 else "FAIL",
  "hygiene_markpos_eq_0.7984": "PASS" if abs(r["markpos"] - 0.7984) < 1e-4 else "FAIL",
}
(ROOT / "runs/mex/mu2_e41/hygiene.json").write_text(json.dumps({**r, **gates}), encoding="utf-8")
print(json.dumps(gates))
