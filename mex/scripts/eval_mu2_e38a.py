"""mex/scripts/eval_mu2_e38a.py — E-38a full-stack composed readout (no training).

E-35 rule (head mark class first, free trunk fallback) while the E-37a bridges
are LIVE at strength 1.0 with per-depth teacher-of-self clean KV. Bridges-off
branch re-verifies identity vs E-35's recorded readout.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "E:/python_projects/nano_SLMs")
ROOT = Path("E:/python_projects/nano_SLMs")
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g4b_head import MarkHead
from mex.scripts.train_mu2_g5_bridge import Bridge, WrappedLayer

device = "cuda" if torch.cuda.is_available() else "cpu"
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g3/final")).to(device).eval()
trunk_sd = load_file(str((ROOT / "runs/mex/mu2_g3/final") / "model.safetensors"))
trunk.load_state_dict(trunk_sd, strict=True)
for p in trunk.parameters():
    p.requires_grad = False

voc = CharVocab()
mark_ids = [int(voc.vocab[c]) for c in "ًٌٍَُِّّْ"]
NONE = 8
mid = int(voc.vocab["|"])

bridges = [Bridge(320, 4).to(device) for _ in range(2)]
sd = torch.load(str(ROOT / "runs/mex/mu2_g37a/bridge.pt"), weights_only=True)
for i, br in enumerate(bridges):
    br.load_state_dict(sd[f"bridge{i}"])
    br._strength = 0.0
    br._hold_kv = None
    trunk.model.layers[i] = WrappedLayer(trunk.model.layers[i], br)

head = MarkHead(320, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_g4b/head.safetensors")))
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

mark_ids_t = torch.tensor(mark_ids, device=device)

@torch.inference_mode()
def run(ds, tag, bridged):
    ok = {k: [0, 0] for k in ("all", "markpos", "nonmark")}
    for b in range(0, len(ds) - 1, 8):
        k = min(8, len(ds) - b)
        clean = np.stack([ds[b + j]["input_ids"] for j in range(k)])
        corr = corrupt(clean)
        clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
        corr_t = torch.as_tensor(corr, dtype=torch.long, device=device)
        hs = trunk.model(input_ids=clean_t, output_hidden_states=True).hidden_states
        for i, br in enumerate(bridges):
            br._hold_kv = hs[i + 1] if bridged else None
            br._strength = 1.0 if bridged else 0.0
        ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        hh = trunk.model(input_ids=ids, output_hidden_states=True)[0]
        tl = trunk(input_ids=ids).logits[:, :-1]
        cls = head(hh[:, :-1]).argmax(-1)
        pick = mark_ids_t[cls.clamp(max=7)]
        pred = torch.where(cls != NONE, pick, tl.argmax(-1))
        truth = clean_t[:, 1:]
        marc = torch.isin(truth, mark_ids_t)
        ok["all"][0] += int((pred == truth).sum()); ok["all"][1] += int(pred.numel())
        ok["markpos"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos"][1] += int(marc.sum())
        ok["nonmark"][0] += int((pred[~marc] == truth[~marc]).sum()); ok["nonmark"][1] += int((~marc).sum())
    res = {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
    print(tag, json.dumps(res))
    return res

res_on = run(val, "bridged+head", True)
res_off = run(val, "bridges-off(head only)", False)
gates = {
  "allpos_beats_E35_0.6196": "PASS" if res_on["all"] > 0.6196 else "FAIL",
  "markpos_beats_E35_0.7556": "PASS" if res_on["markpos"] > 0.7556 else "FAIL",
  "structural_identity_E35": "PASS" if abs(res_off["all"] - 0.6196) < 1e-4 and abs(res_off["markpos"] - 0.7556) < 1e-4 else "FAIL",
}
out = ROOT / "runs/mex/mu2_e38a"
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(res_on | res_off | gates, indent=2), encoding="utf-8")
print(json.dumps(gates))
