"""mex/scripts/eval_mu2_g4b2.py — E-35: corrected composition rule eval (no retraining).

Rule: per position, head mark class wins if != none; ELSE trunk argmax over the
FULL 97-token vocab (E-34's 'restricted to non-marks' branch mark-blinded the
trunk; realized in the E-34 post-mortem). Reuses the E-34 head as-is.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "E:/python_projects/nano_SLMs")
ROOT = Path("E:/python_projects/nano_SLMs")
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g4b_head import MarkHead

device = "cuda" if torch.cuda.is_available() else "cpu"
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g3/final")).to(device).eval()
voc = CharVocab()
mark_ids = [int(voc.vocab[c]) for c in "ًٌٍَُِّّْ"]
NONE = 8
mid = int(voc.vocab["|"])
trunk_sd = load_file(str((ROOT / "runs/mex/mu2_g3/final") / "model.safetensors"))
assert len(trunk_sd) > 0

head = MarkHead(320, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_g4b/head.safetensors")))
head = head.to(device).eval()

seq = 96
val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)

def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H: continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out

mark_ids_t = torch.tensor(mark_ids, device=device)

@torch.inference_mode()
def run(ds, tag):
    ok = {k: [0, 0] for k in ("all", "markpos_head", "markpos_trunk_only", "nonmark")}
    for b in range(0, len(ds) - 1, 8):
        k = min(8, len(ds) - b)
        clean = np.stack([ds[b + j]["input_ids"] for j in range(k)])
        corr = corrupt(clean)
        ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        hh = trunk.model(input_ids=ids)[0]
        tl = trunk(input_ids=ids).logits[:, :-1]
        cls = head(hh[:, :-1]).argmax(-1)
        pick = torch.tensor(mark_ids, device=device)[cls.clamp(max=7)]
        use = cls != NONE
        pred = torch.where(use, pick, tl.argmax(-1))
        pred_trunk = tl.argmax(-1)
        truth = torch.as_tensor(clean, dtype=torch.long, device=device)[:, 1:]
        marc = torch.isin(truth, mark_ids_t)
        ok["all"][0] += int((pred == truth).sum()); ok["all"][1] += int(pred.numel())
        ok["markpos_head"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos_head"][1] += int(marc.sum())
        ok["markpos_trunk_only"][0] += int((pred_trunk[marc] == truth[marc]).sum()); ok["markpos_trunk_only"][1] += int(marc.sum())
        ok["nonmark"][0] += int((pred[~marc] == truth[~marc]).sum()); ok["nonmark"][1] += int((~marc).sum())
    res = {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
    gates = {"rule_beats_E34": "PASS" if res["all"] > 0.36 else "FAIL",
             "markacc_at_or_above_trunk_only": "PASS" if res["markpos_head"] >= res["markpos_trunk_only"] else "FAIL"}
    out = ROOT / "runs/mex/mu2_g4b2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_rule.json").write_text(json.dumps({"res": res, "gates": gates}, indent=2), encoding="utf-8")
    print(json.dumps(res))
    print(json.dumps(gates))
run(val, "E35")
