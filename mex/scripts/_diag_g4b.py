from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
import sys
sys.path.insert(0, "E:/python_projects/nano_SLMs")
ROOT = Path("E:/python_projects/nano_SLMs")
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab

device = "cuda"
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g3/final")).to(device).eval()
voc = CharVocab()
mark_ids = [int(voc.vocab[c]) for c in "ًٌٍَُِّّْ"]
NONE = 8
mid = int(voc.vocab["|"])
seq = 96
train = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)
to_class = torch.full((128,), NONE, dtype=torch.long)
for i, m in enumerate(mark_ids): to_class[m] = i
nonmark = torch.tensor([i for i in range(97) if i not in mark_ids], device=device)
mark_ids_t = torch.tensor(mark_ids, device=device)

from mex.scripts.train_mu2_g4b_head import MarkHead
head = MarkHead(320, 9)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_g4b/head.safetensors")))
head = head.to(device).eval()

def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H: continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out

@torch.inference_mode()
def diag(ds, tag):
    ok = {k: [0,0] for k in ("markpos_head","markpos_trunk","nonmark_head","nonmark_trunk")}
    for b in range(0, len(ds)-1, 8):
        k = min(8, len(ds)-b)
        clean = np.stack([ds[b+j]["input_ids"] for j in range(k)])
        corr = corrupt(clean)
        ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        hh = trunk.model(input_ids=ids)[0]
        tl = trunk(input_ids=ids).logits[:, :-1]
        bis = torch.arange(97, device=device)
        bc = tl.masked_fill(torch.isin(bis, nonmark).view(1,1,-1)==False, float('-inf'))
        # heads branch
        cls = head(hh[:, :-1]).argmax(-1)
        pick = torch.tensor(mark_ids, device=device)[cls.clamp(max=7)]
        use = cls != NONE
        pred = torch.where(use, pick, bc.argmax(-1))
        truth = torch.as_tensor(clean, device=device)[:, 1:]
        marc = torch.isin(truth, torch.tensor(mark_ids, device=device))
        ok["markpos_head"][0] += int((pred[marc] == truth[marc]).sum()); ok["markpos_head"][1] += int(marc.sum())
        # trunk-only at mark positions (plain trunk argmax incl marks)
        pr2 = tl.argmax(-1)
        ok["markpos_trunk"][0] += int((pr2[marc] == truth[marc]).sum()); ok["markpos_trunk"][1] += int(marc.sum())
        ok["nonmark_head"][0] += int((pred[~marc] == truth[~marc]).sum()); ok["nonmark_head"][1] += int((~marc).sum())
        ok["nonmark_trunk"][0] += int((pr2[~marc] == truth[~marc]).sum()); ok["nonmark_trunk"][1] += int((~marc).sum())
        # head usage rate
        ok["markpos_head"][1] += 0
    for k, v in ok.items():
        print(tag, k, v[0], "/", v[1], "=", round(v[0]/max(v[1],1), 4))
diag(val, "val")
