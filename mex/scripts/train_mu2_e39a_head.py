"""mex/scripts/train_mu2_e39a_head.py — E-39a: bridge-aware head finetune (no new layers).

Warm-starts the E-34 MarkHead and finetunes it on BRIDGED hidden states
(trunk + E-37a bridges frozen, strength 1.0, per-depth clean teacher KV).
Composed-readout eval identical to E-38a.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file, save_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g4b_head import MarkHead
from mex.scripts.train_mu2_g5_bridge import Bridge, WrappedLayer

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu2_g3/final")).to(device).eval()
trunk_sd = load_file(str((ROOT / "runs/mex/mu2_g3/final") / "model.safetensors"))
trunk.load_state_dict(trunk_sd, strict=True)
for p in trunk.parameters():
    p.requires_grad = False

voc = CharVocab()
mark_ids = [int(voc.vocab[c]) for c in "ًٌٍَُِّّْ"]
NONE, mid = 8, int(voc.vocab["|"])
to_class = torch.full((128,), NONE, dtype=torch.long)
for i, m in enumerate(mark_ids):
    to_class[m] = i
to_class_dev = to_class.to(device)
mark_ids_t = torch.tensor(mark_ids, device=device)

bridges = [Bridge(320, 4).to(device) for _ in range(2)]
sd = torch.load(str(ROOT / "runs/mex/mu2_g37a/bridge.pt"), weights_only=True)
for i, br in enumerate(bridges):
    br.load_state_dict(sd[f"bridge{i}"])
    br._strength = 0.0
    br._hold_kv = None
    trunk.model.layers[i] = WrappedLayer(trunk.model.layers[i], br)

seq = 96
train = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)

def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H:
            continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out

head = MarkHead(320, 9).to(device)
head.load_state_dict(load_file(str(ROOT / "runs/mex/mu2_g4b/head.safetensors")))
opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
gen = np.random.default_rng(42)
for step in range(args.steps):
    blocks = gen.integers(0, len(train), args.batch)
    clean = np.stack([train[b]["input_ids"] for b in blocks])
    corr = corrupt(clean)
    clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
    corr_t = torch.as_tensor(corr, dtype=torch.long, device=device)
    for br in bridges:
        br._strength = 0.0; br._hold_kv = None
    with torch.no_grad():

        hs = trunk.model(input_ids=clean_t, output_hidden_states=True).hidden_states
    for i, br in enumerate(bridges):
        br._hold_kv = hs[i + 1]; br._strength = 1.0
    with torch.no_grad():
        hh = trunk.model(input_ids=corr_t, output_hidden_states=True)[0]
    nxt = clean_t[:, 1:]
    is_mark = torch.isin(nxt, mark_ids_t)
    tgt = torch.where(is_mark, to_class_dev[nxt], torch.full_like(nxt, NONE))
    tgt = torch.where(nxt >= 0, to_class_dev[nxt], torch.full_like(nxt, NONE))
    loss = F.cross_entropy(head(hh[:, :-1]).reshape(-1, 9), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)

dst = ROOT / "runs/mex/mu2_e39a"
dst.mkdir(parents=True, exist_ok=True)
save_file({n: p.detach().cpu() for n, p in head.named_parameters()}, str(dst / "head.safetensors"))
head.eval()

@torch.inference_mode()
def run(ds, tag, bridged):
    ok = {k: [0, 0] for k in ("all", "markpos", "nonmark")}
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
        ok["nonmark"][0] += int((pred[~marc] == truth[~marc]).sum()); ok["nonmark"][1] += int((~marc).sum())
    res = {k: round(v[0] / max(v[1], 1), 4) for k, v in ok.items()}
    print(tag + " " + json.dumps(res))
    return res

r_on = run(val, "finetuned-head+bridges", True)
gates = {
  "g_a_markpos_gt_0.7862": "PASS" if r_on["markpos"] > 0.7862 else "FAIL",
  "g_b_allpos_gt_0.6485": "PASS" if r_on["all"] > 0.6485 else "FAIL",
}
dst_summary = {"phase": "mu2_e39a", **r_on, **gates, "steps": args.steps, "lr": args.lr}
(dst / "train_summary.json").write_text(json.dumps(dst_summary, indent=2), encoding="utf-8")
print(json.dumps(gates))
