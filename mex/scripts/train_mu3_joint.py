"""mex/scripts/train_mu3_joint.py - E-45: joint two-tower co-train on the settled mu3 trunk."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from src.mount import gate_strength
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
STEPS, BATCH, WARM, HOLD, END, LR = 1500, 32, 400, 600, 1500, 5e-5
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
to_class = torch.full((128,), 8, dtype=torch.long)
for i, m in enumerate(mark_ids):
    to_class[m] = i
to_class_dev = to_class.to(device)
mark_ids_t = torch.tensor(mark_ids, device=device)
mid = int(voc.vocab["|"])
pad = int(voc.vocab["<pad>"])
seq = 96
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
head = head.to(device)
dval = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)
x0 = torch.as_tensor(np.stack([dval[i]["input_ids"] for i in range(4)]), dtype=torch.long, device=device)
with torch.inference_mode():
    for bm, bx in towers:
        bm._strength = 0.0; bm._hold_kv = None
        bx._strength = 0.0; bx._hold_kv = None
    l_off = trunk(input_ids=x0).logits
    l_chk = trunk(input_ids=x0).logits
iden = float((l_off - l_chk).abs().max())
print(f"[identity] max|dlogit| both-off = {iden:.2e}", flush=True)
assert iden == 0.0
opt = torch.optim.AdamW([
    {"params": [pp for bm, bx in towers for pp in bm.parameters()], "lr": LR},
    {"params": [pp for bm, bx in towers for pp in bx.parameters()], "lr": LR},
    {"params": list(head.parameters()), "lr": 1e-3},
], weight_decay=0.1)
dtrain = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
xval = []
xtrain = []
for name, out in (("val", xval), ("train", xtrain)):
    for l in (ROOT / ("data/mex/mixed/" + name + ".jsonl")).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        it = json.loads(l)
        ids = voc.encode(it["prompt"] + it["target"])[:seq]
        if len(ids) >= 8:
            out.append(ids)
def xce(armed, tag):
    tot, n = 0.0, 0
    with torch.no_grad(), torch.inference_mode():
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
                if armed:
                    bm._strength = 1.0; bm._hold_kv = hs[i + 1]
                    bx._strength = 1.0; bx._hold_kv = hs[i + 1]
            logits = trunk(input_ids=x).logits
            nxt = m[:, 1:]
            tot += F.cross_entropy(logits[:, :-1][nxt], x[:, 1:][nxt], reduction="sum").item()
            n += int(nxt.sum())
    print(f"{tag}: {tot / n:.4f} (n {n})", flush=True)
    return tot / n
ce_off_x = xce(False, "both-off mixed x CE")
def corrupt(blocks, H=3):
    out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
    for r in range(out.shape[0]):
        if (r % H) == 3 % H:
            continue
        ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
    return out
gen = np.random.default_rng(1234)
for step in range(1, STEPS + 1):
    s = gate_strength(step, warmup=WARM, hold=HOLD, anneal_end=END)
    if s == 0.0:
        break
    for bm, bx in towers:
        bm._strength = s; bx._strength = s
    use_dia = (gen.random() < 0.5)
    if use_dia:
        b0 = int(gen.integers(0, len(dtrain) - BATCH))
        clean = np.stack([dtrain[b0 + j]["input_ids"] for j in range(BATCH)])
        x = torch.as_tensor(corrupt(clean), dtype=torch.long, device=device)
        lab = torch.as_tensor(clean, dtype=torch.long, device=device)
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
        for i, (bm, bx) in enumerate(towers):
            bm._strength = s; bm._hold_kv = hs[i + 1]
            bx._strength = s; bx._hold_kv = hs[i + 1]
        logits = trunk(input_ids=x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1))
    else:
        blocks = gen.integers(0, len(xtrain), BATCH)
        chunk = [xtrain[b] for b in blocks]
        x = torch.full((BATCH, seq), pad, dtype=torch.long)
        m = torch.zeros((BATCH, seq), dtype=torch.bool)
        for i, t in enumerate(chunk):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long); m[i, :len(t)] = True
        x = x.to(device); m = m.to(device)
        for bm, bx in towers:
            bm._strength = 0.0; bm._hold_kv = None
            bx._strength = 0.0; bx._hold_kv = None
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
        for i, (bm, bx) in enumerate(towers):
            bm._strength = s; bm._hold_kv = hs[i + 1]
            bx._strength = s; bx._hold_kv = hs[i + 1]
        logits = trunk(input_ids=x).logits
        nxt = m[:, 1:]
        loss = F.cross_entropy(logits[:, :-1][nxt], x[:, 1:][nxt])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} s={s:.3f} dia={use_dia} loss {loss.item():.4f}", flush=True)
    if step >= END:
        break
dst = ROOT / "runs/mex/mu3_joint"
dst.mkdir(parents=True, exist_ok=True)
torch.save({f"bridge{i}": towers[i][0].state_dict() for i in range(2)}, str(dst / "mu2_bridges.pt"))
torch.save({f"bridge{i}": towers[i][1].state_dict() for i in range(2)}, str(dst / "x_bridges.pt"))
import safetensors.torch as st
st.save_file({k: v.detach().cpu() for k, v in head.state_dict().items()}, str(dst / "head.safetensors"))
ce_b_x = xce(True, "joint bridged mixed")
(dst / 'trainflag').write_text('done')
print("[train] done, mixed CE armed:", round(ce_b_x, 4))
