"""mex/scripts/train_mu3_xtower.py - E-44: train the widened x-tower on the settled mu3 trunk."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mex"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge, WrappedLayer
from src.mount import gate_strength
device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, WARMUP, HOLD, BATCH = 1000, 8e-5, 400, 600, 32
trunk = LlamaForCausalLM.from_pretrained(str(ROOT / "runs/mex/mu3_g4/final")).to(device).eval()
trunk.load_state_dict(load_file(str((ROOT / "runs/mex/mu3_g4/final") / "model.safetensors")), strict=True)
for p in trunk.parameters():
    p.requires_grad = False
voc = CharVocab()
pad = int(voc.vocab["<pad>"])
def load_jsonl(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]
def encode_item(it):
    return voc.encode(it["prompt"] + it["target"])[:96]
train = [t for t in (encode_item(it) for it in load_jsonl("data/mex/mixed/train.jsonl")) if len(t) >= 8]
val = [v for v in (encode_item(it) for it in load_jsonl("data/mex/mixed/val.jsonl")) if len(v) >= 8]
def make_batch(items, L=96):
    B = len(items)
    x = torch.full((B, L), pad, dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    for i, t in enumerate(items):
        t = t[:L]
        x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        m[i, :len(t)] = True
    return x.to(device), m.to(device)
bridges = [Bridge(640, 8).to(device) for _ in range(2)]
for i, br in enumerate(bridges):
    sw = torch.load(str(ROOT / ("runs/mex/mu3_xtower/bridge_w" + str(i) + ".pt")), weights_only=True)
    br.load_state_dict(sw["bridge"], strict=True)
    br._strength = 0.0; br._hold_kv = None
    trunk.model.layers[i] = WrappedLayer(trunk.model.layers[i], br)
x0, m0 = make_batch(val[:8])
with torch.inference_mode():
    for br in bridges:
        br._strength = 0.0; br._hold_kv = None
    l_off = trunk(input_ids=x0).logits
    l_chk = trunk(input_ids=x0).logits
iden = float((l_off - l_chk).abs().max())
print(f"[identity] max|dlogit| strength0 = {iden:.2e}", flush=True)
assert iden == 0.0
trainable = [pp for br in bridges for pp in br.parameters()]
print(f"[xtower] trainables {sum(pp.numel() for pp in trainable)}", flush=True)
opt = torch.optim.AdamW(trainable, lr=LR)
gen = np.random.default_rng(42)
@torch.inference_mode()
def ce_on(items, tag):
    tot, n = 0.0, 0
    for b in range(0, len(items), 32):
        x, m = make_batch(items[b:b + 32])
        for br in bridges:
            br._strength = 0.0; br._hold_kv = None
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
        for i, br in enumerate(bridges):
            br._hold_kv = hs[i + 1]; br._strength = 1.0
        logits = trunk(input_ids=x).logits
        nxt_real = m[:, 1:]
        lg = logits[:, :-1][nxt_real]
        tg = x[:, 1:][nxt_real]
        tot += F.cross_entropy(lg, tg, reduction="sum").item(); n += int(nxt_real.sum())
    print(f"{tag}: mixed CE {tot / n:.4f} (n tok {n})", flush=True)
    return tot / n
ce_off = ce_on(val, "trunk-off x-tower")
gen = np.random.default_rng(1234)
from src.mount import gate_strength
for step in range(1, STEPS + 1):
    s = gate_strength(step, warmup=WARMUP, hold=HOLD, anneal_end=STEPS)
    if s == 0.0:
        break
    for br in bridges:
        br._strength = s
    blocks = gen.integers(0, len(train), BATCH)
    x, m = make_batch([train[b] for b in blocks])
    for br in bridges:
        br._strength = 0.0; br._hold_kv = None
    with torch.no_grad():
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i, br in enumerate(bridges):
        br._hold_kv = hs[i + 1]; br._strength = s
    logits = trunk(input_ids=x).logits
    nxt_real = m[:, 1:]
    loss = F.cross_entropy(logits[:, :-1][nxt_real], x[:, 1:][nxt_real])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} strength {s:.3f} loss {loss.item():.4f}", flush=True)
dst = ROOT / "runs/mex/mu3_xtower"
dst.mkdir(parents=True, exist_ok=True)
torch.save({f"bridge{i}": br.state_dict() for i, br in enumerate(bridges)}, str(dst / "xtower.pt"))
ce_b = ce_on(val, "bridged x-tower final")
res = dict(phase="mu3_xtower_e44", ce_trunk_off=round(ce_off, 4), ce_xtower=round(ce_b, 4),
           rel_drop=round(1 - ce_b / ce_off, 4), identity_max_dlogit=iden,
           gate_a="PASS" if iden == 0.0 else "FAIL",
           gate_b="PASS" if ce_b < 0.95 * ce_off else "FAIL")
(dst / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("[final] " + json.dumps(res))
