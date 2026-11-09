"""mex/scripts/widen_dia2_mounts.py - E-54a: widen mounts onto the 1280 trunk."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.widen_mu2_g4_mounts import widen_bridge_state, two_op
DIM = 1280
dst = ROOT / "runs/mex/dia2_wide"
dst.mkdir(parents=True, exist_ok=True)
mu2b = torch.load(str(ROOT / "runs/mex/mu3_joint/mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(ROOT / "runs/mex/mu3_joint/x_bridges.pt"), weights_only=True)
out_mu2 = {}; out_x = {}
for i in range(2):
    nw = widen_bridge_state(mu2b["bridge" + str(i)], DIM)
    b = Bridge(DIM, 16)
    b.load_state_dict(nw, strict=True); out_mu2["bridge" + str(i)] = nw
    xw = widen_bridge_state(xb["bridge" + str(i)], DIM)
    bx = Bridge(DIM, 16)
    bx.load_state_dict(xw, strict=True)
    out_x["bridge" + str(i)] = xw
torch.save(out_mu2, str(dst / "mu2_bridges.pt"))
torch.save(out_x, str(dst / "x_bridges.pt"))
# MarkHead: lin.0 (128,640) input widened in-cat*0.5; lin.2 (9,128) untouched
hw = torch.load(str(ROOT / "runs/mex/mu3_joint/head.pt"), weights_only=True) if (ROOT / "runs/mex/mu3_joint/head.pt").exists() else None
from safetensors.torch import load_file, save_file
mh = load_file(str(ROOT / "runs/mex/mu3_joint/head.safetensors"))
w = mh["lin.0.weight"]
assert tuple(w.shape) == (128, 640), tuple(w.shape)
mh["lin.0.weight"] = torch.cat([w] * 2, dim=1) * 0.5
save_file(mh, str(dst / "head.safetensors"))
# router: f1 (640,640) both-ops; f2 (6,640) in-cat*0.5
r = torch.load(str(ROOT / "runs/mex/mu3_router/router.pt"), weights_only=True)
r["f1.weight"] = two_op(r["f1.weight"], DIM, DIM)
r["f1.bias"] = torch.cat([r["f1.bias"]] * 2, dim=0)
r["f2.weight"] = torch.cat([r["f2.weight"]] * 2, dim=1) * 0.5
torch.save(r, str(dst / "router.pt"))
print("dia2 mounts widened OK ->", dst)
