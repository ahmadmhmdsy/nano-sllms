"""mex/scripts/widen_mu3_xtower.py - E-44: widen the E-41 x-tower to the 640 trunk."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mex"))
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.widen_mu2_g4_mounts import widen_bridge_state
dst = ROOT / "runs/mex/mu3_xtower"
dst.mkdir(parents=True, exist_ok=True)
sd = torch.load(str(ROOT / "runs/mex/mu2_e41/xtower.pt"), weights_only=True)
for i in range(2):
    sb = widen_bridge_state(sd["bridge" + str(i)], 640)
    b = Bridge(640, 8)
    b.load_state_dict(sb, strict=True)
    torch.save({"bridge": sb}, str(dst / ("bridge_w" + str(i) + ".pt")))
    print("x-bridge", i, "->", {k: tuple(v.shape) for k, v in sb.items()})
print("xtower widened OK")
