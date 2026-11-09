"""mex/scripts/widen_mu2_g4_mounts.py - E-42: widen TRAINED mounts to the 640 trunk.

Bridge(320,4) -> Bridge(640,8) (head_dim 80 const; q/kv/o out-cat, in-cat*0.5,
kv.bias cat, gate a kept). MarkHead lin[0] in-cat*0.5. Saves widened state.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead


def two_op(w_old, out_new, in_new):
    if w_old.dim() == 0:
        return w_old.clone()
    if w_old.dim() == 1:
        if 2 * w_old.shape[0] == out_new:
            return torch.cat([w_old] * 2, dim=0)
        if w_old.shape[0] == out_new:
            return w_old.clone()
        raise ValueError("1D shape mismatch " + str(tuple(w_old.shape)) + " -> " + str(out_new))
    w = w_old
    if out_new == 2 * w.shape[0] and in_new == 2 * w.shape[1]:
        w = torch.cat([w] * 2, dim=0)
        w = torch.cat([w] * 2, dim=1) * 0.5
    elif w.shape[0] == out_new and in_new == 2 * w.shape[1]:
        w = torch.cat([w] * 2, dim=1) * 0.5
    elif out_new == 2 * w.shape[0] and w.shape[1] == in_new:
        w = torch.cat([w] * 2, dim=0)
    return w


def widen_bridge_state(sd_old, dim_new):
    out = {}
    for key, w in sd_old.items():
        if key == "a":
            out[key] = w.clone()
            continue
        if key == "kv.bias":
            out[key] = torch.cat([w] * 2, dim=0)
            continue
        if "o." in key:
            out[key] = two_op(w, dim_new, dim_new)   # o: out per-head dup; in dup*0.5
            continue
        if key == "kv.weight":
            # kv packs k then v along the out axis (chunk(2)); widen each block
            k, v = w.chunk(2, dim=0)
            kc = torch.cat([k] * 2, dim=0)
            kc = torch.cat([kc] * 2, dim=1) * 0.5
            vc = torch.cat([v] * 2, dim=0)
            vc = torch.cat([vc] * 2, dim=1) * 0.5
            out[key] = torch.cat([kc, vc], dim=0)
        else:
            out[key] = two_op(w, dim_new, dim_new)       # q and o: both ops
    return out


def main():
    dst = ROOT / "runs/mex/mu2_g4_init"
    dst.mkdir(parents=True, exist_ok=True)
    sd37 = torch.load(str(ROOT / "runs/mex/mu2_g37a/bridge.pt"), weights_only=True)
    new = {}
    for i in range(2):
        sb = widen_bridge_state(sd37["bridge" + str(i)], 640)
        b = Bridge(640, 8)
        b.load_state_dict(sb, strict=True)
        torch.save({"bridge": sb}, str(dst / ("bridge_w" + str(i) + ".pt")))
        print("bridge", i, "widened to", {k: tuple(v.shape) for k, v in sb.items()})
    h = MarkHead(640, 9)
    src = {}
    from safetensors.torch import load_file
    hs = load_file(str(ROOT / "runs/mex/mu2_e39a/head.safetensors"))
    for k, w in hs.items():
        if k == "lin.0.weight":
            w = torch.cat([w] * 2, dim=1) * 0.5
        src[k] = w
    h.load_state_dict(src, strict=True)
    out_sd = {k: v for k, v in src.items()}
    import safetensors.torch as st
    st.save_file(out_sd, str(dst / "head_wide.safetensors"))
    print("head widened")


if __name__ == "__main__":
    main()
