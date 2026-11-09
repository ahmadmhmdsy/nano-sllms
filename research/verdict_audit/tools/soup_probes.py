"""Uniform weight soup of a probe trajectory. CPU-only, no GPU.

Reads gate_probe/weights_step*.pt ({model, step, config} wrappers), averages
every tensor in fp32, casts back to the members' dtype, and saves the same
wrapper shape so the eval path loads it unchanged.
"""
import sys
import torch
from pathlib import Path


def probe_files(run):
    gp = Path("runs/diac") / run / "gate_probe"
    return sorted(gp.glob("weights_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))


def soup(paths, out_path, tag):
    if not paths:
        print("NO PROBES for", tag); return
    first = torch.load(paths[0], map_location="cpu")
    dt = {k: v.dtype for k, v in first["model"].items()}
    acc = None
    for p in paths:
        sd = torch.load(p, map_location="cpu")["model"]
        if acc is None:
            acc = {k: v.to(torch.float32).clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.to(torch.float32)
    n = len(paths)
    out_sd = {k: (acc[k] / n).to(dt[k]) for k in acc}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out_sd, "step": "soup_%s_%d" % (tag, n), "config": first.get("config")}, out_path)
    # how far the soup sits from its members (mean L2 over tensors, relative)
    import math
    dists = []
    for p in paths:
        sd = torch.load(p, map_location="cpu")["model"]
        num = sum(float(((out_sd[k].float() - sd[k].float()) ** 2).sum()) for k in out_sd)
        den = sum(float((sd[k].float() ** 2).sum()) for k in out_sd)
        dists.append(math.sqrt(num / max(den, 1e-12)))
    print("%-22s n=%2d  rel_L2_to_members min=%.4f mean=%.4f max=%.4f  -> %s" % (
        tag, n, min(dists), sum(dists) / len(dists), max(dists), out_path))
    return out_path


c4b = probe_files("e70_c4b_ctx256")
c4 = probe_files("e69_c4_ctx256")
print("c4b probes:", len(c4b), "c4 probes:", len(c4))
base = Path("runs/diac/e73_soup")
soup(c4b, base / "c4b_uniform" / "model.pt", "c4b12")
soup(c4, base / "c4_uniform" / "model.pt", "c412")
soup(c4b + c4, base / "all24_uniform" / "model.pt", "all24")
