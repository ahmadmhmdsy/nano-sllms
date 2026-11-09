import torch, sys, os
sys.stdout.reconfigure(encoding="utf-8")
for p in ["runs/diac/e67_c1_ft64_fullv3/gate_probe/best_gate_weights.pt",
          "runs/diac/e67_c1_ft64_fullv3/final/model.pt"]:
    if not os.path.exists(p):
        print("MISSING", p); continue
    o = torch.load(p, map_location="cpu", weights_only=False)
    print("==", p, round(os.path.getsize(p) / 1e6, 2), "MB")
    if isinstance(o, dict):
        ks = list(o.keys())
        print("  top keys:", ks[:8], "total", len(ks))
        sub = o.get("model") or o.get("state_dict")
        if sub is not None:
            print("  NESTED tensors:", len(sub), "| first:", list(sub)[:3])
            print("  shapes:", [tuple(v.shape) for v in list(sub.values())[:3]])
        else:
            tensors = [v for v in o.values() if hasattr(v, "shape")]
            print("  RAW state_dict; tensor count", len(tensors))
            print("  shapes:", [tuple(v.shape) for v in tensors[:3]])
    else:
        print("  type", type(o))
