"""M2/M3 micro-bench: dense GQA vs CED-lite KV-sharing upper half (V4.1 ideas).
Variants (identical data, steps, seed, schedule):
  A_baseline      all layers own K/V (dense GQA)
  B_shared        upper half reuses owner-layer K/V verbatim (YOCO/CSA2-reuse)
  C_proj          upper half CED eq.(1): per-layer K/V projections of H_mid
  D_shared_swa128 B + sliding window 128 on borrow layers (CSA2 local branch)
  E_proj_swa128   C + sliding window 128
Metrics: params, train loss curve, val loss, tokens/s, peak train VRAM.
SCRATCH ONLY - does not touch the pipeline. Run with .venv python from repo root.
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_model import MiniLM
from bench_common import (SEED, TOK, VAL, OUT, batch, guard, load_tokens,
                          make_opt, cosine, train_steps)

CFG = dict(vocab=32768, d=256, layers=6, n_heads=8, kv_heads=4,
           ctx=512, bs=8, steps=300, lr=4e-4)

VARIANTS = {
    "A_baseline": dict(borrow_mode=None, window=None),
    "B_shared": dict(borrow_mode="shared", window=None),
    "C_proj": dict(borrow_mode="proj", window=None),
    "D_shared_swa128": dict(borrow_mode="shared", window=128),
    "E_proj_swa128": dict(borrow_mode="proj", window=128),
}


def build(want_mode, want_window):
    torch.manual_seed(SEED)  # identical init across variants
    return MiniLM(device="cuda", vocab=CFG["vocab"], d=CFG["d"],
                  layers=CFG["layers"], n_heads=CFG["n_heads"],
                  kv_heads=CFG["kv_heads"], borrow_mode=want_mode,
                  window=want_window)


@torch.no_grad()
def val_loss(model, val, n=100):
    model.eval()
    ctx = CFG["ctx"]
    ts = []
    for i in range(n):
        o = ctx + (i * ctx * 3 + 17) % (len(val) - 2 * ctx)
        x = torch.from_numpy(np.asarray(val[o - ctx:o], np.int64).reshape(1, -1)).cuda()
        y = torch.from_numpy(np.asarray(val[o:o + ctx], np.int64).reshape(1, -1)).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        ts.append(F.cross_entropy(lg.view(-1, lg.size(-1)).float(),
                                  y.view(-1)).item())
    model.train()
    return sum(ts) / len(ts)


def main():
    guard()
    tok, vtok = load_tokens(TOK), load_tokens(VAL)
    results = {}
    for name, kw in VARIANTS.items():
        guard()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(kw["borrow_mode"], kw["window"])
        g = torch.Generator(device="cuda").manual_seed(SEED)
        results[name] = {"params_m": round(sum(p.numel() for p in
                                               model.parameters()) / 1e6, 2)}
        t0 = time.time()
        r = train_steps(model, tok, CFG["steps"], CFG["bs"], CFG["ctx"],
                        CFG["lr"], gen=g)
        r["val_loss"] = round(val_loss(model, vtok), 4)
        r["wall_s"] = round(time.time() - t0, 1)
        results[name].update(r)
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, "ced_lite_results.json"), "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
