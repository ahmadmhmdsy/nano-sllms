"""M1: training-VRAM breakdown at pilot-scale proxy (12L d768 GQA, AdamW, fp16 autocast).
ctx in {512, 1024, 2048}, batch 2, 3 warmup steps; peak allocated decomposed into
params / fp32 grads / AdamW states / activations residual (training does not
cache KV; analytic inference KV bytes computed separately).
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
from bench_common import TOK, guard, load_tokens

CFG = dict(vocab=32768, d=768, layers=12, n_heads=12, kv_heads=4, ffn=2048)
BS = 2
OUTP = r"E:/python_projects/nano_SLMs/research/csa2_ced_tests/vram_profile.json"


def one_step(ctx):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = MiniLM(borrow_mode=None, device="cuda", **CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, CFG["vocab"], (BS, ctx), device="cuda")
    y = torch.randint(0, CFG["vocab"], (BS, ctx), device="cuda")
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    pparam = sum(p.numel() for p in model.parameters()) * 4 / 2**30  # fp32 master
    grads = pparam
    adamw = 2 * pparam
    del model, opt, x, y, lg, loss
    torch.cuda.empty_cache()
    resid = peak - pparam - grads - adamw
    return {"ctx": ctx, "batch": BS, "peak_step_gb": round(peak, 3),
            "params_fp32_gb": round(pparam, 3),
            "grads_fp32_gb": round(grads, 3),
            "adamw_m_v_gb": round(adamw, 3),
            "activations_residual_gb": round(resid, 3),
            "residual_pct": round(100 * resid / peak, 1),
            "kv_cmp": round(2 * CFG["layers"] * CFG["kv_heads"] *
                            (CFG["d"] // CFG["n_heads"]) * ctx * 2 / 2**30, 4)}


def prefill_tok_s(ctx, tok, steps=20):
    model = MiniLM(borrow_mode=None, device="cuda", **CFG)
    model.eval()
    xb = torch.from_numpy(np.stack(
        [np.asarray(tok[i:i + ctx], np.int64) for i in range(BS)])).cuda()
    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(xb)
    for _ in range(3):
        fwd()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        fwd()
    torch.cuda.synchronize()
    tps = round(BS * ctx * steps / (time.time() - t0))
    del model, xb
    torch.cuda.empty_cache()
    return tps


if __name__ == "__main__":
    guard()
    tok = load_tokens(TOK)
    out = {"config": CFG, "micro_batch": BS,
           "steps": [one_step(c) for c in (512, 1024, 2048)]}
    for c in (512, 1024, 2048):
        try:
            out[f"prefill_tok_s_ctx{c}"] = prefill_tok_s(c, tok)
        except Exception as e:
            out[f"prefill_tok_s_ctx{c}"] = f"ERR {e}"
    print(json.dumps(out, indent=2), flush=True)
    with open(OUTP, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", flush=True)
