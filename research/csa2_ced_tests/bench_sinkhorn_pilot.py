"""M7: P-scale A/B of the Sinkhorn embedding rebalance (E-42 follow-up).
Pilot-proxy: 12L d768 GQA, bs 2, ctx 512, real pilot tokens, fp32, 300 steps.
Arms: adamw (control) vs sinkhorn_rate 0.1.
"""
import json, sys, os
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import SEED, TOK, OUT, cosine, load_tokens, make_opt, guard
from mini_model import MiniLM
from bench_extra import sinkhorn_rebalance

STEPS, BS, CTX, LR, WARM = 300, 2, 512, 4e-4, 25


def run(name, sink_rate=None):
    guard()
    torch.manual_seed(SEED)
    model = MiniLM(vocab=32768, d=768, layers=12, n_heads=12, kv_heads=4, device="cuda")
    opt = make_opt(model.parameters(), LR)
    tok = load_tokens(TOK)
    max_off = len(tok) - CTX - 1
    losses = []
    gc = torch.Generator().manual_seed(SEED + 1)
    for step in range(STEPS):
        for grp in opt.param_groups: grp["lr"] = cosine(step, STEPS, LR, warm=WARM)
        offs = torch.randint(0, max_off, (BS,), generator=gc).tolist()
        x = torch.from_numpy(np.stack([np.asarray(tok[o:o+CTX]) for o in offs]).astype(np.int64)).cuda()
        y = torch.from_numpy(np.stack([np.asarray(tok[o+1:o+1+CTX]) for o in offs]).astype(np.int64)).cuda()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 32768).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sink_rate: sinkhorn_rebalance(model, sink_rate)
        losses.append(loss.item())
        if step == 0: print(name, "step0", round(loss.item(), 3), flush=True)
    out = {"variant": name, "loss_last50": round(sum(losses[-50:]) / 50, 4),
           "loss_curve": [round(sum(losses[i:i+25]) / 25, 4) for i in range(0, STEPS, 25)]}
    print(out, flush=True)
    del model; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    res = [run("m7_control"), run("m7_sinkhorn_0.1", 0.1)]
    json.dump(res, open(os.path.join(OUT, "sinkhorn_pilot_ab.json"), "w"), indent=2)
    print("DONE", flush=True)
