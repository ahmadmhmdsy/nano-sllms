"""M5/M6: two more V4.1 ideas at nano scale (fp32, real pilot tokens, equal schedules).
M5 sinkhorn : light row-norm rebalance of the tied embedding each step (alpha
              small, geomean target) - proxy for the report's Sinkhorn-balanced
              embedding update; prevents rare-row undertraining.
M6 mtp      : extra tied-embedding head predicting token t+2 from h_t
              (aux CE weight 0.3), main loss unchanged at eval-time.
Arms: adamw (control), +sinkhorn, +mtp.
"""
import json, os, sys, math
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_model import MiniLM
from bench_common import SEED, TOK, OUT, batch, guard, cosine, load_tokens, make_opt

STEPS, BS, CTX, LR, WARM, D_MODEL = 300, 16, 256, 4e-4, 25, 192


def sinkhorn_rebalance(model, rate: float):
    """Push embedding+head row norms toward their geomean (cheap proxy for
    Sinkhorn-balanced updates). Only touches the shared tied weight.
    rate in (0,1): 0 = no-op, larger = harder rebalance."""
    E = model.embed.weight
    with torch.no_grad():
        norms = E.norm(dim=-1).clamp(min=1e-6)
        target = norms.log().mean().exp()
        scale = (target / norms).clamp(0.9, 1.1) ** rate
        E.mul_(scale.unsqueeze(-1))


def run(name, sinkhorn_rate=None, mtp_weight=None):
    guard()
    torch.manual_seed(SEED)
    tok = load_tokens(TOK)
    model = MiniLM(vocab=32768, d=D_MODEL, layers=4, n_heads=6, kv_heads=3,
                   device="cuda")
    mtp = None
    if mtp_weight:
        mtp = torch.nn.Linear(D_MODEL, 32768, bias=False).cuda()  # separate aux head
        torch.nn.init.normal_(mtp.weight, std=0.02)
        opts = [make_opt(list(model.parameters()) + list(mtp.parameters()), LR)]
    else:
        opts = [make_opt(model.parameters(), LR)]
    g = torch.Generator(device="cuda").manual_seed(SEED)
    torch.manual_seed(SEED)
    g_cpu = torch.Generator().manual_seed(SEED)
    losses, aux_losses, losses_main = [], [], []
    max_off = len(tok) - CTX - 1
    for step in range(STEPS):
        lr_s = cosine(step, STEPS, LR, warm=WARM)
        for o in opts:
            for grp in o.param_groups:
                grp["lr"] = lr_s
        offs = torch.randint(0, max_off, (BS,), generator=g_cpu).tolist()
        import numpy as np
        x = torch.from_numpy(np.stack([np.asarray(tok[o:o+CTX]) for o in offs]).astype(np.int64)).cuda()
        y = torch.from_numpy(np.stack([np.asarray(tok[o+1:o+1+CTX]) for o in offs]).astype(np.int64)).cuda()
        hf = model.nf if not mtp else None
        # forward with hidden access
        h_states = None
        def fwd(idx):
            h = model.embed(idx)
            for blk in model.blocks:
                mode = blk.attn.kv_mode
                if mode == "own":
                    h = blk(h)
            hf = model.nf(h)
            lg = model.head(hf)
            return lg, hf
        lg, hs = fwd(x)
        main = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.view(-1))
        loss = main
        losses_main = losses  # main-only tracked separately below
        if mtp is not None and mtp_weight:
            lg2 = mtp(hs[:, :-2])          # position t predicts token t+2
            aux = F.cross_entropy(lg2.reshape(-1, 32768).float(), y[:, 2:].reshape(-1))
            loss = loss + mtp_weight * aux
            aux_losses.append(aux.item())
        losses_main.append(main.item() if mtp is not None else loss.item())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if mtp is not None:
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
        for o in opts:
            o.step()
        if sinkhorn_rate:
            sinkhorn_rebalance(model, sinkhorn_rate)
        losses.append(loss.item())
    out = {"variant": name,
           "loss_last50": round(sum(losses_main[-50:]) / 50, 4),
           "loss_curve": [round(sum(losses_main[i:i+25]) / 25, 4) for i in range(0, STEPS, 25)]}
    if mtp_weight:
        out["aux_curve"] = [round(sum(aux_losses[i:i+50]) / 50, 3)
                            for i in range(0, len(aux_losses), 50)]
    print(out, flush=True)
    del model
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    res = [run("m5_control", None, None),
           run("m5_sinkhorn", 0.05),
           run("m6_mtp_aux", None, 0.3)]
    with open(os.path.join(OUT, "extra_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("DONE", flush=True)
