"""M4: AdamW vs Muon-on-2D (orthogonalized momentum) at nano scale, fp32 clean.
Same mini model (d192 4L ctx256, no borrow), identical data/steps/seed.
"""
import json
import os
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_model import MiniLM
from bench_common import SEED, TOK, OUT, batch, guard, cosine, load_tokens, make_opt

STEPS, BS, CTX, LR, WARM = 300, 16, 256, 4e-4, 25


def zeropower_ns5(M, steps=3):
    """Keller Jordan NS5 quintic; keeps input orientation."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.float() / (M.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(list(params), dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for grp in self.param_groups:
            lr, mom = grp["lr"], grp["momentum"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(mom).add_(p.grad)
                u = (p.grad + mom * buf) if mom != 0 else buf
                p.add_(zeropower_ns5(u), alpha=-lr * 0.2)


def run(name, muon_lr):
    guard()
    torch.manual_seed(SEED)
    tok = load_tokens(TOK)
    model = MiniLM(vocab=32768, d=192, layers=4, n_heads=6, kv_heads=3,
                   device="cuda")
    mat, rest = [], []
    for pname, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in pname and "head" not in pname:
            mat.append(p)
        else:
            rest.append(p)
    opts = ([make_opt(model.parameters(), LR)] if muon_lr is None
            else [Muon(mat, muon_lr), make_opt(rest, LR)])
    g = torch.Generator(device="cuda").manual_seed(SEED)
    losses = []
    for step in range(STEPS):
        lr_s = cosine(step, STEPS, LR, warm=WARM)
        for o in opts:
            for gp in o.param_groups:
                gp["lr"] = lr_s if not isinstance(o, Muon) else muon_lr
        x, y = batch(tok, BS, CTX, g)
        lg = model(x)
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), y.view(-1))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        losses.append(loss.item())
    out = {"variant": name,
           "loss_last50": round(sum(losses[-50:]) / 50, 4),
           "loss_curve": [round(sum(losses[i:i + 25]) / 25, 4)
                          for i in range(0, STEPS, 25)]}
    print(out, flush=True)
    del model
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    res = [run("adamw_4e-4", None), run("muon_1e-2", 1e-2),
           run("muon_3e-2", 3e-2)]
    with open(os.path.join(OUT, "muon_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("DONE", flush=True)
