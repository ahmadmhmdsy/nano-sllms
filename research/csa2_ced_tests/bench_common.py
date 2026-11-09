import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = r"E:/python_projects/nano_SLMs"
TOK = os.path.join(REPO, "data", "pilot", "tokens", "train_000.bin")
VAL = os.path.join(REPO, "data", "pilot", "tokens", "val_000.bin")
OUT = os.path.join(REPO, "research", "csa2_ced_tests")
VAL = os.path.join(REPO, "data", "pilot", "tokens", "val_000.bin")
SEED = 0

def load_tokens(path):
    a = np.memmap(path, dtype=np.uint32, mode="r")
    return a

def batch(tok, bs, ctx, generator=None, max_off=None):
    if max_off is None:
        max_off = len(tok) - ctx - 1
    offs = torch.randint(0, max_off, (bs,)).tolist()
    x = (torch.from_numpy(np.stack([np.asarray(tok[o:o + ctx]) for o in offs]).astype(np.int64)).cuda(), )
    y = torch.from_numpy(np.stack([np.asarray(tok[o + 1:o + 1 + ctx]) for o in offs]).astype(np.int64)).cuda()
    return x[0], y
def guard():
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 2**30
    if free_gb < 4.0:
        raise SystemExit(f"FREE-VRAM GUARD: only {free_gb:.2f} GiB free; aborting (other agent may be active)")
    return free_gb


def make_opt(params, lr):
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

def cosine(step, total, base, warm=25, floor=0.1):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))

def train_steps(model, tok, steps, bs, ctx, lr, window_loss_avg=50, gen=None):
    opt = make_opt(model.parameters(), lr)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    losses, t0, tok_count = [], time.time(), 0
    max_off = len(tok) - ctx - 1
    for step in range(steps):
        lr_s = cosine(step, steps, lr)
        for g in opt.param_groups:
            g["lr"] = lr_s
        x, y = batch(tok, bs, ctx, gen, max_off)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        tok_count += bs * ctx
    dt = time.time() - t0
    return {
        "loss_first50": sum(losses[:50]) / max(1, min(50, len(losses))),
        "loss_last50": sum(losses[-50:]) / min(50, len(losses)),
        "loss_curve": [round(sum(losses[i:i + 25]) / 25, 4) for i in range(0, len(losses), 25)],
        "tokens_per_s": tok_count / dt,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
