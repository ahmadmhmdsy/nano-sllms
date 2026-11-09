"""E-23d arm D1 training - ZM-style BiLSTM on v3q tokens, EXACT arm-C budget:
2500 steps, batch 32, AdamW lr 4e-4, wd 0.1, fp16 autocast with skip-on-overflow.
Gates every 500 steps using bench.load_pairs loader + eval.py compare.
Model age/params ~4.5M. Save: runs/diac/e23d_bilstm/{best.pt, gate_eval.csv, final}
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokenizer as TK
import bench as B
from e23_bilstm_model import build

RUN = REPO / "runs" / "diac" / ("e23d_bilstm_lr" + __import__('os').environ.get('S4_LR', '4e-4'))
RUN.mkdir(parents=True, exist_ok=True)
CTX = 128
import os
STEP_TOTAL = int(__import__('os').environ.get('S4_TOTAL', '2500'))
LR = float(__import__('os').environ.get('S4_LR', '4e-4'))

dev = "cuda"
model = build(TK.VOCAB_SIZE).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
print("params", sum(p.numel() for p in model.parameters()), flush=True)

T = np.load(str(REPO / "data/diac/v3q/tokens/train_ids.npy"), mmap_mode="r")
TY = np.load(str(REPO / "data/diac/v3q/tokens/train_y.npy"), mmap_mode="r")
n = len(T)
rng = np.random.default_rng(20260919)

def batch(bs=32):
    idx = rng.integers(0, n, bs)
    x = torch.from_numpy(np.asarray(T[idx])).to(dev)
    y = torch.from_numpy(np.asarray(TY[idx])).to(dev)
    return x, y

def loss_fn(logits, y):
    m = y >= 0
    flat = logits[m]
    tgt = y[m]
    return torch.nn.functional.cross_entropy(flat, tgt.long())

@torch.no_grad()
def val_loss():
    model.eval()
    vs = np.load(str(REPO / "data/diac/v3q/tokens/val_ids.npy"), mmap_mode="r")
    vy = np.load(str(REPO / "data/diac/v3q/tokens/val_y.npy"), mmap_mode="r")
    vp = np.random.default_rng(7).integers(0, len(vs), 500)
    tot = cnt = 0
    for i in vp:
        x = torch.from_numpy(np.asarray(vs[i].reshape(1, -1))).to(dev)
        y = torch.from_numpy(np.asarray(vy[i].reshape(1, -1))).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lo = model(x)["logits"]
        tot += float(loss_fn(lo, y)); cnt += 1
    model.train()
    return tot / cnt

def gate(name, gstep):
    model.eval()
    d = RUN / "gate_probe"
    d.mkdir(exist_ok=True)
    outp = d / f"step{step}_{name}_pred.txt"
    preds, refs = [], []
    for bare, reflist in bench_load_pairs(name, CTX):
        pieces = []
        for piece in bench_chunks(bare, CTX):
            x = torch.tensor(TK.encode(piece), dtype=torch.long).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)["logits"][0]
            pred = logits.argmax(-1).tolist()
            out = []
            ti = 0
            for ch in piece:
                out.append(ch)
                if is_arabic_base(ch):
                    out.append(marks_for_label(pred[ti]))
                ti += 1
            pieces.append("".join(out))
        preds.append("".join(pieces))
        refs.append("\t".join(reflist))
    outp.write_text("\n".join(preds) + "\n", encoding="utf-8")
    refp = outp.with_suffix(".ref.txt")
    refp.write_text("\n".join(refs) + "\n", encoding="utf-8")
    # in-process compare (same code as eval.py compare mode; subprocess was
    # buffering oddly under the pwsh root redirect)
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    import eval_der as EVALM
    pred_lines = outp.read_text(encoding="utf-8").splitlines()
    ref_sets = [refp.read_text(encoding="utf-8").splitlines()]
    n2 = max(len(pred_lines), len(ref_sets[0]))
    metrics = []
    for i in range(n2):
        ptxt = pred_lines[i] if i < len(pred_lines) else ""
        refs = ref_sets[0][i].split("\t") if i < len(ref_sets[0]) else [""]
        metrics.append(EVALM.score_line(ptxt, refs))
    agg = EVALM.aggregate(metrics)
    der = agg["DER"]
    row = f"{gstep},{name},{der},"
    print(f"[GATE] step {gstep} {name} DER {der}", flush=True)
    model.train()
    return der

# lazy funcs (paths): use bench helpers directly
bench_chunks = None
import bench as _b
bench_load_pairs = _b.load_pairs
bench_chunks = _b.chunks_of
from passthrough import is_arabic_base
from labels import marks_for_label

csv_lines = ["step,gate,DER,WER"]
best = {"mean": 1e9, "step": -1}
state = {}
last_state = RUN / "state.pt"
start_step = 0
if last_state.exists():
    s = torch.load(last_state, map_location=dev)
    model.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    start_step = s["step"]
    best = s["best"]
    csv_lines += s["csv"]
    print("resumed at", start_step, flush=True)

t0 = time.time()
model.train()
# CSV replay of skip steps (gate info already in csv_lines), continue training
for step in range(start_step, STEP_TOTAL):
    x, y = batch()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = loss_fn(out["logits"], y)
    opt.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if all(torch.isfinite(g).all() for g in grads):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    if (step + 1) % 100 == 0:
        print(f"step {step + 1} loss {float(loss):.4f} {(-(step + 1) / (time.time() - t0)):.2f}s/it", flush=True)
    if (step + 1) % 500 == 0:
        ders = []
        for gname in ["fadel_test", "sadeed25", "wikinews2024", "wikinews2014"]:
            der = gate(gname, step + 1)
            ders.append(der)
            csv_lines.append(f"{step + 1},{gname},{der},")
        mean = sum(ders) / len(ders)
        csv_lines.append(f"{step + 1},mean,{mean},0")
        if mean < best["mean"]:
            best = {"mean": mean, "step": step + 1}
            torch.save({"model": model.state_dict(), "step": step + 1, "best": best, "csv": csv_lines},
                       RUN / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                    "best": best, "csv": csv_lines}, last_state)
        (RUN / "gate_eval.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

fdir = RUN / "final"
fdir.mkdir(exist_ok=True)
if (RUN / "best.pt").exists():
    torch.save(torch.load(RUN / "best.pt", map_location="cpu")["model"], fdir / "model.pt")
(RUN / "gate_eval.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
print("DONE best", best, flush=True)
