"""DA-8 (E-56) title ranker trainer v2: shared dual encoder, GPU-pooled, RAM-safe.

Root cause of the earlier RAM/VRAM overflow (fixed here): v1 pre-encoded the
full corpus into feature tensors in RAM (both langs x both sides simultaneously)
and ran two processes together, spilling CUDA shared memory.

v2: per-batch on-the-fly encoding; one EmbeddingBag(mean) kernel per side;
InfoNCE in-batch negatives; checkpoints fp16 on CPU; peak VRAM printed per epoch.
Run ONE training process at a time (user policy).
Usage: python langid/scripts/train_rank.py --lang ar --epochs 8
"""
import argparse, json, os, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model
from langid.scripts.train_topic import Muon

BASE = pathlib.Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "langid" / "title"


def read(p):
    a, b = [], []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        x, y = line.rstrip("\n").split(chr(9), 1)
        a.append(x)
        b.append(y)
    return a, b


def cosine(a, b):
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-6)
    b = b / b.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return a @ b.T


class DualBag(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(65536, d, mode="mean")
        torch.nn.init.normal_(self.bag.weight, std=0.02)
        self.b_a = torch.nn.Parameter(torch.zeros(d))
        self.b_b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, ids, off, side):
        h = self.bag(ids, off)
        return h + (self.b_a if side == "a" else self.b_b)


def encode_pool(model, texts, side, device):
    ids, off = emo_model.encode_batch(texts)
    return model(ids.to(device), off.to(device), side)


def batch_r1(model, ta, tb, device, B=64, top10=False):
    ok1 = ok10 = n = 0
    with torch.no_grad():
        for i in range(0, len(ta) - B + 1, B):
            sl = list(range(i, i + B))
            ha = encode_pool(model, [ta[j] for j in sl], "a", device)
            hb = encode_pool(model, [tb[j] for j in sl], "b", device)
            ranks = cosine(ha, hb).argsort(1, descending=True)
            gold = torch.arange(B, device=ranks.device).unsqueeze(1)
            pos = torch.zeros(B, dtype=torch.long, device=ranks.device)
            pos = (ranks == gold).float().argmax(dim=1)
            ok1 += int((pos == 0).sum())
            if top10:
                ok10 += int((pos < 10).sum())
            n += B
    return ok1 / max(n, 1), ok10 / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "en"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--opt", choices=["adam", "muon"], default="muon")
    ap.add_argument("--lr_muon", type=float, default=3e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument('--patience', type=int, default=4, help='stop after N val drops below best')
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dd = DATA / args.lang
    tr_a, tr_b = read(str(Dd / "train.tsv"))
    va_a, va_b = read(str(Dd / "val.tsv"))
    print(args.lang, "train", len(tr_a), "val", len(va_a), "device", device, flush=True)
    model = DualBag(args.d).to(device)
    params = list(model.parameters())
    opt = Muon(params, args.lr_muon) if args.opt == "muon" else torch.optim.Adam(params, lr=args.lr)
    outp = args.out or str(BASE / "runs" / "langid_da8" / ("rank_%s.pt" % args.lang))
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    best = -1.0
    drops = 0
    t0 = time.time()
    log = []
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr_a))
        tot = 0.0
        nb = 0
        model.train()
        for i in range(0, len(tr_a), args.batch):
            idx = perm[i:i + args.batch].tolist()
            ha = encode_pool(model, [tr_a[j] for j in idx], "a", device)
            hb = encode_pool(model, [tr_b[j] for j in idx], "b", device)
            lab = torch.arange(len(idx), device=device)
            loss = torch.nn.functional.cross_entropy(cosine(ha, hb) / args.tau, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            nb += len(idx)
        model.eval()
        r1, r10 = batch_r1(model, va_a, va_b, device)
        log.append({"epoch": ep, "loss": round(tot / max(nb, 1), 4),
                    "val_r1_batch64": round(r1, 4), "t": round(time.time() - t0, 1)})
        peak = torch.cuda.max_memory_allocated() // (1 << 20) if torch.cuda.is_available() else 0
        print("ep", ep, "loss", round(tot / max(nb, 1), 4), "val_r1", round(r1, 4),
              "peakMB", peak, flush=True)
        if r1 > best:
            drops = 0
        else:
            drops += 1
            if drops >= args.patience:
                print("EARLY_STOP after", ep, "epochs (val_r1 flat/declining)", flush=True)
                log[-1]["early_stop"] = True
                break
        if r1 > best:
            best = r1
            sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
            torch.save({"state": sd, "args": vars(args), "val_r1": best, "d": args.d}, outp)
    with open(outp + ".log.json", "w", encoding="utf-8") as f:
        json.dump({"log": log, "best_val_r1_batch64": best, "args": vars(args)}, f, indent=1)
    print("BEST", round(best, 4))
    print("CKPT", outp)


if __name__ == "__main__":
    main()