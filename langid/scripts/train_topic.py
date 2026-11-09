"""DA-3 (Gist) topic trainer - E-48/E-52: hashed n-gram EmbeddingBag LR (CPU).

Same recipe as DA-2 bag trainer, retargeted at data/langid/topic (17 classes:
ar SANAD 7 topics + en HuffPo top-10).

Usage: python langid/scripts/train_topic.py [--epochs 10] [--lr 0.25]
       [--opt adam|muon]

E-53: --opt muon uses the E-41 Muon recipe (NS5 orthogonalized momentum,
alpha=-lr*0.2) directly on the embedding table + bias; --opt adam is the
E-52 control.
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model  # shared hashed-n-gram model code

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "langid", "topic"))


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            lang, lab, text = parts[0], int(parts[1]), "\t".join(parts[2:])
            rows.append((lang, lab, text))
    return rows


def zeropower_ns5(M, steps=3):
    """Keller Jordan NS5 quintic Newton-Schulz (E-41 recipe)."""
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
                if p.ndim < 2:  # bias: plain SGDM (NS5 needs a matrix)
                    p.add_(buf, alpha=-lr)
                else:
                    p.add_(zeropower_ns5(u), alpha=-lr * 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--opt", choices=["adam", "muon"], default="adam",
                    help="E-53: muon = E-41 orthogonalized-momentum recipe")
    args = ap.parse_args()

    ROOT = str(pathlib.Path(DATA).parents[2])
    outp = args.out or os.path.join(ROOT, "runs", "langid_da3", "topic_bag.pt")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.manual_seed(args.seed)

    labels = json.load(open(os.path.join(DATA, "topic_vocab.json"),
                            encoding="utf-8"))["labels"]
    K = len(labels)

    train = read_tsv(os.path.join(DATA, "train.tsv"))
    val = read_tsv(os.path.join(DATA, "val.tsv"))
    tr_texts = [r[2] for r in train]
    tr_labels = torch.tensor([r[1] for r in train], dtype=torch.long)
    va_texts = [r[2] for r in val]
    va_langs = [r[0] for r in val]
    va_labels = torch.tensor([r[1] for r in val], dtype=torch.long)

    model = emo_model.HashedEmo(K)
    if args.opt == "muon":
        opt = Muon(model.parameters(), args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # per-lang frequency prior on val (for honest D2 bars)
    by_lang = collections.Counter(va_langs)
    prior = {}
    for lang2, n in by_lang.items():
        sub = [r[1] for r in val if r[0] == lang2]
        cc = collections.Counter(sub)
        prior[lang2] = max(cc.values()) / n
    print("val freq prior:", {k: round(v, 4) for k, v in prior.items()})

    log = []
    best = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_texts))
        tot = 0.0
        nb = 0
        for i in range(0, len(tr_texts), args.batch):
            idx = perm[i:i + args.batch]
            b_ids, b_off = emo_model.encode_batch([tr_texts[j] for j in idx.tolist()])
            b_lab = tr_labels[idx]
            opt.zero_grad()
            logits = model(b_ids, b_off)
            loss = torch.nn.functional.cross_entropy(logits, b_lab)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            nb += len(idx)
        model.eval()
        with torch.no_grad():
            correct = {"ar": 0, "en": 0}
            nval = {"ar": 0, "en": 0}
            for i in range(0, len(va_texts), 1024):
                v_ids, v_off = emo_model.encode_batch(va_texts[i:i + 1024])
                pred = model(v_ids, v_off).argmax(dim=1)
                for j, ridx in enumerate(range(i, min(i + 1024, len(va_texts)))):
                    lg = va_langs[ridx]
                    nval[lg] += 1
                    correct[lg] += int(pred[j].item() == va_labels[ridx].item())
        accs = {lg: correct[lg] / max(nval[lg], 1) for lg in nval}
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                    "val_acc": accs, "t": time.time() - t0})
        score = min(accs.values())
        print("ep", ep, "loss", round(tot / max(nb, 1), 4),
              "val ar", round(accs["ar"], 4), "en", round(accs["en"], 4), flush=True)
        if score > best:
            best = score
            ck = {"arch": "bag", "labels": labels, "val_acc": accs,
                  "args": vars(args), "epoch": ep,
                  "emb": model.emb.weight.detach().half(),
                  "bias": model.bias.detach().half()}
            torch.save(ck, outp)
    with open(outp + ".log.json", "w", encoding="utf-8") as f:
        json.dump({"log": log, "best_worst_lang_val_acc": best,
                   "epochs": args.epochs, "num_classes": K,
                   "num_train": len(train), "num_val": len(val),
                   "val_freq_prior": prior, "args": vars(args)}, f, indent=1)
    print("BEST", round(best, 4))
    print("CKPT", outp)


if __name__ == "__main__":
    main()
