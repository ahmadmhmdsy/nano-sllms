"""DA-2 emo trainer: EmbeddingBag multinomial LR (or tiny transformer) on CPU.

Usage:
  python langid/scripts/train_emo.py [--epochs 5] [--lr 0.25] [--seed 0]
      [--weighted] [--label_smoothing 0.1] [--arch bag|transformer]
      [--out runs/langid_da2b/emo_arm1.pt]

Arms (E-47 registration):
  (a) bigger Arabic corpus - handled in build_emo_eval.py (dialects), all arms
  (b) --arch transformer
  (c) --weighted --label_smoothing 0.1
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
from langid.src import emo_model

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "langid", "emo"))


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            lang, lab, text = parts[0], int(parts[1]), "\t".join(parts[2:])
            rows.append((lang, lab, text))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weighted", action="store_true",
                    help="class-balanced weighted CE (inverse sqrt frequency)")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip")
    ap.add_argument("--arch", choices=["bag", "transformer"], default="bag")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ROOT = str(pathlib.Path(DATA).parents[2])
    outp = args.out or os.path.join(ROOT, "runs", "langid_da2b",
                                   "emo_%s%s%s.pt" % (
                                       args.arch,
                                       "_w" if args.weighted else "",
                                       "_ls" if args.label_smoothing else ""))
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.manual_seed(args.seed)

    vocab = json.load(open(os.path.join(DATA, "emo_vocab.json"), encoding="utf-8"))
    labels = vocab["labels"]
    K = len(labels)

    train = read_tsv(os.path.join(DATA, "train.tsv"))
    val = read_tsv(os.path.join(DATA, "val.tsv"))
    tr_texts = [r[2] for r in train]
    tr_langs = [r[0] for r in train]
    tr_labels = torch.tensor([r[1] for r in train], dtype=torch.long)
    va_texts = [r[2] for r in val]
    va_labels = torch.tensor([r[1] for r in val], dtype=torch.long)

    model = emo_model.HashedEmo(K) if args.arch == "bag" else emo_model.TinyTransformerEmo(K)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights = None
    if args.weighted:
        cnt = collections.Counter(tr_labels.tolist())
        w = torch.tensor([1.0 / max(cnt[k], 1) ** 0.5 for k in range(K)], dtype=torch.float32)
        weights = w / w.mean()

    ls = args.label_smoothing
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
            loss = torch.nn.functional.cross_entropy(
                logits, b_lab, weight=weights, label_smoothing=ls)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            nb += len(idx)
        model.eval()
        with torch.no_grad():
            # eval in slices (transformer pads per batch, bag is exact)
            correct = 0
            nval = 0
            for i in range(0, len(va_texts), 1024):
                v_ids, v_off = emo_model.encode_batch(va_texts[i:i + 1024])
                pred = model(v_ids, v_off).argmax(dim=1)
                correct += int((pred == va_labels[i:i + 1024]).sum())
                nval += 1024 and pred.shape[0]
            acc = correct / max(nval, 1)
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_acc": acc,
                    "t": time.time() - t0})
        print("ep", ep, "loss", round(tot / max(nb, 1), 4), "val_acc", round(acc, 4), flush=True)
        if acc > best:
            best = acc
            ck = {"arch": args.arch, "labels": labels, "val_acc": acc,
                  "args": vars(args), "epoch": ep}
            if args.arch == "bag":
                ck["emb"] = model.emb.weight.detach().half()
                ck["bias"] = model.bias.detach().half()
            else:
                ck["state"] = {k: v.detach().half() for k, v in model.state_dict().items()}
            torch.save(ck, outp)
    with open(outp + ".log.json", "w", encoding="utf-8") as f:
        json.dump({"log": log, "best_val_acc": best, "epochs": args.epochs,
                   "num_classes": K, "num_train": len(train), "num_val": len(val),
                   "args": vars(args)}, f, indent=1)
    print("BEST", round(best, 4))
    print("CKPT", outp)


if __name__ == "__main__":
    main()
