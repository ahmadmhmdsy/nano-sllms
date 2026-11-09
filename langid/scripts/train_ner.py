"""DA-7 (Redact) NER trainer - E-54: token-level hashed char-n-gram tagger (CPU).

Per-token feature text = "<prev> <cur> <next>" (context words included in the
same FNV-1a hashed bag as DA-1/2/3). HashedEmo -> linear over BIO tags.
Optimizer default muon 3e-2 per E-53 (E-41/E-53 lever).

Usage: python langid/scripts/train_ner.py [--epochs 3] [--lr_muon 3e-2]
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
from langid.scripts.train_topic import Muon

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "langid", "ner"))


def read_sents(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t") if line.strip() else []
            toks = [parts[i] for i in range(0, len(parts), 2)]
            tags = [parts[i] for i in range(1, len(parts), 2)]
            if toks and len(toks) == len(tags):
                out.append((toks, tags))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192, help="tokens per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", choices=["adam", "muon"], default="muon")
    ap.add_argument("--lr_muon", type=float, default=3e-2)
    ap.add_argument("--lr", type=float, default=0.02, help="adam lr")
    ap.add_argument("--out", default="")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--clip", type=float, default=0.5)
    args = ap.parse_args()

    ROOT = str(pathlib.Path(DATA).parents[2])
    outp = args.out or os.path.join(ROOT, "runs", "langid_da7",
                                    "ner_tagger_%s.pt" % args.opt)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.manual_seed(args.seed)

    labels = json.load(open(os.path.join(DATA, "ner_vocab.json"),
                            encoding="utf-8"))["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    O_ID = lab2id["O"]

    train = read_sents(os.path.join(DATA, "train.tsv"))
    val = read_sents(os.path.join(DATA, "val.tsv"))

    def flatten(sents):
        texts, labs, sent_ix = [], [], []
        for si, (toks, tags) in enumerate(sents):
            for i in range(len(toks)):
                prev = toks[i - 1] if i > 0 else "<s>"
                nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
                texts.append(prev + " " + toks[i] + " " + nxt)
                sent_ix.append(si)
                sent_toks = toks
                lab = tags[i]
                labs.append(lab2id[lab])
        return texts, torch.tensor(labs, dtype=torch.long), sent_ix

    tr_texts, tr_labs, _ = flatten(train)
    va_texts, va_labs, va_si = flatten(val)
    print("train tokens", len(tr_texts), "val tokens", len(va_texts))

    model = emo_model.HashedEmo(K)
    weights = None
    if args.weighted:
        cnt = collections.Counter(tr_labs.tolist())
        w = torch.tensor([1.0 / max(cnt[k], 1) ** 0.5 for k in range(K)], dtype=torch.float32)
        weights = w / w.mean()
    if args.opt == "muon":
        opt = Muon(model.parameters(), args.lr_muon)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            b_lab = tr_labs[idx]
            opt.zero_grad()
            logits = model(b_ids, b_off)
            loss = torch.nn.functional.cross_entropy(logits, b_lab, weight=weights)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            nb += len(idx)
        model.eval()
        with torch.no_grad():
            correct = 0
            ent_correct = 0
            ent_total = 0
            conf = collections.Counter()
            for i in range(0, len(va_texts), 8192):
                v_ids, v_off = emo_model.encode_batch(va_texts[i:i + 8192])
                pred = model(v_ids, v_off).argmax(dim=1)
                vb = va_labs[i:i + 8192]
                correct += int((pred == vb).sum())
                ent_mask = vb != O_ID
                ent_correct += int((pred[ent_mask] == vb[ent_mask]).sum())
                ent_total += int(ent_mask.sum())
        acc = correct / len(va_texts)
        ent_acc = ent_correct / max(ent_total, 1)
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                    "val_acc": acc, "val_entity_acc": ent_acc,
                    "t": time.time() - t0})
        print("ep", ep, "loss", round(tot / max(nb, 1), 4),
              "val_acc", round(acc, 4), "ent_acc", round(ent_acc, 4), flush=True)
        if ent_acc > best:
            best = ent_acc
            torch.save({"arch": "bag", "labels": labels, "val_acc": acc,
                        "val_entity_acc": ent_acc, "args": vars(args), "epoch": ep,
                        "emb": model.emb.weight.detach().half(),
                        "bias": model.bias.detach().half()}, outp)
    with open(outp + ".log.json", "w", encoding="utf-8") as f:
        json.dump({"log": log, "best_entity_val_acc": best,
                   "epochs": args.epochs, "num_classes": K,
                   "num_train_tokens": len(tr_texts),
                   "num_val_tokens": len(va_texts), "args": vars(args)}, f, indent=1)
    print("BEST", round(best, 4))
    print("CKPT", outp)


if __name__ == "__main__":
    main()
