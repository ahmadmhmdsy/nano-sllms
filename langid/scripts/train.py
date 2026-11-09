"""Train the DA-1 language-ID model on CPU (GPU-safe beside any run)."""
import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import features
from langid.src.data import LANGS, LANG_TO_ID
from langid.src.model import HashedLangID, encode_batch


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lang, _, text = line.rstrip("\n").partition("\t")
            if lang in LANG_TO_ID and text:
                rows.append((LANG_TO_ID[lang], text))
    return rows


def evaluate(model, rows, batch_size=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            ids, offsets = encode_batch([t for _, t in chunk])
            pred = model(ids, offsets).argmax(dim=1)
            gold = torch.tensor([lang for lang, _ in chunk], dtype=torch.long)
            correct += int((pred == gold).sum())
    return correct / max(1, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/langid")
    ap.add_argument("--out", default="runs/langid_da1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(42)
    train = read_tsv(pathlib.Path(args.data_dir) / "train.tsv")
    val = read_tsv(pathlib.Path(args.data_dir) / "val.tsv")
    print(f"[train] train={len(train)} val={len(val)} langs={len(LANGS)} "
          f"buckets={features.NUM_BUCKETS}")

    model = HashedLangID(len(LANGS), features.NUM_BUCKETS)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(train)).tolist()
        total_loss = 0.0
        steps = 0
        for i in range(0, len(train), args.batch_size):
            chunk = [train[j] for j in order[i:i + args.batch_size]]
            ids, offsets = encode_batch([t for _, t in chunk])
            gold = torch.tensor([lang for lang, _ in chunk], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(model(ids, offsets), gold)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            steps += 1
        acc = evaluate(model, val)
        print(f"[train] epoch {epoch + 1}/{args.epochs} "
              f"loss={total_loss / max(1, steps):.4f} val_acc={acc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc = acc
            torch.save({"emb": model.emb.weight.detach().cpu(),
                        "bias": model.bias.detach().cpu()},
                       out / "model_fp32.pt")

    (out / "train_summary.json").write_text(json.dumps({
        "epochs": args.epochs, "best_val_acc": best_acc, "langs": LANGS,
        "num_buckets": features.NUM_BUCKETS, "lr": args.lr,
        "batch_size": args.batch_size,
    }, indent=2), encoding="utf-8")
    print(f"[train] best val_acc={best_acc:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
