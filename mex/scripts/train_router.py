# mex/scripts/train_router.py — mu1 arm C: train the dispatch router (CPU).
r"""Trains the tiny prompt->task classifier (mex/src/router.py) on the labeled
union pool data/mex/mixed/train.jsonl (8000 items = 2000/task, TRAIN-legal
pools) and evaluates routing on data/mex/mixed/val.jsonl (200 held-out).

Spec: CharVocab ids -> embedding(32), masked mean-pool, MLP 32-32-4 logits,
cross-entropy on the task index. Seed 42, Adam 1e-3, 3 epochs, batch 256,
--cap 8000 keeps runtime in the minutes range on CPU.

Saves runs/mex/router/router.pt (state dict) + router_meta.json with the
val routing accuracy and its Wilson 95% CI against the 0.25 random floor
(4 tasks => 1/4 chance routing).

Usage:
    & .\.venv\Scripts\python.exe mex\scripts\train_router.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.metrics import wilson_ci  # noqa: E402
from mex.src.router import (TASKS,  # noqa: E402
                            PromptRouter,
                            encode_batch,
                            route_tasks,
                            save_router)
from mex.src.vocab import CharVocab  # noqa: E402

MIXED_DIR = Path(__file__).resolve().parents[2] / "data" / "mex" / "mixed"
LABEL = {t: i for i, t in enumerate(TASKS)}


def read_jsonl(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def batch_tensors(items: list[dict], voc: CharVocab):
    ids, mask = encode_batch([i["prompt"] for i in items], voc)
    y = torch.tensor([LABEL[i["task"]] for i in items], dtype=torch.long)
    return ids, mask, y


def main() -> None:
    ap = argparse.ArgumentParser(description="train the mu1 arm-C router")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap", type=int, default=8000,
                    help="max train items (runtime guard)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "mex" / "router"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    voc = CharVocab()

    train = read_jsonl(MIXED_DIR / "train.jsonl")[:args.cap]
    val = read_jsonl(MIXED_DIR / "val.jsonl")
    print(f"train={len(train)} val={len(val)}", flush=True)

    model = PromptRouter(vocab_size=len(voc.vocab))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total, n_batches = 0.0, 0
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            ids, mask, y = batch_tensors([train[k] for k in idx], voc)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={total / max(n_batches, 1):.4f}", flush=True)

    preds = route_tasks(model, [i["prompt"] for i in val], voc)
    hits = sum(1 for p, i in zip(preds, val) if p == i["task"])
    n = len(val)
    acc = hits / n
    ci = wilson_ci(hits, n)
    floor = 1.0 / len(TASKS)
    meta = {
        "model": "PromptRouter(emb32 meanpool MLP32-32-4)",
        "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
        "batch": args.batch, "train_items": len(train), "val_items": n,
        "val_routing_accuracy": acc,
        "ci95": ci,
        "random_floor": floor,
        "floor_margin": acc - floor,
        "confusion": {t: {p: sum(1 for pr, ti in zip(preds, val)
                                 if ti["task"] == t and pr == p)
                          for p in TASKS} for t in TASKS},
    }
    out = Path(args.out)
    save_router(out / "router.pt", model, meta)
    print(json.dumps({k: meta[k] for k in ("val_routing_accuracy", "ci95",
                                           "random_floor", "floor_margin")},
                     indent=1), flush=True)
    print(f"saved {out / 'router.pt'} "
          f"({time.time() - t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
