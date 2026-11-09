"""DA-2b transformer eval: top-1/top-3 per language vs frequency prior.

Usage: python langid/scripts/eval_emo_tf.py --model runs/langid_da2b/emo_transformer.pt
Writes <model>.eval.json (dispatches on ck['arch'] == 'transformer').
Protocol: ties (top1-top2 margin < 1e-3) REPORTED, never counted (E-43 rule).
"""
import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data", "langid", "emo")


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), "\t".join(parts[2:])))
    return rows


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    assert ck.get("arch") == "transformer", "use eval_emo.py for bag ckpts"
    labels = ck["labels"]
    K = len(labels)
    model = emo_model.TinyTransformerEmo(K)
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()

    test = read_tsv(os.path.join(DATA, "test.tsv"))
    train = read_tsv(os.path.join(DATA, "train.tsv"))
    marg = np.zeros(K)
    for _, lab, _ in train:
        marg[lab] += 1.0
    marg /= marg.sum()
    prior_top1 = float(marg.max())

    res = {"model": args.model, "prior_top1": prior_top1, "num_classes": K,
           "num_test": len(test), "bars": {"2x_prior": 2 * prior_top1,
                                           "prior_plus_0.05": prior_top1 + 0.05},
           "per_lang": {}}
    for lang in ["ar", "en"]:
        rows = [r for r in test if r[0] == lang]
        n = len(rows)
        top1 = top3 = ties = 0
        used = collections.Counter()
        for i in range(0, n, 128):
            chunk = rows[i:i + 128]
            ids, offs = emo_model.encode_batch([r[2] for r in chunk])
            with torch.no_grad():
                logits = model(ids, offs).numpy()
            gold = [r[1] for r in chunk]
            for row, g in zip(logits, gold):
                order = np.argsort(-row)
                margin = row[order[0]] - row[order[1]]
                if margin < 1e-3:
                    ties += 1
                else:
                    top1 += int(order[0] == g)
                    used[order[0]] += 1
                top3 += int(g in order[:3])
        res["per_lang"][lang] = {
            "n": n, "top1": top1 / n, "top3": top3 / n, "ties": ties,
            "distinct_pred": len(used), "top1_2x_prior": (top1 / n) >= 2 * prior_top1,
            "top1_prior_plus": (top1 / n) >= prior_top1 + 0.05,
        }
    outp = args.report or (args.model + ".eval.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res["per_lang"], ensure_ascii=True))


if __name__ == "__main__":
    main()
