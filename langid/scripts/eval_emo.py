"""DA-2 emo eval: top-1 / top-3 vs label-frequency prior, per language.

Usage:
  python langid/scripts/eval_emo.py [--model runs/langid_da2/emo_best.pt]

Outputs runs/langid_da2/eval_report.json. Ties (top1-top2 < 1e-3) are
reported, never counted right or wrong (E-43 rule carried over).
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
OUT = os.path.join(BASE, "runs", "langid_da2")


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), "\t".join(parts[2:])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(OUT, "emo_best.pt"))
    ap.add_argument("--report", default=os.path.join(OUT, "eval_report.json"))
    args = ap.parse_args()

    ck = torch_load(args.model)
    labels = ck["labels"]
    K = len(labels)
    import torch
    if ck.get("arch", "bag") == "bag":
        W = ck["emb"].float().numpy()
        B = ck["bias"].float().numpy()
    else:
        raise SystemExit("eval_emo.py: use eval_emo_tf.py for transformer ckpts")

    test = read_tsv(os.path.join(DATA, "test.tsv"))
    train = read_tsv(os.path.join(DATA, "train.tsv"))

    # train-marginal frequency prior
    marg = np.zeros(K)
    for _, lab, _ in train:
        marg[lab] += 1.0
    marg /= marg.sum()
    prior_top1 = float(marg.max())

    res = {"per_lang": {}, "num_test": len(test), "prior_top1": prior_top1,
           "num_classes": K}
    for lang in ["ar", "en"]:
        rows = [r for r in test if r[0] == lang]
        if not rows:
            res["per_lang"][lang] = {"note": "no test rows"}
            continue
        texts = [r[2] for r in rows]
        golds = [r[1] for r in rows]
        top1 = 0
        top3 = 0
        ties = 0
        abstain = 0
        per_gold = collections.Counter()   # gold -> predicted
        per_pred = collections.Counter()
        hist = collections.Counter()
        for t, g in zip(texts, golds):
            ids, offs = emo_model.encode_batch([t])
            x = np.zeros(K, dtype=np.float64)
            for b in ids.tolist():
                x += W[b]
            scores = x + B
            order = np.argsort(-scores)
            top = list(order[:3])
            hist[int(top[0])] += 1
            per_pred[int(top[0])] += 1
            if scores[order[0]] - scores[order[1]] < 1e-3:
                ties += 1
            else:
                if int(order[0]) == g:
                    top1 += 1
                if g in top:
                    top3 += 1
            per_gold[g] += 1
        n = len(rows)
        per_lang = {
            "num": n,
            "top1": top1 / n,
            "top3": top3 / n,
            "ties_reported_not_counted": ties,
            "distinct_gold_labels": len(set(golds)),
            "distinct_pred_labels": len(per_pred),
        }
        res["per_lang"][lang] = per_lang
    res["gates"] = {
        "D1_rows": {"train": len(train), "test": len(test)},
        "D2_bar_top1_ge_2x_prior": 2.0 * res["prior_top1"],
        "D2_bar_top1_ge_prior_plus_005": res["prior_top1"] + 0.05,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=True)[:2200])


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
