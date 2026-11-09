"""DA-3 (E-48) test eval: per-lang top-1 vs frequency prior + bars.

Bars (pre-registered): 2x_prior and prior+0.05.
Dumps <model>.topic_eval.json
"""
import collections
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "langid", "topic"))


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), "\t".join(parts[2:])))
    return rows


def main():
    ckp = sys.argv[1] if len(sys.argv) > 1 else "runs/langid_da3/topic_bag.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    labels = ck["labels"]
    K = len(labels)
    model = emo_model.HashedEmo(K)
    model.emb.weight.data = ck["emb"].float()
    model.bias.data = ck["bias"].float()
    model.eval()

    test = read_tsv(os.path.join(DATA, "test.tsv"))
    texts = [r[2] for r in test]
    labs = torch.tensor([r[1] for r in test], dtype=torch.long)
    langs = [r[0] for r in test]

    stats = collections.defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "cls": collections.Counter()})
    with torch.no_grad():
        for i in range(0, len(texts), 1024):
            ids, off = emo_model.encode_batch(texts[i:i + 1024])
            logits = model(ids, off)
            top3 = logits.topk(3, dim=1).indices
            for j in range(logits.shape[0]):
                st = stats[langs[i + j]]
                st["n"] += 1
                st["cls"][int(labs[i + j])] += 1
                if int(top3[j, 0]) == int(labs[i + j]):
                    st["top1"] += 1
                if int(labs[i + j]) in top3[j].tolist():
                    st["top3"] += 1

    out = {"model": ckp, "labels": labels, "langs": {}}
    for lg, st in sorted(stats.items()):
        n = st["n"]
        prior = max(st["cls"].values()) / n
        top1 = st["top1"] / n
        top3 = st["top3"] / n
        out["langs"][lg] = {
            "n": n, "top1": top1, "top3": top3, "freq_prior": prior,
            "bar_2x": 2 * prior, "bar_prior_plus_005": prior + 0.05,
            "pass_2x": top1 >= 2 * prior,
            "pass_prior_plus_005": top1 >= prior + 0.05,
        }
    outp = ckp + ".topic_eval.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for lg, v in out["langs"].items():
        print(lg, "n", v["n"], "top1", round(v["top1"], 4),
              "prior", round(v["freq_prior"], 4),
              "top3", round(v["top3"], 4),
              "2x", round(v["bar_2x"], 4), "pass2x", v["pass_2x"],
              "p+.05", round(v["bar_prior_plus_005"], 4),
              "passBarB", v["pass_prior_plus_005"])
    print("EVAL_JSON", outp)


if __name__ == "__main__":
    main()
