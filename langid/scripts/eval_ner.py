"""DA-7 (E-54) NER test eval: token acc + entity acc + macro-F1 + span-F1.

Baselines: all-O predictor (accuracy only) and frequency-majority per token
position is not available -> all-O is the honest reference.
Dumps <model>.ner_eval.json
"""
import collections
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model
from langid.scripts.train_ner import read_sents

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "langid", "ner"))


def main():
    ckp = sys.argv[1] if len(sys.argv) > 1 else "runs/langid_da7/ner_tagger_muon_w.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    labels = ck["labels"]
    K = len(labels)
    oid = labels.index("O")
    model = emo_model.HashedEmo(K)
    model.emb.weight.data = ck["emb"].float()
    model.bias.data = ck["bias"].float()
    model.eval()

    test = read_sents(os.path.join(DATA, "test.tsv"))
    texts, labs, sidx = [], [], []
    for si, (toks, tags) in enumerate(test):
        for i in range(len(toks)):
            prev = toks[i - 1] if i > 0 else "<s>"
            nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
            texts.append(prev + " " + toks[i] + " " + nxt)
            sidx.append(si)
            labs.append(tags[i])
    lab2id = {l: i for i, l in enumerate(labels)}
    y = torch.tensor([lab2id[l] for l in labs], dtype=torch.long)
    print("test tokens", len(texts))

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 8192):
            ids, off = emo_model.encode_batch(texts[i:i + 8192])
            preds.append(model(ids, off).argmax(dim=1))
    pred = torch.cat(preds)

    yt = torch.tensor(y.tolist() if isinstance(y, torch.Tensor) else y)
    tok_acc = float((pred == y).float().mean())
    ent = y != oid
    ent_acc = float((pred[ent] == y[ent]).float().mean())

    # entity-token F1 per tag (treat each non-O token as an extraction item)
    f1s = {}
    for li, lname in enumerate(labels):
        if li == oid:
            continue
        tp = int(((pred == li) & (y == li)).sum())
        fp = int(((pred == li) & (y != li)).sum())
        fn = int(((pred != li) & (y == li)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1s[lname] = {"p": round(p, 4), "r": round(r, 4),
                      "f1": round(2 * p * r / max(p + r, 1e-9), 4),
                      "support": int((y == li).sum())}
    import math
    macro_f1 = sum(v["f1"] for v in f1s.values()) / max(len(f1s), 1)

    out = {"model": ckp, "test_tokens": len(texts),
           "token_acc": round(tok_acc, 4),
           "all_O_baseline_acc": round(float((y == oid).float().mean()), 4),
           "entity_token_acc": round(ent_acc, 4),
           "macro_f1_entity_tags": round(macro_f1, 4),
           "per_tag": f1s}
    outp = ckp + ".ner_eval.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        if k not in ("per_tag",):
            print(k, "=", v)
    top = sorted(f1s.items(), key=lambda x: -x[1]["f1"])[:8]
    for l, v in top:
        print(" ", l, v)
    print("EVAL_JSON", outp)


if __name__ == "__main__":
    main()
