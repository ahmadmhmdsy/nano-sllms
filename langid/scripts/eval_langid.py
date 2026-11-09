"""DA-1 evaluation harness (Tongue-style protocol).

Pre-registered gates (docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md):
  full >= 0.97, 5-word >= 0.95, 3-word >= 0.90, 1-word >= 0.70,
  Arabic-script group (ar/fa/ur/ps) mean 3-word >= 0.85.
Ties (top1-top2 logit margin < --margin) and abstains (no letter features)
are reported separately and NEVER counted as correct.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import features
from langid.src.data import ARABIC_SCRIPT_GROUP, LANGS, LANG_TO_ID
from langid.src.model import HashedLangID, encode_batch

GATES = {"full": 0.97, 5: 0.95, 3: 0.90, 1: 0.70, "arabic_group_3w": 0.85}


def read_eval(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lang, _, text = line.rstrip("\n").partition("\t")
            if lang in LANG_TO_ID and text:
                rows.append((lang, text))
    return rows


def k_window(words, k, rand):
    if len(words) <= k:
        return words
    start = rand.randrange(len(words) - k + 1)
    return words[start:start + k]


def score_split(model, rows, k=None, margin=0.5):
    rand = random.Random(42)
    texts = []
    for _, text in rows:
        words = text.split()
        sel = k_window(words, k, rand) if k else words
        texts.append(" ".join(sel))
    featless = [not any(features.text_features(t)) for t in texts]
    correct = ties = abstain = 0
    per_lang_ok = collections.Counter()
    per_lang_total = collections.Counter()
    confusion = collections.Counter()
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            chunk = rows[i:i + 512]
            ids, offsets = encode_batch(texts[i:i + 512])
            top2 = model(ids, offsets).topk(2, dim=1)
            for j, (lang, _) in enumerate(chunk):
                gold = LANG_TO_ID[lang]
                i1, i2 = int(top2.indices[j][0]), int(top2.indices[j][1])
                v1, v2 = float(top2.values[j][0]), float(top2.values[j][1])
                per_lang_total[lang] += 1
                if featless[i + j]:
                    abstain += 1
                elif i1 == gold:
                    correct += 1
                    per_lang_ok[lang] += 1
                elif v1 - v2 < margin:
                    ties += 1
                    confusion[f"tie:{LANGS[i1]}->{lang}"] += 1
                else:
                    confusion[f"{LANGS[i1]}->{lang}"] += 1
    n = max(1, len(rows))
    return {
        "k": k, "n": len(rows), "acc": correct / n, "tie_rate": ties / n,
        "abstain_rate": abstain / n,
        "per_lang_acc": {l: per_lang_ok[l] / per_lang_total[l]
                         for l in sorted(per_lang_total)},
        "top_confusions": confusion.most_common(10),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/langid/eval.tsv")
    ap.add_argument("--model", default="runs/langid_da1/model_fp32.pt")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--report", default="runs/langid_da1/eval_report.json")
    args = ap.parse_args()

    model = HashedLangID(len(LANGS), features.NUM_BUCKETS)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.emb.weight.data = state["emb"]
    model.bias.data = state["bias"]
    model.eval()

    rows = read_eval(args.eval)
    results = {}
    for k in (None, 5, 3, 1):
        key = "full" if k is None else str(k)
        results[key] = score_split(model, rows, k=k, margin=args.margin)
        r = results[key]
        print(f"[eval] k={key:4s} acc={r['acc']:.4f} tie={r['tie_rate']:.4f} "
              f"abstain={r['abstain_rate']:.4f}")
    group = [results["3"]["per_lang_acc"].get(l, 0.0) for l in ARABIC_SCRIPT_GROUP]
    results["arabic_group_3w"] = sum(group) / len(group)
    print("[eval] arabic-script group 3-word:", " ".join(
        f"{l}={v:.3f}" for l, v in zip(ARABIC_SCRIPT_GROUP, group)),
        f"mean={results['arabic_group_3w']:.4f}")

    counts = collections.Counter(lang for lang, _ in rows)
    majority = max(counts.values()) / max(1, len(rows))
    results["trivial_majority_acc"] = majority
    print(f"[eval] trivial majority baseline acc={majority:.4f}")

    gates = {
        "G2_full>=0.97": results["full"]["acc"] >= GATES["full"],
        "G2_5w>=0.95": results["5"]["acc"] >= GATES[5],
        "G2_3w>=0.90": results["3"]["acc"] >= GATES[3],
        "G2_1w>=0.70": results["1"]["acc"] >= GATES[1],
        "G3_arabic_group_3w>=0.85": results["arabic_group_3w"] >= GATES["arabic_group_3w"],
    }
    print("[gates]", " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    results["gates"] = gates
    report = pathlib.Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[eval] report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
