"""DA-7 (Redact) NER dataset build - E-54.

ar: iahlt/arabic_ner_mafat (40,000 sentences, tokens + BILUO raw_tags -> BIO).
Output: data/langid/ner/{train,val,test}.tsv, one sentence per line:
  tok tag tok tag ...  (tab between pairs, space inside)
"""
import collections
import json
import os
import random

import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "data", "langid", "ner")
RAW = os.path.join(BASE, "data", "langid", "raw", "mafat_ner.parquet")


def biluo_to_bio(tag):
    if tag == "O":
        return "O"
    pre, typ = tag.split("-", 1)
    if pre in ("B", "S"):
        return "B-" + typ
    return "I-" + typ


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(42)
    t = pq.read_table(RAW)
    sents = []
    tagc = collections.Counter()
    for tk, rt in zip(t.column("tokens").to_pylist(), t.column("raw_tags").to_pylist()):
        if not tk or len(tk) < 3:
            continue
        if any(x is None for x in tk) or any(x is None for x in rt):
            continue
        bio = [biluo_to_bio(x) for x in rt]
        if len(tk) != len(bio):
            continue
        for x in bio:
            tagc[x] += 1
        sents.append((tk, bio))
    print("sentences:", len(sents))
    print({k: v for k, v in tagc.most_common()})
    random.shuffle(sents)
    n = len(sents)
    cuts = (int(n * 0.9), int(n * 0.95))
    parts = {"train": sents[:cuts[0]], "val": sents[cuts[0]:cuts[1]], "test": sents[cuts[1]:]}
    for name, rows in parts.items():
        with open(os.path.join(OUT, name + ".tsv"), "w", encoding="utf-8") as f:
            for tk, bio in rows:
                f.write("\n" if False else "")
                f.write("\t".join("%s\t%s" % (a, b) for a, b in zip(tk, bio)) + "\n")
    labels = sorted(tagc)
    with open(os.path.join(OUT, "ner_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=1)
    print("labels", len(labels))
    print({k: len(v) for k, v in parts.items()})


if __name__ == "__main__":
    main()
