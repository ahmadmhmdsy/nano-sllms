"""DA-3 (Gist) topic-tagging dataset build - E-48.

ar: arbml/SANAD (131k rows, 7 topic classes: Tech/Culture/Finance/Sports/
    Politics/Religion/Medical).
en: heegyu/news-category-balanced-top10 (HuffPo headline+description,
    83,878 rows, 10 balanced category classes).

Output: data/langid/topic/{train,val,test}.tsv (lang\tlabel\ttext) +
topic_vocab.json. Stratified per class: 2,000 val + 2,000 test items per
language (quotas split by class count), remainder shuffled (seed 42) into
train with a per-language cap.
"""
import collections
import json
import os
import random

import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(BASE, "data", "langid", "raw")
OUT = os.path.join(BASE, "data", "langid", "topic")
SANAD_LABELS = ["Tech", "Culture", "Finance", "Sports", "Politics",
                "Religion", "Medical"]


def cleanup(t):
    t = t.replace(chr(13), " ").replace(chr(10), " ").replace(chr(9), " ")
    return " ".join(t.split())


def dedupe(rows, n=12):
    seen = set()
    kept = []
    for s in rows:
        key = " ".join(s.split()[:n]).lower()[:120]
        if key not in seen:
            seen.add(key)
            kept.append(s)
    return kept


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(42)
    val_rows, test_rows, train_rows = [], [], []

    # ---- Arabic: SANAD
    t = pq.read_table(os.path.join(RAW, "sanad.parquet"))
    ar_by = collections.defaultdict(list)
    for tx, lab in zip(t.column("Article").to_pylist(),
                       t.column("label").to_pylist()):
        s = cleanup(tx)
        if len(s) >= 30:
            ar_by[SANAD_LABELS[lab]].append(s)
    ar_by = {k: dedupe(v) for k, v in ar_by.items()}
    print("ar per class:", {k: len(v) for k, v in sorted(ar_by.items())})

    # ---- English: HuffPo top-10 jsonl
    en_by = collections.defaultdict(list)
    for line in open(os.path.join(RAW, "huff_top10.json"), encoding="utf-8"):
        r = json.loads(line)
        s = cleanup((r.get("headline") or "") + " " + (r.get("short_description") or ""))
        if len(s) >= 30:
            en_by[r["category"]].append(s)
    en_by = {k: dedupe(v) for k, v in en_by.items()}
    print("en per class:", {k: len(v) for k, v in sorted(en_by.items())})

    labels = ["ar:" + k for k in ar_by] +              ["en:" + k.lower().replace(" & ", "_") for k in en_by]
    lab2id = {l: i for i, l in enumerate(labels)}

    for lang, pool_by, cap in (("ar", ar_by, 60000), ("en", en_by, 60000)):
        ncls = len(pool_by)
        vq = max(1, 2000 // ncls)
        tq = max(1, 2000 // ncls)
        tr = []
        for k, rows2 in sorted(pool_by.items()):
            random.shuffle(rows2)
            for i, s in enumerate(rows2):
                lab = lang + ":" + (k.lower().replace(" & ", "_") if lang == "en" else k)
                if i < vq:
                    val_rows.append((lang, lab, s))
                elif i < vq + tq:
                    test_rows.append((lang, lab, s))
                else:
                    tr.append((lang, lab, s))
        random.shuffle(tr)
        train_rows.extend(tr[:cap])

    for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        random.shuffle(rows)
        with open(os.path.join(OUT, name + ".tsv"), "w", encoding="utf-8") as f:
            for lang, lab, s in rows:
                f.write("%s\t%s\t%s\n" % (lang, lab2id[lab], s.replace("\t", " ")))
    with open(os.path.join(OUT, "topic_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=1)
    print("labels:", len(labels))
    print("train", len(train_rows), "val", len(val_rows), "test", len(test_rows))


if __name__ == "__main__":
    main()
