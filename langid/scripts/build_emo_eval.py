# Build DA-2 emo train/eval sets from ungated self-labeled tweet sources.
# Sources:
#   ar: arbml/TEAD (HF)  - self-labeled by author-chosen emoji, one-type-only filter
#   en: tweet_eval/emoji (HF, Apache? card says 'unknown'; research use)
# Output: data/langid/emo/{train.tsv,val.tsv,test.tsv} + emo_vocab.json

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import unicodedata

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (scripts -> langid -> root)
RAW = os.path.join(BASE, "data", "langid", "raw")
OUT = os.path.join(BASE, "data", "langid", "emo")

EMOJI_MIN = 0x1F000
EMOJI_SYM = (0x2600, 0x27BF)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def emoji_chars(s: str):
    return [ch for ch in s if ord(ch) >= EMOJI_MIN or EMOJI_SYM[0] <= ord(ch) <= EMOJI_SYM[1]]


def cleanup_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = WS_RE.sub(" ", text).strip()
    return text.lower()


def strip_emoji(text: str) -> str:
    return "".join(ch for ch in text if not (ord(ch) >= EMOJI_MIN or EMOJI_SYM[0] <= ord(ch) <= EMOJI_SYM[1]))


def load_tead_arabic(top_k: int):
    import pyarrow.parquet as pq
    path = os.path.abspath(os.path.join(RAW, "tead.parquet"))  # abspath: pyarrow rejects ".." path parts
    t = pq.read_table(path)
    rows = t.select(["text"]).to_pylist()
    counts = collections.Counter()
    by_type = collections.defaultdict(list)
    for r in rows:
        s = r["text"]
        em = emoji_chars(s)
        types = sorted(set(em))
        if len(types) == 1 and len(em) >= 1:
            label = types[0]
            text = strip_emoji(s)
            text = cleanup_text(text)
            if len(text) >= 10:
                counts[label] += 1
                by_type[label].append(text)
    keep = {lab for lab, _ in counts.most_common(top_k)}
    out = []
    for lab, texts in by_type.items():
        if lab in keep:
            for t2 in texts:
                out.append(("ar", t2, lab))
    return out, counts


def load_tweet_eval_en(top_k: int):
    import pyarrow.parquet as pq
    out = []
    # tweet_eval emoji: 20 classes, integer label; map to the emoji char per
    # official TweetEval emoji mapping (0..19)
    # Official TweetEval emoji class mapping, copied verbatim from the
    # cardiffnlp/tweet_eval dataset_info features (class_label.names) -
    # 20 emoji strings indexed 0..19 (Camacho-Collados et al. 2018).
    EMOJI_MAPPING = ["❤","😍","😂","💕","🔥","😊","😎","✨","💙","😘","📷","🇺🇸","☀","💜","😉","💯","😁","🎄","📸","😜"]
    counts = collections.Counter()
    for split in ["train", "validation"]:
        path = os.path.abspath(os.path.join(RAW, "tweet_eval_" + split + ".parquet"))
        t = pq.read_table(path)
        cols = t.schema.names
        print("tweet_eval", split, "cols", cols)
        text_col = "text" if "text" in cols else cols[0]
        lab_col = "label" if "label" in cols else cols[1]
        for r in t.select([text_col, lab_col]).to_pylist():
            s = r[text_col]
            idx = int(r[lab_col])
            if idx >= len(EMOJI_MAPPING):
                continue
            lab = EMOJI_MAPPING[idx]
            text = cleanup_text(s)
            if len(text) >= 10:
                counts[lab] += 1
                out.append(("en", text, lab))
    return out, counts


def load_dialects_arabic(top_k: int):
    # amgadhasan/arabic_tweets_dialects (HF, ungated): 147,725 Arabic tweets;
    # one-emoji-type rule survives on 34,514 rows / 435 distinct emojis.
    import pyarrow.parquet as pq
    path = os.path.abspath(os.path.join(RAW, "arabic_tweets_dialects.parquet"))
    t = pq.read_table(path)
    texts = t.column("text").to_pylist()
    counts = collections.Counter()
    by_type = collections.defaultdict(list)
    for s in texts:
        em = emoji_chars(s)
        types = sorted(set(em))
        if len(types) == 1 and len(em) >= 1:
            label = types[0]
            text = cleanup_text(strip_emoji(s))
            if len(text) >= 10:
                counts[label] += 1
                by_type[label].append(text)
    keep = {lab for lab, _ in counts.most_common(top_k)}
    out = []
    for lab, texts2 in by_type.items():
        if lab in keep:
            for t2 in texts2:
                out.append(("ar", t2, lab))
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import random

    ar_rows, ar_counts = load_tead_arabic(args.topk)
    ar_rows2, ar_counts2 = load_dialects_arabic(args.topk)
    print("dialect_a rows:", len(ar_rows2), "emoji types:", len(ar_counts2))
    ar_rows = ar_rows + ar_rows2
    en_rows, en_counts = load_tweet_eval_en(args.topk)
    print("ar rows:", len(ar_rows), "emoji types(tead):", len(ar_counts))
    print("en rows:", len(en_rows), "emoji types(tweet_eval):", len(en_counts))

    all_labels = sorted({lab for _, _, lab in ar_rows} | {lab for _, _, lab in en_rows})
    lab2id = {l: i for i, l in enumerate(all_labels)}
    print("total classes:", len(all_labels))

    rng = random.Random(args.seed)
    # stratified split: per (lang,label) split 80/10/10
    groups = collections.defaultdict(list)
    for r in ar_rows + en_rows:
        groups[(r[0], r[2])].append(r)
    splits = {"train": [], "val": [], "test": []}
    for k, lst in groups.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        splits["train"].extend(lst[:n_train])
        splits["val"].extend(lst[n_train:n_train + n_val])
        splits["test"].extend(lst[n_train + n_val:])
    for name, lst in splits.items():
        rng.shuffle(lst)
        # tab-safe: no internal tabs expected after cleanup
        p = os.path.join(OUT, name + ".tsv")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for lang, text, lab in lst:
                f.write(lang + "\t" + lab2id[lab].__str__() + "\t" + text + "\n")
        print(name, len(lst))
    with open(os.path.join(OUT, "emo_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=True, indent=1)
    print("vocab written; classes:", len(all_labels))


if __name__ == "__main__":
    main()
