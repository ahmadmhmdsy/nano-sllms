r"""Build the Laya-line L2 generalist mixture (E-63 pre-registration).

CPU + network only - safe while a GPU train job runs. Downloads public
datasets, packs Laya-format items (same pack_sequence as L1), saves
data/laya/mixture/{train,heldout}.pt (90/10 split per source).

Mixture (E-63, "~80k" target; BoolQ hard-caps at 9,427 so actual ~69k,
recorded honestly in build_stats.json):
  BoolQ 9.4k (noul) - SQuAD v2 10k (noul, has_answer) - SNLI 15k (choice-3)
  - MNLI 20k (choice-3) - ANLI 10k (choice-3, r1+r2+r3) - SciTail 5k (noul)
  - yelp_review_full 5k (score-5)

Usage: & .\.venv\Scripts\python.exe laya/scripts/build_data_l2.py --config configs/laya_l1.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import pack_sequence, qtype_id  # noqa: E402

CHOICE3 = [("entailment", "the statement follows from the passage"),
           ("neutral", "the statement is neither entailed nor contradicted"),
           ("contradiction", "the statement contradicts the passage")]
NOUL = [("false", "the statement is false"), ("true", "the statement is true")]
SCORE5 = [("level %d" % i, "%d-star rating" % (i + 1)) for i in range(5)]


def onehot(idx, n):
    t = [0.0] * n
    t[idx] = 1.0
    return t


def sample(ds, n):
    n = min(n, len(ds))
    return ds.shuffle(seed=42).select(range(n)) if n < len(ds) else ds


def build_source(tok, cfg, stats):
    from datasets import load_dataset
    items = []

    def emit(qtype, instructions, option_texts, target, state_text, source):
        if option_texts and isinstance(option_texts[0], (tuple, list)):
            option_texts = ["%s: %s" % (l, d) if d else str(l)
                            for l, d in option_texts]
        ids, markers = pack_sequence(tok, qtype_id(qtype), instructions,
                                     option_texts, state_text,
                                     max_len=int(cfg["max_len"]),
                                     head_max_len=int(cfg["head_max_len"]),
                                     opt_max=int(cfg["option_token_max"]))
        if len(markers) != len(target):
            return
        items.append({"input_ids": ids, "marker_pos": markers,
                      "n_options": len(target), "qtype": qtype_id(qtype),
                      "target": target,
                      "gold_idx": max(range(len(target)), key=lambda i: target[i]),
                      "workflow": source, "case_id": source,
                      "qname": source, "qtype_name": qtype,
                      "instructions": instructions,
                      "option_texts": option_texts,
                      "state_text": state_text})

    try:
        boolq = load_dataset("google/boolq", split="train")
        for r in sample(boolq, 9427):
            tgt = onehot(1 if r["answer"] else 0, 2)
            emit("noul", "Answer the yes/no question about the passage.",
                 NOUL, tgt, r["passage"], "boolq")
        stats["boolq"] = sum(1 for i in items if i["workflow"] == "boolq")
        print("[l2build] boolq ok", flush=True)
    except Exception as e:
        stats["boolq"] = "FAILED: %s" % str(e)[:120]
        print("[l2build] boolq FAILED %s" % e, flush=True)

    try:
        sq = None
        for _id in ("rajpurkar/squad_v2", "rajpur/squad_v2"):
            try:
                sq = load_dataset(_id, split="train")
                break
            except Exception:
                continue
        if sq is None:
            raise RuntimeError("squad_v2 not loadable")
        for r in sample(sq, 10000):
            impossible = len(r["answers"]["text"]) == 0
            tgt = onehot(0 if impossible else 1, 2)
            emit("noul", "Does the passage contain the answer to: %s" % r["question"],
                 NOUL, tgt, r["context"], "squad_v2")
        stats["squad_v2"] = sum(1 for i in items if i["workflow"] == "squad_v2")
        print("[l2build] squad_v2 ok", flush=True)
    except Exception as e:
        stats["squad_v2"] = "FAILED: %s" % str(e)[:120]
        print("[l2build] squad_v2 FAILED %s" % e, flush=True)

    for name, hf_id, n in (("snli", "stanfordnlp/snli", 15000),
                           ("mnli", "nyu-mll/glue", 20000),
                           ("anli", "facebook/anli", 10000)):
        try:
            if name == "mnli":
                ds = load_dataset(hf_id, "mnli", split="train")
                rows = [{"premise": r["premise"], "hypothesis": r["hypothesis"],
                         "label": r["label"]} for r in sample(ds, n)]
            elif name == "anli":
                ds = load_dataset(hf_id)
                merged = []
                for sp in ("train_r1", "train_r2", "train_r3"):
                    merged += list(ds[sp])
                rows = [{"premise": r["premise"], "hypothesis": r["hypothesis"],
                         "label": r["label"]} for r in merged]
                import random as _rd
                _rd.seed(42)
                _rd.shuffle(rows)
                rows = rows[:n]
            else:
                ds = load_dataset(hf_id, split="train")
                rows = [{"premise": r["premise"], "hypothesis": r["hypothesis"],
                         "label": r["label"]} for r in ds]
                rows = [r for r in rows if r["label"] in (0, 1, 2)]
                import random as _rd
                _rd.seed(42)
                _rd.shuffle(rows)
                rows = rows[:n]
            for r in rows:
                if r["label"] not in (0, 1, 2):
                    continue
                emit("choice", "Choose the relationship: %s" % r["hypothesis"],
                     CHOICE3, onehot(r["label"], 3), r["premise"], name)
            stats[name] = sum(1 for i in items if i["workflow"] == name)
            print("[l2build] %s ok" % name, flush=True)
        except Exception as e:
            stats[name] = "FAILED: %s" % str(e)[:120]
            print("[l2build] %s FAILED %s" % (name, e), flush=True)

    try:
        sc = load_dataset("allenai/scitail", "snli_format", split="train")
        cols = sc.column_names
        prem = "premise" if "premise" in cols else "sentence1"
        hyp = "hypothesis" if "hypothesis" in cols else "sentence2"
        stats["scitail_columns"] = str(cols)
        for r in sample(sc, 5000):
            lab = r.get("gold_label", r.get("label", ""))
            lstr = str(lab).lower()
            pos = 1 if lstr.startswith(("entails", "supports")) else 0
            emit("noul", "Does the first statement support the second: %s" % r[hyp],
                 NOUL, onehot(pos, 2), r[prem], "scitail")
        stats["scitail"] = sum(1 for i in items if i["workflow"] == "scitail")
        print("[l2build] scitail ok", flush=True)
    except Exception as e:
        stats["scitail"] = "FAILED: %s" % str(e)[:120]
        print("[l2build] scitail FAILED %s" % e, flush=True)

    try:
        yp = None
        for _id in ("Yelp/yelp_review_full", "yelp_review_full"):
            try:
                yp = load_dataset(_id, split="train")
                break
            except Exception:
                continue
        if yp is None:
            raise RuntimeError("yelp_review_full not loadable")
        for r in sample(yp, 5000):
            emit("score", "Rate the review quality from 1 to 5 stars.",
                 SCORE5, onehot(r["label"], 5), r["text"], "yelp5")
        stats["yelp5"] = sum(1 for i in items if i["workflow"] == "yelp5")
        print("[l2build] yelp5 ok", flush=True)
    except Exception as e:
        stats["yelp5"] = "FAILED: %s" % str(e)[:120]
        print("[l2build] yelp5 FAILED %s" % e, flush=True)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l1.yaml")
    ap.add_argument("--out", default="data/laya/mixture",
                    help="output dir (use data/laya/mixture_raw to keep the E-63 baseline intact)")
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    stats = {}
    items = build_source(tok, cfg, stats)

    # deterministic 90/10 per source (seeded by item order within source)
    train, held = [], []
    counters = {}
    for it in items:
        c = counters.get(it["workflow"], 0)
        counters[it["workflow"]] = c + 1
        (held if c % 10 == 9 else train).append(it)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(held, os.path.join(out_dir, "heldout.pt"))
    lens = sorted(len(i["input_ids"]) for i in items)
    stats["_total"] = len(items)
    stats["_train"] = len(train)
    stats["_heldout"] = len(held)
    stats["_len_p50"] = lens[len(lens) // 2] if lens else 0
    stats["_len_max"] = lens[-1] if lens else 0
    with io.open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print("[l2build] total %d (train %d / heldout %d) len_p50=%d -> %s" %
          (len(items), len(train), len(held), stats["_len_p50"], out_dir), flush=True)


if __name__ == "__main__":
    main()
