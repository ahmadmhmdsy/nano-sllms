r"""Build Laya-line packed datasets from LocalLLaMA/typed-decisions.

Downloads the official benchmark (config "all": 1,200 train / 400 test
cases), packs one sequence per (case, question) with [MASK] option markers,
and saves torch .pt files consumed by train_l1.py / eval_l1.py.
CPU + network only - safe while a GPU train job runs.

Usage: & .\.venv\Scripts\python.exe laya/scripts/build_data.py --config configs/laya_l1.yaml
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


def load_json(v):
    if isinstance(v, str):
        return json.loads(v)
    return v


def question_spec(q, gold_entry, row, qname):
    """Return (qtype, instructions, option_texts, target) or None if the row's
    gold cannot be aligned (counted as a skip - never guessed)."""
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, dict):
            labels = list(crit.keys())
            descs = [str(crit[k]) for k in labels]
        elif isinstance(crit, list) and crit and isinstance(crit[0], dict):
            labels = [str(c.get("label", i)) for i, c in enumerate(crit)]
            descs = [str(c.get("description", "")) for c in crit]
        elif isinstance(crit, list):
            labels = [str(c) for c in crit]
            descs = ["" for _ in crit]
        else:
            return None
    elif t == "score":
        levels = crit if isinstance(crit, list) and len(crit) > 0 else None
        if levels is None:
            levels = ["level %d" % i for i in range(4)]
        labels = ["level %d" % i for i in range(len(levels))]
        descs = [str(c) for c in levels]
    elif t == "noul":
        labels = ["false", "true"]
        descs = ["the statement is false", "the statement is true"]
    else:
        return None

    probs = None
    if isinstance(gold_entry, dict):
        probs = gold_entry.get("probabilities")
    if probs is None:
        col = row.get(qname + "__probabilities")
        probs = load_json(col) if col is not None else None
    if t == "noul" and not (isinstance(probs, dict) and all(l in probs for l in labels)):
        p_true = None
        if isinstance(gold_entry, dict):
            p_true = gold_entry.get("probability_true")
        if p_true is None:
            col = row.get(qname + "__probability_true")
            p_true = load_json(col) if col is not None else None
        if p_true is not None:
            probs = [1.0 - float(p_true), float(p_true)]
    if probs is None:
        return None
    if isinstance(probs, dict):
        if all(l in probs for l in labels):
            tgt = [float(probs[l]) for l in labels]
        elif all(str(i) in probs for i in range(len(labels))):
            tgt = [float(probs[str(i)]) for i in range(len(labels))]
        else:
            return None
    else:
        tgt = [float(x) for x in probs]
    if len(tgt) != len(labels):
        return None
    if any(v != v or v < 0.0 for v in tgt):
        return None
    s = sum(tgt)
    if s <= 0:
        return None
    tgt = [v / s for v in tgt]
    option_texts = ["%s: %s" % (l, d) if d else str(l) for l, d in zip(labels, descs)]
    instructions = q.get("instructions", "")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return t, instructions, option_texts, tgt


def dump_schema(row, questions, gold):
    print("[build] SCHEMA DUMP (skips > 20%):", flush=True)
    for k, v in row.items():
        print("  %s = %s" % (k, str(v)[:220]), flush=True)
    print("  questions: %s" % str(questions)[:600], flush=True)
    print("  gold: %s" % str(gold)[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l1.yaml")
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    ds = load_dataset("LocalLLaMA/typed-decisions", "all")
    out_dir = os.path.join("data", "laya", "typed_decisions")
    os.makedirs(out_dir, exist_ok=True)

    stats = {}
    for split in ("train", "test"):
        items, skipped = [], 0
        rows = ds[split]
        for ri, row in enumerate(rows):
            state = load_json(row.get("state"))
            if isinstance(state, dict):
                state_text = json.dumps(state, ensure_ascii=False)
            else:
                state_text = str(state or "")
            questions = load_json(row.get("questions")) or {}
            gold = load_json(row.get("gold")) or {}
            workflow = str(row.get("workflow", "?"))
            case_id = str(row.get("id", ri))
            for qname, q in questions.items():
                if not isinstance(q, dict) or "type" not in q:
                    skipped += 1
                    continue
                ge = gold.get(qname) if isinstance(gold, dict) else None
                spec = question_spec(q, ge, row, qname)
                if spec is None:
                    skipped += 1
                    continue
                t, instructions, option_texts, tgt = spec
                ids, markers = pack_sequence(
                    tok, qtype_id(t), instructions, option_texts, state_text,
                    max_len=int(cfg["max_len"]), head_max_len=int(cfg["head_max_len"]),
                    opt_max=int(cfg["option_token_max"]))
                if len(markers) != len(tgt):
                    skipped += 1
                    continue
                items.append({
                    "input_ids": ids, "marker_pos": markers,
                    "n_options": len(tgt), "qtype": qtype_id(t),
                    "target": tgt,
                    "gold_idx": max(range(len(tgt)), key=lambda i: tgt[i]),
                    "workflow": workflow, "case_id": case_id,
                    "qname": qname, "qtype_name": t,
                    "instructions": instructions,
                    "option_texts": [str(o) for o in option_texts],
                    "state_text": state_text,
                })
        if items and skipped > 0.2 * max(1, len(items)):
            row0 = rows[0]
            dump_schema(row0, load_json(row0.get("questions")) or {},
                        load_json(row0.get("gold")) or {})
        torch.save(items, os.path.join(out_dir, split + ".pt"))
        n_q = {}
        for it in items:
            n_q[it["qtype_name"]] = n_q.get(it["qtype_name"], 0) + 1
        lens = sorted(len(it["input_ids"]) for it in items)
        stats[split] = {"cases": len(rows), "items": len(items), "skipped": skipped,
                        "by_qtype": n_q,
                        "len_p50": lens[len(lens) // 2] if lens else 0,
                        "len_max": lens[-1] if lens else 0}
        print("[build] %s: %d items (%d skipped) %s len_p50=%d len_max=%d" %
              (split, len(items), skipped, n_q, stats[split]["len_p50"],
               stats[split]["len_max"]), flush=True)
    with io.open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print("[build] done -> %s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
