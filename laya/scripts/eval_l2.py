r"""Laya-line L2 evaluation (E-63 gates): mixture heldout macro, typed test, probes.

Gates (research/EXPERIMENTS.md E-63, judged on runs/laya/l2/final/model.pt):
  G1: mixture heldout macro accuracy >= 0.74.
  G2: typed-decisions test accuracy >= 0.60.
  G3: permutation-invariance probe agreement >= 0.95 for the L2 model
      (recorded for the E-62 L1 model too, as the pre-registered comparison).

Diagnostic: ONE global temperature fit on 2000 typed train items (E-62
lesson: no per-group fits on small slices); raw vs temp ECE reported.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_l2.py --config configs/laya_l2.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_head import LayaDecisionModel, collate  # noqa: E402
import train_l1 as t1  # noqa: E402
from eval_l1 import ece  # noqa: E402


def load_model(cfg, ckpt, device):
    model = LayaDecisionModel(cfg["encoder"], head_layers=int(cfg["head_layers"]),
                              dropout=float(cfg["dropout"])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model


def predict(model, items, pad_id, device):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            batch = collate(chunk, pad_id)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["marker_pos"], batch["marker_batch"], batch["qtype"])
            logits = logits.float().cpu()
            ofs = 0
            for b, n in enumerate(batch["n_options"]):
                p = torch.softmax(logits[ofs:ofs + n], dim=-1)
                preds.append(int(p.argmax().item()))
                confs.append(float(p.max().item()))
                ofs += n
    model.train()
    return preds, confs


def permuted_items(items, tok, n=200, seed=7):
    """Reverse-shuffle the option blocks inside input_ids; adjusted copies.

    Blocks are contiguous [MASK]..(next [MASK] or the [SEP] before state);
    swapping them is exact. Targets/markers permuted with the same mapping.
    """
    rng = random.Random(seed)
    sep = tok.sep_token_id
    out, origs = [], []
    cand = [it for it in items if it["qtype_name"] == "choice" and it["n_options"] >= 2]
    rng.shuffle(cand)
    for it in cand[:n]:
        ids = it["input_ids"]
        mp = it["marker_pos"]
        n_opt = it["n_options"]
        spans = []
        for k in range(n_opt):
            start = mp[k]
            end = mp[k + 1] if k + 1 < n_opt else ids.index(sep, start + 1)
            spans.append((start, end))
        prefix = ids[:spans[0][0]]
        tail = ids[spans[-1][1]:]
        blocks = [ids[s:e] for (s, e) in spans]
        perm = list(range(n_opt))
        rng.shuffle(perm)
        new_ids, new_mp, pos = [], [], len(prefix)
        for p in perm:
            nb = blocks[p]
            new_mp.append(pos)
            new_ids += nb
            pos += len(nb)
        new_ids = prefix + new_ids + tail
        new_target = [it["target"][perm[j]] for j in range(n_opt)]
        out.append({"input_ids": new_ids, "marker_pos": new_mp,
                    "n_options": n_opt, "qtype": it["qtype"],
                    "target": new_target,
                    "gold_idx": max(range(n_opt), key=lambda i: new_target[i]),
                    "workflow": it["workflow"], "case_id": it["case_id"],
                    "qname": it["qname"], "qtype_name": it["qtype_name"]})
        origs.append(it)
    return out, origs


def agreement(model, items_a, items_b, pad_id, device):
    pa, _ = predict(model, items_a, pad_id, device)
    pb, _ = predict(model, items_b, pad_id, device)
    return sum(int(a == b) for a, b in zip(pa, pb)) / max(1, len(pa))


def global_temp(model, items, pad_id, device, n=2000):
    subset = items[:n]
    zs, tgts = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(subset), 16):
            chunk = subset[i:i + 16]
            batch = collate(chunk, pad_id)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["marker_pos"], batch["marker_batch"], batch["qtype"])
            logits = logits.float().cpu()
            ofs = 0
            for b, cnt in enumerate(batch["n_options"]):
                zs.append(logits[ofs:ofs + cnt].tolist())
                tgts.append(batch["targets"][ofs:ofs + cnt].tolist())
                ofs += cnt
    model.train()
    best_T, best_nll = 1.0, None
    for gi in range(64):
        T = math.exp(math.log(0.2) + gi * (math.log(5.0) - math.log(0.2)) / 63)
        nll = 0.0
        for z, t in zip(zs, tgts):
            lp = torch.log_softmax(torch.tensor(z) / T, dim=-1)
            nll -= float((torch.tensor(t) * lp).sum().item())
        nll /= len(zs)
        if best_nll is None or nll < best_nll:
            best_nll, best_T = nll, T
    return round(best_T, 4)


def ece_at(model, items, pad_id, device, T=1.0):
    preds, confs = predict(model, items, pad_id, device)
    corr = [int(p == it["gold_idx"]) for p, it in zip(preds, items)]
    acc = sum(corr) / max(1, len(corr))
    return acc, ece([c / T for c in confs], corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l2.yaml")
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    out_dir = cfg["out_dir"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id

    model = load_model(cfg, os.path.join(out_dir, "final", "model.pt"), device)
    mixture_held = torch.load(cfg["mixture_heldout"], weights_only=False)
    typed_train = torch.load(cfg["typed_train"], weights_only=False)
    typed_test = torch.load(cfg["typed_test"], weights_only=False)

    by_src = {}
    for it in mixture_held:
        by_src.setdefault(it["workflow"], []).append(it)
    per_src = {s: round(t1.evaluate(model, its, pad_id, device)["acc"], 4)
               for s, its in sorted(by_src.items())}
    macro = round(sum(per_src.values()) / len(per_src), 4)

    preds, confs = predict(model, typed_test, pad_id, device)
    typed_acc = sum(int(p == it["gold_idx"]) for p, it in
                    zip(preds, typed_test)) / len(typed_test)
    ev = t1.evaluate(model, typed_test, pad_id, device)

    T = global_temp(model, typed_train, pad_id, device)
    _, ece_raw = ece_at(model, typed_test, pad_id, device, T=1.0)
    _, ece_T = ece_at(model, typed_test, pad_id, device, T=T)

    perm_items, orig_items = permuted_items(typed_test, tok, n=200)
    agree_l2 = agreement(model, orig_items, perm_items, pad_id, device)
    agree_l1 = None
    l1_ckpt = cfg.get("l1_model")
    if l1_ckpt and os.path.exists(l1_ckpt):
        m1 = load_model(cfg, l1_ckpt, device)
        agree_l1 = agreement(m1, orig_items, perm_items, pad_id, device)
        del m1
        torch.cuda.empty_cache()

    gates = {
        "G1_mixture_macro": {"actual": macro, "required": ">= 0.74",
                             "verdict": "PASS" if macro >= 0.74 else "FAIL"},
        "G2_typed_acc": {"actual": round(typed_acc, 4), "required": ">= 0.60",
                         "verdict": "PASS" if typed_acc >= 0.60 else "FAIL"},
        "G3_perm_invariance": {"actual_l2": round(agree_l2, 4),
                               "actual_l1": None if agree_l1 is None else round(agree_l1, 4),
                               "required": ">= 0.95",
                               "verdict": "PASS" if agree_l2 >= 0.95 else "FAIL"},
    }
    report = {
        "experiment": "E-63 (laya-line L2)",
        "checkpoint": os.path.join(out_dir, "final", "model.pt"),
        "mixture_heldout": {"per_source": per_src, "macro": macro, "n": len(mixture_held)},
        "typed_test": {"acc": round(typed_acc, 4), "soft_acc": round(ev["soft_acc"], 4),
                       "brier": round(ev["brier"], 4), "n": len(typed_test)},
        "calibration": {"ece_raw": round(ece_raw, 4), "ece_global_T": round(ece_T, 4),
                        "T": T, "note": "one global T on 2000 typed train items (E-62 lesson)"},
        "probe_permutation": {"n": len(orig_items), "agreement_l2": round(agree_l2, 4),
                              "agreement_l1": None if agree_l1 is None else round(agree_l1, 4),
                              "note": "option-block reshuffle, exact repack; higher = rename-stable"},
        "gates": gates,
        "verdict": "PASS" if all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL",
    }
    path = os.path.join(out_dir, "eval_report.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print("[eval-l2] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()
