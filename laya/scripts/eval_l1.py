r"""Laya-line L1 evaluation: official test-split metrics + pre-registered gates.

Gates (research/EXPERIMENTS.md E-62, judged on the FINAL checkpoint):
  G1 feasibility: peak allocated VRAM <= 4096 MiB during training
     (from train_summary.json; allocated-only, shared spill N/A on this run).
  G2 capability:  test accuracy >= 0.587 (MiniLM-L6 22M specialist baseline).
  G3 diagnostic:  post-hoc temperature (fit on a train calibration slice,
     per (qtype, n_options) group) must not increase test ECE.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_l1.py --config configs/laya_l1.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, collate  # noqa: E402
from laya_head import QTYPES  # noqa: E402


def ece(conf, corr, bins=15):
    n = len(conf)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i in range(n) if conf[i] > lo and
               (conf[i] <= hi or (b == bins - 1 and conf[i] <= 1.0))]
        if not sel:
            continue
        acc = sum(corr[i] for i in sel) / len(sel)
        c = sum(conf[i] for i in sel) / len(sel)
        total += len(sel) / n * abs(acc - c)
    return total


def collect_logits(model, items, pad_id, device, micro_batch=16):
    """Per-decision records: (logits list, target list, group key, meta)."""
    model.eval()
    recs = []
    with torch.no_grad():
        for i in range(0, len(items), micro_batch):
            chunk = items[i:i + micro_batch]
            batch = collate(chunk, pad_id)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["marker_pos"], batch["marker_batch"],
                               batch["qtype"])
            logits = logits.float().cpu()
            ofs = 0
            for b, cnt in enumerate(batch["n_options"]):
                it = chunk[b]
                recs.append({
                    "z": logits[ofs:ofs + cnt].tolist(),
                    "tgt": batch["targets"][ofs:ofs + cnt].tolist(),
                    "group": "%s|%d" % (it["qtype_name"], cnt),
                    "qtype": it["qtype_name"], "workflow": it["workflow"],
                    "gold_idx": int(it["gold_idx"]),
                    "case_id": it["case_id"], "qname": it["qname"],
                })
                ofs += cnt
    model.train()
    return recs


def metrics_from(recs, temps=None):
    conf, corr = [], []
    per_qt, per_wf = {}, {}
    score_mae, score_within, score_n = 0.0, 0, 0
    soft_sum, brier_sum = 0.0, 0.0
    for r in recs:
        z = torch.tensor(r["z"])
        T = 1.0 if temps is None else temps.get(r["group"], 1.0)
        p = torch.softmax(z / T, dim=-1)
        pred = int(p.argmax().item())
        gold = r["gold_idx"]
        ok = int(pred == gold)
        conf.append(float(p[pred].item()))
        corr.append(ok)
        soft_sum += float(p[gold].item())
        onehot = torch.zeros_like(p)
        onehot[gold] = 1.0
        brier_sum += float(((p - onehot) ** 2).mean().item())
        for key, store in ((r["qtype"], per_qt), (r["workflow"], per_wf)):
            s = store.setdefault(key, {"n": 0, "correct": 0})
            s["n"] += 1
            s["correct"] += ok
        if r["qtype"] == "score":
            idx = torch.arange(len(p), dtype=torch.float)
            pred_exp = float((p * idx).sum().item())
            gold_exp = float((torch.tensor(r["tgt"]) * idx).sum().item())
            d = abs(pred_exp - gold_exp)
            score_mae += d
            score_within += int(d <= 1.0 + 1e-6)
            score_n += 1
    n = max(1, len(recs))
    out = {
        "n": n,
        "acc": sum(corr) / n,
        "soft_acc": soft_sum / n,
        "brier_mean_over_classes": brier_sum / n,
        "ece_raw": ece(conf, corr),
        "per_qtype": {k: {"n": v["n"], "acc": v["correct"] / v["n"]}
                      for k, v in per_qt.items()},
        "per_workflow": {k: {"n": v["n"], "acc": v["correct"] / v["n"]}
                         for k, v in per_wf.items()},
    }
    if score_n:
        out["score_exp_index_mae"] = score_mae / score_n
        out["score_within1"] = score_within / score_n
    return out


def fit_temperatures(recs):
    """Grid-fit one temperature per (qtype, n_options) group on NLL."""
    groups = {}
    for r in recs:
        groups.setdefault(r["group"], []).append(r)
    temps = {}
    grid = [math.exp(x) for x in
            [math.log(0.2) + i * (math.log(5.0) - math.log(0.2)) / 63
             for i in range(64)]]
    for g, rs in groups.items():
        best_T, best_nll = 1.0, None
        for T in grid:
            nll = 0.0
            for r in rs:
                lp = torch.log_softmax(torch.tensor(r["z"]) / T, dim=-1)
                nll -= float((torch.tensor(r["tgt"]) * lp).sum().item())
            nll /= len(rs)
            if best_nll is None or nll < best_nll:
                best_nll, best_T = nll, T
        temps[g] = {"T": round(best_T, 4), "nll": round(best_nll, 4),
                    "n": len(rs)}
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l1.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="defaults to <out_dir>/final/model.pt")
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["out_dir"]
    ckpt = args.ckpt or os.path.join(out_dir, "final", "model.pt")
    device = "cuda"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = LayaDecisionModel(cfg["encoder"], head_layers=int(cfg["head_layers"]),
                              dropout=float(cfg["dropout"])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))

    test_items = torch.load(cfg["data_test"], weights_only=False)
    train_items = torch.load(cfg["data_train"], weights_only=False)
    calib = train_items[::15][:400]  # notebook-style calibration slice
    print("[eval] test %d decisions, calib %d" % (len(test_items), len(calib)),
          flush=True)

    test_recs = collect_logits(model, test_items, tok.pad_token_id, device)
    m = metrics_from(test_recs)

    temps = fit_temperatures(collect_logits(model, calib, tok.pad_token_id, device))
    m["ece_temp"] = metrics_from(test_recs, temps={k: v["T"] for k, v in temps.items()})["ece_raw"]
    m["temperatures"] = temps

    # ---- pre-registered gates ----
    summary_path = os.path.join(out_dir, "train_summary.json")
    gates = {}
    if os.path.exists(summary_path):
        with io.open(summary_path, "r", encoding="utf-8") as f:
            ts = json.load(f)
        peak = ts.get("peak_vram_mib", 1e9)
        gates["G1_vram"] = {"actual": peak, "required": "<= 4096 MiB",
                            "verdict": "PASS" if peak <= 4096 else "FAIL"}
    else:
        gates["G1_vram"] = {"verdict": "SKIPPED", "reason": "train_summary.json missing"}
    gates["G2_accuracy"] = {"actual": round(m["acc"], 4), "required": ">= 0.587",
                            "verdict": "PASS" if m["acc"] >= 0.587 else "FAIL"}
    gates["G3_temp_calibration"] = {
        "actual_raw": round(m["ece_raw"], 4), "actual_temp": round(m["ece_temp"], 4),
        "required": "temp_ece <= raw_ece",
        "verdict": "PASS" if m["ece_temp"] <= m["ece_raw"] + 1e-6 else "FAIL"}

    verdict = "PASS" if all(g.get("verdict") == "PASS"
                            for g in gates.values()) else "FAIL"
    report = {
        "experiment": "E-62 (laya-line L1)", "checkpoint": ckpt,
        "baselines": {"majority": 0.461, "MiniLM-L6_22M": 0.587,
                      "ModernBERT-base_149M": 0.646, "Jev_generalist": 0.727,
                      "teacher_ceiling": 0.735, "Laya-421M": 0.766},
        "metrics": m, "gates": gates, "verdict": verdict,
        "notes": ["Brier = mean over classes per decision, averaged over decisions.",
                  "score metrics use expected level index (0-based) predicted vs gold.",
                  "ECE bins = 15 on argmax confidence vs argmax correctness "
                  "(confidence-accuracy pairing caveat from the Luni critique applies)."],
    }
    path = os.path.join(out_dir, "eval_report.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"acc": round(m["acc"], 4), "soft_acc": round(m["soft_acc"], 4),
                      "brier": round(m["brier_mean_over_classes"], 4),
                      "ece_raw": round(m["ece_raw"], 4),
                      "ece_temp": round(m["ece_temp"], 4),
                      "per_qtype": m["per_qtype"], "per_workflow": m["per_workflow"],
                      "gates": gates, "verdict": verdict}, indent=1), flush=True)
    print("[eval] wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()
