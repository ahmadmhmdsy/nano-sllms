r"""E-65 gates for the from-scratch L3 model + E-64 protocol reruns.

Pre-registered bars (research/EXPERIMENTS.md E-65 row):
  G1 mixture retention macro >= 0.74 post-stage-B (L2: 0.6856 post-B / 0.7478 exit-A)
  G2 typed acc >= 0.60 (L2 incumbent 0.6585; L1 0.6205)
  G3 permutation agreement >= 0.90 (L2 0.385, L1 0.325)
  G4 phishing AUROC >= 0.60 (Laya 0.678, Jev 0.689, ours L1 0.576)
Reruns the E-64 probe suite and AG News/emotion choice eval on the L3 model
for the comparison table. Eval sets are never trained on.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_l3.py --config configs/laya_l3.yaml
"""
import argparse, json, os, sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_l1 as t1
from train_l3 import eval_A
import eval_phish as ep
import eval_l2 as el2
import eval_probes as epr
import eval_public_choice as epc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l3_ladder.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["out_dir"]
    model_path = args.model or os.path.join(out_dir, "final", "model.pt")
    assert os.path.exists(model_path), "missing model: %s" % model_path

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id
    # the reused eval modules construct models from their module-global ENCODER
    ep.ENCODER = cfg["encoder"]; epr.ENCODER = cfg["encoder"]; epc.ENCODER = cfg["encoder"]

    from laya_head import LayaDecisionModel
    model = LayaDecisionModel(cfg["encoder"])
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    model = model.to(device).eval()

    report = {"model": model_path, "encoder": cfg["encoder"], "gates": {}}

    # G1: mixture retention macro (post-stage-B)
    held = torch.load(cfg["mixture_heldout"], weights_only=False)
    m1 = eval_A(model, held, pad_id, device)
    report["gates"]["G1"] = {"metric": "mixture_macro", "value": m1["macro"],
                             "bar": 0.74, "pass": m1["macro"] >= 0.74,
                             "incumbent": {"l2_postB": 0.6856, "l2_exitA": 0.7478},
                             "per_source": m1["per_source"]}
    print("[G1] macro %.4f (bar 0.74)" % m1["macro"], flush=True)

    # G2: typed accuracy
    test = torch.load(cfg["typed_test"], weights_only=False)
    m2 = t1.evaluate(model, test, pad_id, device)
    acc2 = float(m2.get("acc", m2.get("accuracy", 0.0)))
    report["gates"]["G2"] = {"metric": "typed_acc", "value": acc2, "bar": 0.60,
                             "pass": acc2 >= 0.60,
                             "incumbent": {"l2": 0.6585, "l1": 0.6205}, "raw": m2}
    print("[G2] acc %.4f (bar 0.60)" % acc2, flush=True)

    # G3: option-permutation agreement
    perm, origs = el2.permuted_items(test, tok, n=200, seed=7)
    ag = el2.agreement(model, origs, perm, pad_id, device)
    report["gates"]["G3"] = {"metric": "perm_agreement", "value": ag, "bar": 0.90,
                             "pass": ag >= 0.90, "incumbent": {"l2": 0.385, "l1": 0.325}}
    print("[G3] agreement %.4f (bar 0.90)" % ag, flush=True)

    # G4: zero-shot phishing AUROC (Luni protocol verbatim)
    ds = load_dataset("AreLit/PhishNChips", "emails", split="core")
    rows = [(ep.email_text(r), int(r["phish_label"])) for r in ds]
    y = np.array([lab for _, lab in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows)); half = len(rows) // 2
    cal_i, test_i = idx[:half], idx[half:]
    items = ep.make_items(rows, tok)
    p = ep.predict_p_true(model, items, pad_id, device)
    a, b = ep.fit_platt(p[cal_i], y[cal_i])
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    q = 1 / (1 + np.exp(-(a * z + b)))
    g4 = {"raw_acc_test_half": float(((p[test_i] >= 0.5).astype(int) == y[test_i]).mean()),
          "auroc_test_half": float(ep.auroc(y[test_i], p[test_i])),
          "ece15_test_half": ep.ece15(y[test_i].astype(float), p[test_i])}
    report["gates"]["G4"] = {"metric": "phish_auroc", "value": g4["auroc_test_half"],
                             "bar": 0.60, "pass": g4["auroc_test_half"] >= 0.60,
                             "published": {"laya": 0.678, "jev": 0.689},
                             "ours_l1": 0.576, "raw": g4, "platt": [float(a), float(b)]}
    print("[G4] auroc %.4f (bar 0.60)" % g4["auroc_test_half"], flush=True)

    # E-64 protocol reruns for the comparison table
    report["probes"] = epr.run_model("l3", model_path, tok, pad_id, device)
    print("[probes] done", flush=True)
    report["public_choice"] = {}
    for suite in epc.SUITES:
        dsc = epc.load_any(suite["candidates"], suite["split"])
        rng2 = np.random.default_rng(42)
        idx2 = sorted(rng2.permutation(len(dsc))[:suite["n_eval"]].tolist())
        n_opt = len(suite["options"])
        from laya_head import pack_sequence, qtype_id
        qt = qtype_id("choice")
        citems, gold = [], []
        for i in idx2:
            r = dsc[int(i)]
            ids, markers = pack_sequence(tok, qt, epc.INSTR, suite["options"],
                                         str(r[suite["text_col"]]))
            lab = int(r[suite["label_col"]])
            gold.append(lab)
            tgt = [0.0] * n_opt; tgt[lab] = 1.0
            citems.append({"input_ids": ids, "marker_pos": markers, "n_options": n_opt,
                           "qtype": qt, "target": tgt, "gold_idx": lab})
        gold = np.array(gold)
        pp = epc.predict_probs(model, citems, n_opt, pad_id, device)
        pred = pp.argmax(1); conf = pp.max(1)
        report["public_choice"][suite["name"]] = {
            "acc": float((pred == gold).mean()),
            "ece15": epc.ece15(conf, (pred == gold).astype(float)), "n": len(gold)}
        print("[choice] %s acc %.3f" % (suite["name"], report["public_choice"][suite["name"]]["acc"]), flush=True)

    n_pass = sum(1 for g in report["gates"].values() if g["pass"])
    report["summary"] = {"gates_passed": n_pass, "gates_total": len(report["gates"]),
                         "verdict": "PASS" if n_pass == len(report["gates"]) else "FAIL"}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[eval_l3] %d/%d gates passed -> %s" %
          (n_pass, len(report["gates"]), os.path.join(out_dir, "eval_report.json")), flush=True)


if __name__ == "__main__":
    main()
