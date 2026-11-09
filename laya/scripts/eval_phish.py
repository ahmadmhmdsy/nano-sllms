r"""E-64a: zero-shot PhishNChips eval, replicating Luni's bench_platt.py protocol.

The only phishing benchmark with PUBLISHED numbers for both Laya and Jev:
  Laya base raw 0.505 / Platt-fitted 0.611 / AUROC 0.678
  Jev 1.13.0 raw 0.626 / ECE 0.154 / AUROC 0.689
Our models never saw any PhishNChips item (not in the L2 mixture, not in
typed-decisions) - pure zero-shot. The Platt calibration half is the same
2-parameter allowance Luni gave Laya; it is never a training set.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_phish.py
"""
import json, os, sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, pack_sequence, collate, qtype_id
import train_l1 as t1

ENCODER = "microsoft/MiniLM-L12-H384-uncased"
INSTR = "Is this email a phishing or scam attempt?"
OPTS = ["false: the statement is false", "true: the statement is true"]
QT = qtype_id("noul")
MODELS = [("l1_final", "runs/laya/l1/final/model.pt"),
          ("l2_stageA", "runs/laya/l2/A_last.pt"),
          ("l2_final", "runs/laya/l2/final/model.pt")]


def email_text(r):
    c = r["email_content"]
    try:
        d = json.loads(c)
        if isinstance(d, dict):
            return "\n".join("%s: %s" % (k, v) for k, v in d.items()
                              if v not in (None, "", []))
    except Exception:
        pass
    return str(c)


def make_items(rows, tok):
    items = []
    for text, label in rows:
        ids, markers = pack_sequence(tok, QT, INSTR, OPTS, text)
        items.append({"input_ids": ids, "marker_pos": markers, "n_options": 2,
                      "qtype": QT,
                      "target": [1.0, 0.0] if int(label) == 0 else [0.0, 1.0],
                      "gold_idx": int(label)})
    return items


def load_model(path, device):
    model = LayaDecisionModel(ENCODER)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def predict_p_true(model, items, pad_id, device, micro=32):
    """P(option 'true') per item; items are uniform 2-option noul."""
    model.eval()
    out = []
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        _, logp, _ = t1.forward_batch(model, collate(chunk, pad_id), device)
        out.append(logp.view(len(chunk), 2).float().exp()[:, 1].cpu())
    return torch.cat(out).numpy()


def auroc(y, p):
    o = np.argsort(p); r = np.empty(len(p), float); r[o] = np.arange(1, len(p) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_platt(p, y, iters=600, lr=0.1):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b)))
        a -= lr * ((q - y) * z).mean()
        b -= lr * (q - y).mean()
    return a, b


def ece15(y, p):
    bins = np.linspace(0, 1, 16)
    e, n = 0.0, len(p)
    for i in range(15):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("AreLit/PhishNChips", "emails", split="core")
    rows = [(email_text(r), int(r["phish_label"])) for r in ds]
    y = np.array([lab for _, lab in rows])
    rng = np.random.default_rng(0)          # Luni's exact split
    idx = rng.permutation(len(rows)); half = len(rows) // 2
    cal_i, test_i = idx[:half], idx[half:]

    tok = AutoTokenizer.from_pretrained(ENCODER)
    items = make_items(rows, tok)
    pad_id = tok.pad_token_id

    report = {"dataset": "AreLit/PhishNChips core", "n": len(rows),
              "protocol": "Luni bench_platt.py replication, zero-shot",
              "models": {}}
    for name, path in MODELS:
        if not os.path.exists(path):
            print("[phish] skip %s (missing %s)" % (name, path), flush=True)
            continue
        model = load_model(path, device)
        p = predict_p_true(model, items, pad_id, device)
        a, b = fit_platt(p[cal_i], y[cal_i])
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        q = 1 / (1 + np.exp(-(a * z + b)))
        m = {
            "raw_acc_test_half": float(((p[test_i] >= 0.5).astype(int) == y[test_i]).mean()),
            "platt_acc_test_half": float(((q[test_i] >= 0.5).astype(int) == y[test_i]).mean()),
            "auroc_test_half": float(auroc(y[test_i], p[test_i])),
            "ece15_test_half": ece15(y[test_i].astype(float), p[test_i]),
            "raw_acc_full": float(((p >= 0.5).astype(int) == y).mean()),
            "platt_a": float(a), "platt_b": float(b),
            "mean_p": float(p.mean()),
        }
        report["models"][name] = m
        print("[phish] %-10s raw %.3f  platt %.3f  auroc %.3f  ece %.3f"
              % (name, m["raw_acc_test_half"], m["platt_acc_test_half"],
                 m["auroc_test_half"], m["ece15_test_half"]), flush=True)
        del model
        torch.cuda.empty_cache()

    report["published"] = {"laya_raw": 0.505, "laya_platt": 0.611,
                           "laya_auroc": 0.678, "jev_raw": 0.626, "jev_auroc": 0.689}
    os.makedirs("runs/laya", exist_ok=True)
    with open("runs/laya/phish_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[phish] saved runs/laya/phish_eval.json", flush=True)


if __name__ == "__main__":
    main()
