r"""E-74 phase-2a: PhishNChips zero-shot under the SYMMETRIC packing.

Same Luni split, same Platt allowance, same metrics as eval_phish.py, but each
email is scored as an isolated option window:
  [CLS] <noul> Is this email a phishing or scam attempt? [SEP] m option state [SEP]
p_true = sigmoid(raw score of the 'true' option). No cross-option positions.
Published comparators: Laya AUROC 0.678, Jev 0.689.
"""
import json, os, sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, pack_sequence, collate, qtype_id
import eval_phish as ep

try:
    import yaml
    ENCODER = yaml.safe_load(open("configs/laya_l3_e74.yaml"))["encoder"]
except Exception:
    ENCODER = "microsoft/MiniLM-L12-H384-uncased"
INSTR = "Is this email a phishing or scam attempt?"
QT = qtype_id("noul")
MODELS = [("l3_e72", "runs/laya/l3_e72/final/model.pt"),
          ("l3_e74_sym", "runs/laya/l3_e74/final/model.pt")]


def make_items_sym(rows, tok, opt_max):
    items = []
    for text, label in rows:
        ids, markers = pack_sequence(tok, QT, INSTR, ["true: the statement is true"], text,
                                     opt_max=opt_max)
        if len(markers) != 1:
            continue
        items.append({"input_ids": ids, "marker_pos": markers, "n_options": 1,
                      "qtype": QT, "target": [1.0] if int(label) == 1 else [0.0],
                      "gold_idx": int(label)})
    return items


@torch.no_grad()
def predict_p_true_sym(model, items, pad_id, device, micro=32):
    model.eval()
    out = []
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate(chunk, pad_id).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                      batch["marker_batch"], batch["qtype"]).float().view(-1)
        out.append(torch.sigmoid(s).cpu())
    return torch.cat(out).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("AreLit/PhishNChips", "emails", split="core")
    rows = [(ep.email_text(r), int(r["phish_label"])) for r in ds]
    y = np.array([lab for _, lab in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows)); half = len(rows) // 2
    cal_i, test_i = idx[:half], idx[half:]
    tok = AutoTokenizer.from_pretrained(ENCODER)
    items = make_items_sym(rows, tok, 256)
    print("[phish-sym] %d items packed (of %d rows)" % (len(items), len(rows)), flush=True)
    pad_id = tok.pad_token_id
    report = {"dataset": "AreLit/PhishNChips core", "packing": "symmetric-isolated",
              "n": len(rows), "protocol": "Luni bench_platt.py replication, zero-shot",
              "models": {}}
    for name, path in MODELS:
        if not os.path.exists(path):
            print("[phish-sym] skip %s (missing %s)" % (name, path), flush=True)
            continue
        model = LayaDecisionModel(ENCODER)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd if not isinstance(sd, dict) or "model" not in sd else sd["model"])
        model = model.to(device).eval()
        p = predict_p_true_sym(model, items, pad_id, device)
        a, b = ep.fit_platt(p[cal_i], y[cal_i])
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        q = 1 / (1 + np.exp(-(a * z + b)))
        m = {"raw_acc_test_half": float(((p[test_i] >= 0.5).astype(int) == y[test_i]).mean()),
             "platt_acc_test_half": float(((q[test_i] >= 0.5).astype(int) == y[test_i]).mean()),
             "auroc_test_half": float(ep.auroc(y[test_i], p[test_i])),
             "ece15_test_half": ep.ece15(y[test_i].astype(float), p[test_i]),
             "raw_acc_full": float(((p >= 0.5).astype(int) == y).mean()),
             "platt_a": float(a), "platt_b": float(b), "mean_p": float(p.mean())}
        report["models"][name] = m
        print("[phish-sym] %-11s raw %.3f platt %.3f auroc %.3f ece %.3f"
              % (name, m["raw_acc_test_half"], m["platt_acc_test_half"],
                 m["auroc_test_half"], m["ece15_test_half"]), flush=True)
        del model; torch.cuda.empty_cache()
    report["published"] = {"laya_auroc": 0.678, "jev_auroc": 0.689}
    with open("runs/laya/phish_eval_sym.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[phish-sym] saved runs/laya/phish_eval_sym.json", flush=True)


if __name__ == "__main__":
    main()
