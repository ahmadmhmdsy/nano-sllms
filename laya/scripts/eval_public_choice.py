r"""E-64b: zero-shot choice eval on AG News + dair-ai/emotion.

Both appear in the vendor table (NandhaKishorM/laya BENCHMARKS.md) with
published numbers for Laya variants AND Jev:
  AG News  Laya 0.950/0.930/0.953 vs Jev 0.910
  emotion  Laya 0.595/0.530/0.600 vs Jev 0.480
Caveat recorded up front: vendor prompts/sample sizes were not published, so
this is an approximate comparison (our standard choice format, deterministic
2000-item test samples, seed 42). Test splits only - never trained on.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_public_choice.py
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
MODELS = [("l1_final", "runs/laya/l1/final/model.pt"),
          ("l2_stageA", "runs/laya/l2/A_last.pt"),
          ("l2_final", "runs/laya/l2/final/model.pt")]
SUITES = [
    {"name": "ag_news", "candidates": ["fancyzhx/ag_news", "ag_news"],
     "split": "test", "text_col": "text", "label_col": "label",
     "options": ["World", "Sports", "Business", "Sci/Tech"], "n_eval": 2000},
    {"name": "emotion", "candidates": ["dair-ai/emotion", "emotion"],
     "split": "test", "text_col": "text", "label_col": "label",
     "options": ["sadness", "joy", "love", "anger", "fear", "surprise"],
     "n_eval": 2000},
]
INSTR = "Pick the correct category for the text."


def load_any(cands, split):
    last = None
    for rid in cands:
        try:
            return load_dataset(rid, split=split)
        except Exception as e:
            last = e
    raise RuntimeError("no candidate loadable: %s (%s)" % (cands, last))


def load_model(path, device):
    model = LayaDecisionModel(ENCODER)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, items, n_opt, pad_id, device, micro=32):
    model.eval()
    out = []
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        _, logp, _ = t1.forward_batch(model, collate(chunk, pad_id), device)
        out.append(logp.view(len(chunk), n_opt).float().exp().cpu())
    return torch.cat(out).numpy()


def ece15(conf, correct):
    bins = np.linspace(0, 1, 16)
    e, n = 0.0, len(conf)
    for i in range(15):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ENCODER)
    pad_id = tok.pad_token_id
    report = {"suites": {}, "caveat": "vendor prompts unpublished; approximate comparison"}
    for suite in SUITES:
        ds = load_any(suite["candidates"], suite["split"])
        rng = np.random.default_rng(42)
        idx = sorted(rng.permutation(len(ds))[:suite["n_eval"]].tolist())
        n_opt = len(suite["options"])
        qt = qtype_id("choice")
        items = []
        gold = []
        for i in idx:
            r = ds[int(i)]
            ids, markers = pack_sequence(tok, qt, INSTR, suite["options"], str(r[suite["text_col"]]))
            lab = int(r[suite["label_col"]])
            gold.append(lab)
            tgt = [0.0] * n_opt
            tgt[lab] = 1.0
            items.append({"input_ids": ids, "marker_pos": markers,
                          "n_options": n_opt, "qtype": qt, "target": tgt,
                          "gold_idx": lab})
        gold = np.array(gold)
        report["suites"][suite["name"]] = {}
        for name, path in MODELS:
            if not os.path.exists(path):
                print("[%s] skip %s (missing %s)" % (suite["name"], name, path), flush=True)
                continue
            model = load_model(path, device)
            p = predict_probs(model, items, n_opt, pad_id, device)
            pred = p.argmax(1); conf = p.max(1)
            m = {"acc": float((pred == gold).mean()),
                 "ece15": ece15(conf, (pred == gold).astype(float)),
                 "n": len(gold)}
            report["suites"][suite["name"]][name] = m
            print("[%s] %-10s acc %.3f  ece %.3f"
                  % (suite["name"], name, m["acc"], m["ece15"]), flush=True)
            del model
            torch.cuda.empty_cache()
    os.makedirs("runs/laya", exist_ok=True)
    with open("runs/laya/public_choice_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[choice] saved runs/laya/public_choice_eval.json", flush=True)


if __name__ == "__main__":
    main()
