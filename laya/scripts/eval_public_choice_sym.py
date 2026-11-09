r"""E-74 phase-2b: zero-shot AG News + emotion under SYMMETRIC packing.
Same suites/samples (seed 42, n 2000) as eval_public_choice; each option gets
its own isolated window; group decision = argmax over sigmoid scores via qid.
Published: AG News Laya 0.950 / Jev 0.910; emotion Laya 0.595 / Jev 0.480.
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, pack_sequence, collate, qtype_id
import eval_public_choice as epc

MODELS = [("l3_e72", "runs/laya/l3_e72/final/model.pt"),
          ("l3_e74_sym", "runs/laya/l3_e74/final/model.pt")]


def load_sym_model(path, device):
    import yaml
    enc = yaml.safe_load(open("configs/laya_l3_e74.yaml"))["encoder"]
    model = LayaDecisionModel(enc)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd if not isinstance(sd, dict) or "model" not in sd else sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print("[choice-sym] warn unexpected keys: %s" % str(unexpected)[:120], flush=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_probs_sym(model, items, n_opt, pad_id, device, micro=32):
    model.eval()
    out = []
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate(chunk, pad_id).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                      batch["marker_batch"], batch["qtype"]).float().view(-1)
        out.append(torch.sigmoid(s).cpu())
    v = torch.cat(out)
    return v.view(len(out) if False else -1, len(v) // max(1, len(np.zeros(0))) or 0)
    # placeholder replaced below


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import yaml
    enc = yaml.safe_load(open("configs/laya_l3_e74.yaml"))["encoder"]
    tok = AutoTokenizer.from_pretrained(enc)
    pad_id = tok.pad_token_id
    qt = qtype_id("choice")
    report = {"suites": {}, "caveat": "vendor prompts unpublished; approximate comparison",
              "packing": "symmetric-isolated"}
    for suite in epc.SUITES:
        ds = epc.load_any(suite["candidates"], suite["split"])
        rng = np.random.default_rng(42)
        idx = sorted(rng.permutation(len(ds))[:suite["n_eval"]].tolist())
        n_opt = len(suite["options"])
        items = []
        for row, i in enumerate(idx):
            r = ds[int(i)]
            for oi, opt in enumerate(suite["options"]):
                ids, markers = pack_sequence(tok, qt, epc.INSTR, [opt], str(r[suite["text_col"]]))
                if len(markers) != 1:
                    continue
                items.append({"input_ids": ids, "marker_pos": markers, "n_options": 1,
                              "qtype": qt, "row": row, "opt_i": oi,
                              "target": [1.0] if int(r[suite["label_col"]]) == oi else [0.0]})
        model = None
        preds = {}
        for name, path in MODELS:
            if not os.path.exists(path):
                print("[%s] skip %s (missing %s)" % (suite["name"], name, path), flush=True)
                continue
            model = load_sym_model(path, device)
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(items), micro := 64):
                    chunk = items[i:i + micro]
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate(chunk, pad_id).items()}
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        s = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                  batch["marker_batch"], batch["qtype"]).float().view(-1)
                    out.append(torch.sigmoid(s).cpu())
            pv = torch.cat(out).numpy()
            rows = np.zeros((len(idx), n_opt))
            cnt = np.zeros(len(idx))
            for k, it in enumerate(items):
                rows[it["row"], it["opt_i"]] += pv[k]
                cnt[it["row"]] += 1
            rows = np.divide(rows, cnt[:, None], out=np.zeros_like(rows), where=cnt[:, None] > 0)
            pred = rows.argmax(1)
            conf = rows.max(1)
            gold = []
            for i in idx:
                gold.append(int(ds[int(i)][suite["label_col"]]))
            gold = np.array(gold)
            m = {"acc": float((pred == gold).mean()),
                 "ece15": epc.ece15(conf, (pred == gold).astype(float)),
                 "n": len(gold)}
            report["suites"].setdefault(suite["name"], {})[name] = m
            print("[%s] %-11s acc %.3f ece %.3f" % (suite["name"], name, m["acc"], m["ece15"]), flush=True)
            del model
            torch.cuda.empty_cache()
    with open("runs/laya/public_choice_eval_sym.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[choice-sym] saved runs/laya/public_choice_eval_sym.json", flush=True)


if __name__ == "__main__":
    main()
