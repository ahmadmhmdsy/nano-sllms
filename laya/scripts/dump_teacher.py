"""E-69 (lever P1): dump L2 teacher soft distributions over typed TRAIN items.

The typed items store raw texts (instructions / option_texts / state_text), so
the teacher is packed with ITS OWN tokenizer from L2's config. Teacher probs
are softmax(logits / T) with T=2 (pre-registered), aligned to the canonical
option order = the student's index order. Keys are workflow|case_id|qname.

GPU job - never run while a train run is active (single-GPU rule).

Usage: python laya/scripts/dump_teacher.py --l2_config configs/laya_l2.yaml --typed_train data/laya/typed_decisions/train.pt --out data/laya/kd/teacher_l2_train.pt
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_head import LayaDecisionModel, collate  # noqa: E402
from laya_head import pack_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2_config", default="configs/laya_l2.yaml")
    ap.add_argument("--typed_train", default="data/laya/typed_decisions/train.pt")
    ap.add_argument("--out", default="data/laya/kd/teacher_l2_train.pt")
    ap.add_argument("--T", type=float, default=2.0)
    args = ap.parse_args()
    import yaml
    from transformers import AutoTokenizer
    with open(args.l2_config, "r", encoding="utf-8") as f:
        l2cfg = yaml.safe_load(f)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(l2cfg["encoder"])
    pad_id = tok.pad_token_id
    model = LayaDecisionModel(l2cfg["encoder"], head_layers=int(l2cfg["head_layers"]),
                              dropout=float(l2cfg["dropout"])).to(device)
    model.load_state_dict(torch.load(os.path.join(l2cfg["out_dir"], "final", "model.pt"),
                                     map_location=device, weights_only=False))
    model.eval()
    print("[kd] teacher %s loaded (T=%.1f)" % (l2cfg["encoder"], args.T), flush=True)

    items = torch.load(args.typed_train, weights_only=False)
    packable, missing = [], 0
    for it in items:
        if all(k in it for k in ("instructions", "option_texts", "state_text")):
            packable.append(it)
        else:
            missing += 1
    print("[kd] packable %d / %d (missing raw texts: %d)" %
          (len(packable), len(items), missing), flush=True)

    views = []
    for it in packable:
        ids, markers = pack_sequence(
            tok, it["qtype"], it["instructions"], it["option_texts"],
            it["state_text"], max_len=int(l2cfg["max_len"]),
            head_max_len=int(l2cfg["head_max_len"]),
            opt_max=int(l2cfg["option_token_max"]))
        if len(markers) != it["n_options"]:
            continue
        views.append((it, {"input_ids": ids, "marker_pos": markers,
                           "n_options": it["n_options"], "qtype": it["qtype"],
                           "target": it["target"]}))
    print("[kd] teacher views: %d" % len(views), flush=True)

    kd, n_cov = {}, 0
    with torch.no_grad():
        for i in range(0, len(views), 16):
            chunk = views[i:i + 16]
            batch = collate([v for _, v in chunk], pad_id)
            batch = {k: (x.to(device) if torch.is_tensor(x) else x)
                     for k, x in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["marker_pos"], batch["marker_batch"],
                               batch["qtype"])
            logits = logits.float().cpu()
            ofs = 0
            for b, cnt in enumerate(batch["n_options"]):
                p = torch.softmax(logits[ofs:ofs + cnt] / args.T, dim=-1)
                it = chunk[b][0]
                kd["%s|%s|%s" % (it.get("workflow", "?"), it.get("case_id", "?"),
                                 it["qname"])] = [float(x) for x in p]
                n_cov += 1
                ofs += cnt
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(kd, args.out)
    print("[kd] saved %d teacher distributions (%d packed) -> %s" %
          (len(kd), n_cov, args.out), flush=True)
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"teacher": l2cfg["encoder"], "T": args.T,
                   "items": len(items), "packed": n_cov, "keys": len(kd)}, f, indent=1)


if __name__ == "__main__":
    main()
