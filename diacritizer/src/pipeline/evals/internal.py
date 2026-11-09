"""Single internal-eval entry point (plan S2): the ONLY in-loop eval.

Returns loss / acc / per-class acc vector + logging dict, fp16 autocast
(Turing), batches of 128 windows. Used by train.py every eval_every step.
"""
import numpy as np
import torch


def evaluate_unseen(model, val_ids, val_y, device, batch=128):
    model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n_tok = 0
    per_hit = [0] * 15
    per_n = [0] * 15
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16,
                                              enabled=(device == "cuda")):
        for i in range(0, len(val_ids), batch):
            x = torch.from_numpy(val_ids[i:i + batch]).to(device)
            y = torch.from_numpy(val_y[i:i + batch]).to(device)
            out = model(x, y)
            m = y >= 0
            n = int(m.sum())
            tot_loss += float(out["loss"]) * n
            tot_acc += float(out["acc"]) * n
            n_tok += n
            g = y[m]
            pred = out["logits"][m].argmax(-1)
            gn = g.cpu().numpy().astype(np.int64)
            hn = (pred == g).cpu().numpy().astype(np.int64)
            tot = np.bincount(gn, minlength=15)
            for c in range(15):
                cnt = int(tot[c])
                if cnt:
                    per_n[c] += cnt
                    per_hit[c] += int(hn[gn == c].sum())
    res = {"val_loss": tot_loss / max(n_tok, 1), "val_acc": tot_acc / max(n_tok, 1),
           "per_class_acc": [round(per_hit[c] / max(1, per_n[c]), 4) if per_n[c]
                             else None for c in range(15)]}
    return res["val_loss"], res["val_acc"], res
