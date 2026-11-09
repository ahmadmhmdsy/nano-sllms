"""E-67: eval-side selection-bias calibration for the L3 line (E-66 base).

Pre-registered (research/EXPERIMENTS.md, E-67 block). Mechanisms, fitted on
typed TRAIN only (the test split is never used for fitting):
  (a) permutation-averaged scoring: each item scored under k views
      (identity + k-1 sampled option-block permutations, exact block swap);
      log-probs mapped back to original option indices and averaged.
  (b) per-position prior correction: mean train log-prob per option POSITION,
      subtracted at each view's local position.
Gates: G3 agreement >= 0.90 PRIMARY; G2 typed acc >= 0.5630 non-regression
(identity anchor must reproduce E-66 within noise); the full panel is
reported for every decision rule regardless of outcome.

Usage: python laya/scripts/eval_calibrated.py --config configs/laya_l3_e66.yaml [--k 5]
"""
import argparse, json, os, random, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_head import collate  # noqa: E402
import train_l1 as t1  # noqa: E402
import eval_l2 as el2  # noqa: E402


def block_perm_view(it, tok, perm):
    """Exact option-block swap (eval_l2.permuted_items technique, explicit perm)."""
    sep = tok.sep_token_id
    ids, mp, n_opt = it["input_ids"], it["marker_pos"], it["n_options"]
    spans = []
    for k in range(n_opt):
        start = mp[k]
        end = mp[k + 1] if k + 1 < n_opt else ids.index(sep, start + 1)
        spans.append((start, end))
    prefix, tail = ids[:spans[0][0]], ids[spans[-1][1]:]
    blocks = [ids[s:e] for (s, e) in spans]
    new_ids, new_mp, pos = [], [], len(prefix)
    for p in perm:
        new_mp.append(pos)
        new_ids += blocks[p]
        pos += len(blocks[p])
    new_target = [it["target"][perm[j]] for j in range(n_opt)]
    return {"input_ids": prefix + new_ids + tail, "marker_pos": new_mp,
            "n_options": n_opt, "qtype": it["qtype"], "target": new_target,
            "gold_idx": max(range(n_opt), key=lambda i: new_target[i]),
            "workflow": it["workflow"], "case_id": it["case_id"],
            "qname": it["qname"], "qtype_name": it["qtype_name"]}


def score_all(model, items, pad_id, device, micro=16):
    """Per item: list of per-option log-probs (float list)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), micro):
            chunk = items[i:i + micro]
            batch, logp, _ = t1.forward_batch(model, collate(chunk, pad_id), device)
            ofs = 0
            for b, cnt in enumerate(batch["n_options"]):
                out.append(logp[ofs:ofs + cnt].tolist())
                ofs += cnt
    model.train()
    return out


def probs_from_scores(score_lists):
    """score_lists: per item, list of (orig_idx, summed_logp). Returns prob vectors."""
    out = []
    for scores in score_lists:
        tot = {}
        for orig, lp in scores:
            tot[orig] = tot.get(orig, 0.0) + lp
        n_opt = max(tot) + 1
        vec = [tot.get(o, float("-inf")) for o in range(n_opt)]
        mx = max(vec)
        ex = [pow(2.718281828459045, v - mx) for v in vec]
        s = sum(ex)
        out.append([e / s for e in ex])
    return out


def metrics(items, probs_list):
    acc = soft = brier = 0.0
    for pv, it in zip(probs_list, items):
        gold = int(it["gold_idx"])
        acc += int(max(range(len(pv)), key=lambda i: pv[i]) == gold)
        soft += pv[gold]
        onehot = [0.0] * len(pv)
        onehot[gold] = 1.0
        brier += sum((a - b) ** 2 for a, b in zip(pv, onehot)) / len(pv)
    n = max(1, len(items))
    return {"acc": acc / n, "soft_acc": soft / n, "brier": brier / n, "n": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l3_e66.yaml")
    ap.add_argument("--k", type=int, default=5, help="views per item (incl. identity)")
    args = ap.parse_args()
    import yaml
    from transformers import AutoTokenizer
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    out_dir = cfg["out_dir"]
    from laya_head import LayaDecisionModel
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id
    model = LayaDecisionModel(cfg["encoder"])
    ck = torch.load(os.path.join(out_dir, "final", "model.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    model = model.to(device).eval()

    train_items = torch.load(cfg["typed_train"], weights_only=False)
    test_items = torch.load(cfg["typed_test"], weights_only=False)
    k = int(args.k)
    print("[e67] train %d, test %d, k=%d views" % (len(train_items), len(test_items), k),
          flush=True)

    # (b) per-position prior from TRAIN (identity view only)
    train_logps = score_all(model, train_items, pad_id, device)
    pos_sum, pos_n = {}, {}
    for lps in train_logps:
        for j, lp in enumerate(lps[:15]):
            pos_sum[j] = pos_sum.get(j, 0.0) + lp
            pos_n[j] = pos_n.get(j, 0) + 1
    prior = {j: pos_sum[j] / pos_n[j] for j in sorted(pos_sum)}
    print("[e67] position prior: " + json.dumps({str(j): round(v, 3) for j, v in prior.items()}),
          flush=True)

    # decision rules over multi-views
    def decisions(items):
        all_views, view_index = [], []   # per view: list of (item_idx, orig, local)
        rng = random.Random(1234)
        for i, it in enumerate(items):
            n_opt = it["n_options"]
            perms = [list(range(n_opt))]
            if it["qtype_name"] == "choice" and n_opt >= 2:
                for _ in range(k - 1):
                    p = list(range(n_opt))
                    rng.shuffle(p)
                    perms.append(p)
            for perm in perms:
                view = it if perm == list(range(n_opt)) else block_perm_view(it, tok, perm)
                view_index.append([(i, orig, local)
                                   for local, orig in enumerate(perm)])
                all_views.append(view)
        lps = score_all(model, all_views, pad_id, device)
        acc_avg = [[] for _ in items]
        acc_prio = [[] for _ in items]
        for entries, lp_list in zip(view_index, lps):
            for (i, orig, local), lp in zip(entries, lp_list):
                acc_avg[i].append((orig, lp))
                acc_prio[i].append((orig, lp - prior.get(local, 0.0)))
        rules = {}
        for name, acc in (("perm_avg", acc_avg), ("perm_avg_prior", acc_prio)):
            rules[name] = probs_from_scores(acc)
        return rules

    base = t1.evaluate(model, test_items, pad_id, device)
    print("[e67] identity anchor acc %.4f soft %.4f brier %.4f (E-66 G2 0.5630)"
          % (base["acc"], base["soft_acc"], base["brier"]), flush=True)

    rules = decisions(test_items)
    panel = {"identity": {"acc": base["acc"], "soft_acc": base["soft_acc"],
                          "brier": base["brier"], "n": base["n"]}}
    for name, pl in rules.items():
        m = metrics(test_items, pl)
        panel[name] = m
        print("[e67] %s acc %.4f soft %.4f brier %.4f" %
              (name, m["acc"], m["soft_acc"], m["brier"]), flush=True)

    # G3 agreement (el2 protocol verbatim: 200 items, seed 7)
    perm_items, origs = el2.permuted_items(test_items, tok, n=200, seed=7)
    agree_ident = el2.agreement(model, origs, perm_items, pad_id, device)
    r_orig = decisions(origs)
    r_perm = decisions(perm_items)
    agree_rules = {}
    for name in ("perm_avg", "perm_avg_prior"):
        po = [max(range(len(pv)), key=lambda i: pv[i]) for pv in r_orig[name]]
        pp = [max(range(len(pv)), key=lambda i: pv[i]) for pv in r_perm[name]]
        agree_rules[name] = sum(int(a == b) for a, b in zip(po, pp)) / max(1, len(po))
    print("[e67] G3 agreement: identity %.4f | perm_avg %.4f | perm_avg_prior %.4f"
          % (agree_ident, agree_rules["perm_avg"], agree_rules["perm_avg_prior"]),
          flush=True)

    best = max(agree_rules, key=lambda n: agree_rules[n])
    report = {
        "experiment": "E-67 eval-side calibration", "model": out_dir,
        "k_views": k, "position_prior": {str(j): round(v, 4) for j, v in prior.items()},
        "panel": panel,
        "g3_agreement": {"identity": agree_ident,
                         **{n: round(a, 4) for n, a in agree_rules.items()},
                         "protocol": "el2.permuted_items n=200 seed=7; same rule both sides"},
        "gates": {
            "G3_perm_agreement": {"value": agree_rules[best], "rule": best, "bar": 0.90,
                                  "pass": agree_rules[best] >= 0.90},
            "G2_typed_nonregression": {"value": panel[best]["acc"], "rule": best,
                                       "bar": 0.5630, "pass": panel[best]["acc"] >= 0.5630,
                                       "identity_anchor": base["acc"]},
        },
    }
    path = os.path.join(out_dir, "eval_calibrated.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "bench_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "e67_eval_calibrated", "k": k,
                            "panel": panel, "g3": report["g3_agreement"],
                            "gates": report["gates"]}) + "\n")
    print("[e67] G3 %s | G2 %s -> %s" %
          (report["gates"]["G3_perm_agreement"]["pass"],
           report["gates"]["G2_typed_nonregression"]["pass"], path), flush=True)


if __name__ == "__main__":
    main()
