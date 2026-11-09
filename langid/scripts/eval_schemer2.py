"""E-58 strict span-F1 eval + rules-only baseline for schemer_v2."""
import json, pathlib, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from langid.scripts.train_schemer import read_sents, DATA
from langid.scripts.train_schemer2 import SchemerNet, sent_texts, feat
from langid.scripts.eval_schemer import spans, rules

ck = torch.load(str(ROOT / "runs" / "langid_da9" / "schemer_v2.pt"), map_location="cpu", weights_only=False)
labels = ck["labels"]; K = len(labels)
m = SchemerNet(K)
m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
m.eval()
test = read_sents(DATA / "test.tsv")
types = ["DATE_G", "DATE_H", "DATE_REL", "TIME", "NUM_AI"]


def counts(gold, sys_, t):
    G, P = spans(gold, t), spans(sys_, t)
    return len(G & P), len(P), len(G)


per, per_r = {t: [0, 0, 0] for t in types}, {t: [0, 0, 0] for t in types}
mm, pp, gg = 0, 0, 0
rm, rp, rg = 0, 0, 0
with torch.no_grad():
    for toks, tags in test:
        texts = sent_texts(toks)
        flat, off = [], []
        for t in texts:
            f = feat(t)
            off.append(len(flat)); flat += f
        ids = torch.tensor(flat, dtype=torch.long); offs = torch.tensor(off, dtype=torch.long)
        pred = [labels[j] for j in m(ids, offs, torch.device("cpu")).argmax(1).tolist()]
        pv = "O"
        for k in range(len(pred)):
            if pred[k].startswith("I-") and pv not in ("B-" + pred[k][2:], "I-" + pred[k][2:]):
                pred[k] = "B-" + pred[k][2:]
            pv = pred[k]
        rr = rules(toks)
        for t in types:
            a, b, c = counts(tags, pred, t)
            per[t][0] += a; per[t][1] += b; per[t][2] += c
            G, P = spans(tags, t), rr.get(t, set())
            d, e, f = len(G & P), len(P), len(G)
            per_r[t][0] += d; per_r[t][1] += e; per_r[t][2] += f
            mm += a; pp += b; gg += c; rm += d; rp += e; rg += f


def f1(v):
    a, b, c = v
    return 2 * a / max(b + c, 1)


out = {
    "model_span_f1": {t: round(f1(per[t]), 4) for t in types},
    "rules_span_f1": {t: round(f1(per_r[t]), 4) for t in types},
    "model_micro_f1": round(2 * mm / max(pp + gg, 1), 4),
    "rules_micro_f1": round(2 * rm / max(rp + rg, 1), 4),
}
json.dump(out, open(ROOT / "runs" / "langid_da9" / "schemer_v2.pt.eval.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
