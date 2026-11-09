import sys, os, json
import torch
sys.path.insert(0, os.path.abspath("."))
from langid.src import redact_rules
from langid.src import emo_model
from langid.scripts.train_ner import read_sents

test = read_sents(os.path.abspath("data/langid/ner/test.tsv"))
ck = torch.load("runs/langid_da7/ner_tagger_muon_w.pt", map_location="cpu", weights_only=False)
labels = ck["labels"]
K = len(labels)
model = emo_model.HashedEmo(K)
model.emb.weight.data = ck["emb"].float()
model.bias.data = ck["bias"].float()
model.eval()
toks_all, gold = [], []
for toks, tags in test:
    toks_all.extend(toks)
    gold.extend(tags)
texts = []
flat = []
for si, (toks, tags) in enumerate(test):
    for i in range(len(toks)):
        prev = toks[i - 1] if i > 0 else "<s>"
        nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
        texts.append(prev + " " + toks[i] + " " + nxt)
preds = []
with torch.no_grad():
    for i in range(0, len(texts), 8192):
        ids, off = emo_model.encode_batch(texts[i:i + 8192])
        preds.append(model(ids, off).argmax(dim=1))
pred_ids = torch.cat(preds).tolist()
id2l = {i: l for i, l in enumerate(labels)}
predl = [id2l[i] for i in pred_ids]

NUMGOLD = {"B-ANG", "I-ANG", "B-DUC", "I-DUC"}
DATGOLD = {"B-TIMEX", "I-TIMEX"}
res = {}
for sub, name in [(NUMGOLD, "numeric"), (DATGOLD, "temporal")]:
    tp = fp = fn = mtp = mfp = 0
    for w, g, pl in zip(toks_all, gold, predl):
        isg = g in sub
        r = redact_rules.match(w)
        is_hit = r in ("NUM", "DATE", "PHONE", "ID")
        is_m = pl.split("-", 1)[-1] in ("ANG", "DUC") if name == "numeric" else pl.split("-", 1)[-1] == "TIMEX"
        if isg and is_hit: tp += 1
        elif is_hit and not isg: fp += 1
        elif isg and not is_hit: fn += 1
        if isg and is_m: mtp += 1
        elif is_m and not isg: mfp += 1
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    mp = mtp / max(mtp + mfp, 1)
    res[name] = {"regex": {"p": round(p, 4), "r": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4), "tp": tp, "fp": fp, "fn": fn},
                 "model_p": round(mp, 4), "model_tp": mtp, "model_fp": mfp}
print(json.dumps(res, indent=1))
json.dump(res, open("runs/langid_da7/redact_regex_eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
