"""DA-9 (E-57) Schemer test eval: strict span F1 + rules-only baseline."""
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model
from langid.scripts.train_schemer import read_sents, DATA

BASE = pathlib.Path(__file__).resolve().parents[2]
AI = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
DG = "\u064a\u0646\u0627\u064a\u0631 \u0641\u0628\u0631\u0627\u064a\u0631 \u0645\u0627\u0631\u0633 \u0623\u0628\u0631\u064a\u0644 \u0645\u0627\u064a\u0648 \u064a\u0648\u0646\u064a\u0648 \u064a\u0648\u0644\u064a\u0648 \u0623\u063a\u0633\u0637\u0633 \u0633\u0628\u062a\u0645\u0628\u0631 \u0623\u0643\u062a\u0648\u0628\u0631 \u0646\u0648\u0641\u0645\u0628\u0631 \u062f\u064a\u0633\u0645\u0628\u0631".split()
DH = "\u0645\u062d\u0631\u0645 \u0635\u0641\u0631 \u0631\u0645\u0636\u0627\u0646 \u0634\u0648\u0627\u0644 \u0631\u062c\u0628 \u0634\u0639\u0628\u0627\u0646".split()
undef = set(DG)

def spans(tags, typ):
    out, cur = [], None
    for i, t in enumerate(tags):
        p, ty = (t.split("-", 1) + [""])[:2] if t != "O" else ("O", "")
        if typ and ty != typ:
            p = "O"
        if p == "B":
            if cur: out.append(cur)
            cur = [i, i]
        elif p == "I" and cur is not None:
            cur[1] = i
        else:
            if cur: out.append(cur); cur = None
    if cur: out.append(cur)
    return set(map(tuple, out))

def rules(toks):
    """Deterministic harness baseline: digits+year, Greg/Hijri month keywords,
    colon-time, Arabic-Indic numeral, relative catchphrases."""
    out = {"DATE_G": set(), "DATE_H": set(), "TIME": set(), "NUM_AI": set(), "DATE_REL": set()}
    for i, w in enumerate(toks):
        if any(ch in AI for ch in w) and ":" in w:
            out["TIME"].add((i, i))
        elif any(ch in AI for ch in w) and len(w) >= 2:
            n = int(w.translate({ord(c): None for c in AI}))
            out["NUM_AI" if (n < 100 or len(w) <= 3) else "NUM_AI"].add((i, i))
        elif w in DH or w in {"\u0647\u0640", "\u0627\u0644\u0647\u062c\u0631\u064a\u0629"}:
            out["DATE_H"].add((i, i))
        elif w in DG:
            out["DATE_G"].add((i, i))
        elif w in {"\u0623\u0645\u0633", "\u064a\u0648\u0645\u064a\u0646", "\u0623\u0633\u0628\u0648\u0639", "\u0623\u0633\u0628\u0648\u0639\u064a\u0646", "\u0634\u0647\u0631\u064a\u0646", "\u0645\u0646\u0630", "\u063a\u062f\u064b\u0627", "\u0627\u0644\u064a\u0648\u0645", "\u062d\u0627\u0644\u064a\u064b\u0627", "\u0627\u0644\u0645\u0627\u0636\u064a", "\u0627\u0644\u0645\u0642\u0628\u0644", "\u0627\u0644\u0645\u0633\u0627\u0621\u064b", "\u0635\u0628\u0627\u062d\u064b\u0627", "\u0627\u0644\u0639\u0627\u0634\u0631\u0629", "\u0627\u0644\u062b\u0644\u0627\u062b\u0629", "\u0627\u0644\u0631\u0627\u0628\u0639\u0629", "\u0627\u0644\u062e\u0627\u0645\u0633\u0629", "\u0627\u0644\u0633\u0627\u062f\u0633\u0629", "\u0627\u0644\u0633\u0627\u0628\u0639\u0629", "\u0627\u0644\u062b\u0627\u0645\u0646\u0629", "\u0627\u0644\u062a\u0627\u0633\u0639\u0629", "\u0627\u0644\u0623\u0648\u0644\u0649"}:
            out["DATE_REL" if w in {"\u0623\u0645\u0633", "\u064a\u0648\u0645\u064a\u0646", "\u0623\u0633\u0628\u0648\u0639", "\u0623\u0633\u0628\u0648\u0639\u064a\u0646", "\u0634\u0647\u0631\u064a\u0646", "\u0645\u0646\u0630", "\u0627\u0644\u0627\u0644\u0645\u0627\u0636\u064a", "\u0627\u0644\u0645\u0642\u0628\u0644"} else "DATE_REL"].add((i, i))
        if ":" in w and any(ch in AI for ch in w):
            out["TIME"].add((i, i))
    return out

def main():
    lang = "schemer"
    ckp = str(BASE / "runs" / "langid_da9" / "schemer_tagger.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    labels = ck["labels"]; lab2id = {l: i for i, l in enumerate(labels)}
    from langid.src import emo_model as em
    model = em.HashedEmo(len(labels))
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test = read_sents(DATA / "test.tsv")
    TYPES = ["DATE_G", "DATE_H", "DATE_REL", "TIME", "NUM_AI"]
    mtp = {t: [0, 0, 0] for t in TYPES}  # matched,pred,gold per type
    rtp = {t: [0, 0, 0] for t in TYPES}
    with torch.no_grad():
        for toks, tags in test:
            texts = []
            for i in range(len(toks)):
                prev = toks[i - 1] if i > 0 else "<s>"
                nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
                texts.append(prev + " " + toks[i] + " " + nxt)
            ids, off = em.encode_batch(texts)
            pred_ids = model(ids.to(device), off.to(device)).argmax(1).cpu().tolist()
            pred_tags = [labels[j] for j in pred_ids]
            # deterministic-constrained decoding: force B- at type transitions
            prev = "O"
            for k in range(len(pred_tags)):
                if pred_tags[k].startswith("I-") and prev != "B-" + pred_tags[k][2:] and prev != "I-" + pred_tags[k][2:]:
                    pred_tags[k] = "B-" + pred_tags[k][2:]
                prev = pred_tags[k]
            for t in TYPES:
                pg, gg = spans(pred_tags, t), spans(tags, t)
                mtp[t][0] += len(pg & gg); mtp[t][1] += len(pg); mtp[t][2] += len(gg)
                rg = rules(toks)
                mtp_rules = spans([("B-" + t) if (i, i) in rg[t] else "O" for i in range(len(toks))], t)
                rtp[t][0] += len(mtp_rules & gg); rtp[t][1] += len(mtp_rules); rtp[t][2] += len(gg)
    def f1(tp):
        p = tp[0] / max(tp[1], 1); r = tp[0] / max(tp[2], 1)
        return round(2 * p * r / max(p + r, 1e-9), 4)
    res = {"model_span_f1": {t: f1(mtp[t]) for t in TYPES},
           "model_micro_f1": f1([sum(x) for x in zip(*mtp.values())]),
           "rules_span_f1": {t: f1(rtp[t]) for t in TYPES},
           "rules_micro_f1": f1([sum(x) for x in zip(*rtp.values())])}
    json.dump(res, open(ckp + ".eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(k, "=", v)

if __name__ == "__main__":
    main()
