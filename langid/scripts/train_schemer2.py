"""DA-9 (E-58) Schemer v2 trainer: MLP head + char-class sentinels + FeatPreload.

E-58a smoke (runs/langid_da9/e58a_smoke_bench.json): hash-in-loop 96.2 ms/batch
vs GPU-only 10.4 ms/batch -> data-bound. This version precomputes features once
with an on-disk cache; per-batch work is pure GPU.
"""
import argparse, collections, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import features
from langid.scripts.train_topic import Muon
from langid.scripts.train_schemer import read_sents, DATA

BASE = pathlib.Path(__file__).resolve().parents[2]


def tok_cls(tok):
    if tok and "\u0660" <= tok[0] <= "\u0669":
        return "#AI"
    if any(ch.isdigit() for ch in tok):
        return "#EN"
    if ":" in tok:
        return "#COLON"
    return "#W"


def sent_texts(toks):
    out = []
    for i in range(len(toks)):
        prev = toks[i - 1] if i > 0 else "<s>"
        nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
        out.append(prev + tok_cls(prev) + " " + toks[i] + tok_cls(toks[i]) + " " + nxt + tok_cls(nxt))
    return out


def feat(text):
    words = features._WORD_RE.findall(features.normalize(text))
    out = []
    for i, w in enumerate(words):
        out.append(features.bucket("W:" + w))
        if i + 1 < len(words):
            out.append(features.bucket("B:" + w + "_" + words[i + 1]))
    for w in words[:40]:
        for g in features.word_ngrams(w):
            out.append(features.bucket("C:" + g))
    return out


class SchemerNet(torch.nn.Module):
    def __init__(self, K, nb=1 << 18, d=64):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(nb, 96, mode="sum")
        torch.nn.init.normal_(self.bag.weight, std=0.05)
        self.h1 = torch.nn.Linear(96, d)
        self.out = torch.nn.Linear(d, K)
        self.bias = torch.nn.Parameter(torch.zeros(K))

    def forward(self, flat, off, device):
        h = self.bag(flat, off)
        h = torch.relu(self.h1(h))
        return self.out(h) + self.bias


def precompute(texts, cache=None):
    import array
    if cache and pathlib.Path(cache).exists():
        payload = torch.load(cache, weights_only=False)
        print("FeatCache hit:", cache, flush=True)
        return payload["flat"].long(), payload["bounds"].long()
    flat = array.array("i")
    bounds = array.array("i", [0])
    t0 = time.perf_counter()
    for n, t in enumerate(texts):
        f = feat(t)
        flat.extend(f)
        bounds.append(len(flat))
        if n and n % 100000 == 0:
            print("FeatPreload", n, "/", len(texts), flush=True)
    ft = torch.frombuffer(memoryview(flat), dtype=torch.int32).clone()
    bt = torch.tensor(bounds, dtype=torch.long)
    if cache:
        torch.save({"flat": ft, "bounds": bt}, cache)
    print("FeatPreload", len(texts), "tokens,", len(flat), "feats in", round(time.perf_counter() - t0, 1), "s", flush=True)
    return ft, bt


def spans_for(flat, bounds, idxs):
    fs, off, tot = [], [], 0
    for j in idxs:
        s, e = int(bounds[j]), int(bounds[j + 1])
        fs.append(flat[s:e]); off.append(tot); tot += e - s
    ids = torch.cat(fs).long() if fs else torch.zeros(0, dtype=torch.long)
    return ids, torch.tensor(off, dtype=torch.long)


def flatten(sents, lab2id):
    texts, labs = [], []
    for toks, tags in sents:
        texts += sent_texts(toks)
        labs += [lab2id[x] for x in tags]
    return texts, torch.tensor(labs, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.load(open(DATA / "schemer_vocab.json", encoding="utf-8"))["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}
    K = len(labels); O_ID = lab2id["O"]
    train = read_sents(DATA / "train.tsv")
    val = read_sents(DATA / "val.tsv")
    print("train sents", len(train), "val sents", len(val), "device", device, flush=True)
    tr_t, tr_l = flatten(train, lab2id)
    va_t, va_l = flatten(val, lab2id)
    print("train tokens", len(tr_t), "val tokens", len(va_t), flush=True)
    t_pre = time.perf_counter()
    tr_flat, tr_bounds = precompute(tr_t, BASE / "runs" / "langid_da9" / "feat_cache_tr.pt")
    va_flat, va_bounds = precompute(va_t, BASE / "runs" / "langid_da9" / "feat_cache_va.pt")
    print("FeatPreload total", round(time.perf_counter() - t_pre, 1), "s (once)", flush=True)
    model = SchemerNet(K).to(device)
    cnt = collections.Counter(tr_l.tolist())
    w = torch.tensor([max(cnt[k], 1) ** -0.5 for k in range(K)])
    weights = (w / w.mean()).to(device) if args.weighted else None
    opt = Muon(list(model.parameters()), 3e-2)
    outp = str(BASE / "runs" / "langid_da9" / "schemer_v2.pt")
    pathlib.Path(outp).parent.mkdir(parents=True, exist_ok=True)
    log, best, drops = [], -1.0, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_t))
        tot = nb = 0
        for i in range(0, len(tr_t), args.batch):
            idx = perm[i:i + args.batch].tolist()
            ids, off = spans_for(tr_flat, tr_bounds, idx)
            logits = model(ids.to(device), off.to(device), device)
            loss = torch.nn.functional.cross_entropy(logits, tr_l[idx].to(device), weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 0.5); opt.step()
            tot += float(loss.detach()) * len(idx); nb += len(idx)
        model.eval()
        ec = et = c = 0
        with torch.no_grad():
            for i in range(0, len(va_t), 8192):
                v_ids, v_off = spans_for(va_flat, va_bounds, list(range(i, min(i + 8192, len(va_t)))))
                pred = model(v_ids.to(device), v_off.to(device), device).argmax(1).cpu()
                vb = va_l[i:i + 8192]
                c += int((pred == vb).sum()); msk = vb != O_ID
                ec += int((pred[msk] == vb[msk]).sum()); et += int(msk.sum())
        acc, eacc = c / len(va_t), ec / max(et, 1)
        log.append({"epoch": ep, "loss": round(tot / max(nb, 1), 4), "ent_acc": round(eacc, 4)})
        peak = torch.cuda.max_memory_allocated() // (1 << 20) if torch.cuda.is_available() else 0
        print("ep", ep, "loss", round(tot / max(nb, 1), 4), "ent_acc", round(eacc, 4), "peakMB", peak, flush=True)
        json.dump({"log": log, "best_ent_acc": best}, open(outp + ".log.json", "w", encoding="utf-8"), indent=1)
        if eacc > best:
            best, drops = eacc, 0
            sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
            torch.save({"state": sd, "labels": labels, "arch": "schemer2", "ent_acc": best}, outp)
        else:
            drops += 1
            if drops >= args.patience:
                print("EARLY_STOP", ep); log[-1]["early_stop"] = True; break
    json.dump({"log": log, "best_ent_acc": best}, open(outp + ".log.json", "w", encoding="utf-8"), indent=1)
    print("BEST", best); print("CKPT", outp)

if __name__ == "__main__":
    main()
