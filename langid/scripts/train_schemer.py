"""DA-9 (E-57) Schemer trainer: token-level hashed tagger (DA-7 recipe reuse).

DA-9 of their ladder: pruned mmBERT + per-type heads + deterministic harness ->
ours: same HashedEmo token tagger as E-54, schemer slot labels, managed device
(GPU when free per user policy; hard single-process, VRAM-only usage).
Bar registered in EXPERIMENTS.md before training.
"""
import argparse, collections, json, os, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model
from langid.scripts.train_topic import Muon

BASE = pathlib.Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "langid" / "schemer"


def read_sents(p):
    out = []
    cur = []
    for line in open(p, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t") if line.strip() else []
        if not parts:
            if cur:
                out.append(tuple(cur))
                cur = []
            continue
        for i in range(0, len(parts), 2):
            cur.append((parts[i], parts[i + 1]))
    if cur:
        out.append(tuple(cur))
    return [([w for w, l in sent], [l for w, l in sent]) for sent in out]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", choices=["adam", "muon"], default="muon")
    ap.add_argument("--lr_muon", type=float, default=3e-2)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.load(open(DATA / "schemer_vocab.json", encoding="utf-8"))["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}
    K = len(labels); O_ID = lab2id["O"]
    train = read_sents(DATA / "train.tsv")
    val = read_sents(DATA / "val.tsv")

    def flatten(sents):
        texts, labs = [], []
        for toks, tags in sents:
            for i in range(len(toks)):
                prev = toks[i - 1] if i > 0 else "<s>"
                nxt = toks[i + 1] if i < len(toks) - 1 else "</s>"
                texts.append(prev + " " + toks[i] + " " + nxt)
                labs.append(lab2id[tags[i]])
        return texts, torch.tensor(labs, dtype=torch.long)

    tr_t, tr_l = flatten(train)
    va_t, va_l = flatten(val)
    print("train tokens", len(tr_t), "val tokens", len(va_t) , "device", device, flush=True)
    model = emo_model.HashedEmo(K).to(device)
    weights = None
    if args.weighted:
        cnt = collections.Counter(tr_l.tolist())
        w = torch.tensor([max(cnt[k], 1) ** -0.5 for k in range(K)])  # zero-count classes exist in schemer vocab
        weights = (w / w.mean()).to(device)
    opt = Muon(list(model.parameters()), args.lr_muon) if args.opt == "muon" else torch.optim.Adam(model.parameters(), lr=0.02)
    outp = str(BASE / "runs" / "langid_da9" / "schemer_tagger.pt")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    log, best, drops, t0 = [], -1.0, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_t))
        tot = nb = 0
        for i in range(0, len(tr_t), args.batch):
            idx = perm[i:i + args.batch].tolist()
            b_ids, b_off = emo_model.encode_batch([tr_t[j] for j in idx])
            logits = model(b_ids.to(device), b_off.to(device))
            loss = torch.nn.functional.cross_entropy(logits, tr_l[idx].to(device), weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 0.5)
            opt.step()
            tot += float(loss.detach()) * len(idx); nb += len(idx)
        model.eval()
        ec = et = c = 0
        with torch.no_grad():
            for i in range(0, len(va_t), 8192):
                v_ids, v_off = emo_model.encode_batch(va_t[i:i + 8192])
                pred = model(v_ids.to(device), v_off.to(device)).argmax(1).cpu()
                vb = va_l[i:i + 8192]
                c += int((pred == vb).sum()); m = vb != O_ID
                ec += int((pred[m] == vb[m]).sum()); et += int(m.sum())
        acc, eacc = c / len(va_t), ec / max(et, 1)
        log.append({"epoch": ep, "loss": round(tot / max(nb, 1), 4), "val_acc": round(acc, 4), "ent_acc": round(eacc, 4), "t": round(time.time() - t0, 1)})
        peak = torch.cuda.max_memory_allocated() // (1 << 20) if torch.cuda.is_available() else 0
        print("ep", ep, "loss", round(tot / max(nb, 1), 4), "ent_acc", round(eacc, 4), "peakMB", peak, flush=True)
        if eacc > best:
            best, drops = eacc, 0
            sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
            torch.save({"state": sd, "labels": labels, "ent_acc": best, "args": vars(args)}, outp)
        else:
            drops += 1
            if drops >= args.patience:
                print("EARLY_STOP", ep, flush=True); log[-1]["early_stop"] = True; break
    json.dump({"log": log, "best_ent_acc": best, "args": vars(args)}, open(outp + ".log.json", "w", encoding="utf-8"), indent=1)
    print("BEST", round(best, 4)); print("CKPT", outp)

if __name__ == "__main__":
    main()
