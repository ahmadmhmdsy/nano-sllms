"""Stage 0.5e (TASKS 139) - train the character LM used to rescore CTC N-best.

Corpus is TRAIN-ONLY text (never an eval split): the in-domain line-training
labels by default; the mixed text pool data/ocr_textpool/texts.jsonl is opt-in
via --pool and must be preceded by an overlap audit against the eval splits.

PRE-REGISTERED (TASKS 139): the LM is tuned on data/hf/muharaf/validation and
frozen before any test set is decoded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
sys.path.insert(0, str(MODEL_DIR))

from submission_code.torch_models import load_ctc_backbone   # noqa: E402

DEFAULT_CKPT = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"
DEFAULT_OUT = ROOT / "arabic_ocr" / "runs" / "stage05e"
DEFAULT_EXTRA = ["data/hf/muharaf/train", "data/hf/khatt_teklia/train", "data/hflines/train"]


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_sibling("char_lm_common", Path(__file__).with_name("char_lm_common.py"))


def build_corpus(args, ch2id):
    """-> (list of encoded sequences, audit dict)."""
    texts, audit = [], {"extra_chars": 0, "pool_chars": 0, "dropped_chars": 0}
    for repeat in range(max(1, args.extra_repeat)):
        for spec in args.extra:
            folder = Path(spec) if Path(spec).is_absolute() else ROOT / spec
            labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
            for rec in labels:
                raw = rec.get("text", "") or ""
                text = " ".join(C.strip_diacritics(raw).split())
                clean = "".join(ch for ch in text if ch in ch2id)
                audit["dropped_chars"] += len(text) - len(clean)
                if len(clean) >= 2:
                    texts.append(clean)
        audit["extra_chars"] = sum(len(t) for t in texts)
    if args.pool:
        with open(ROOT / "data" / "ocr_textpool" / "texts.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                raw = rec.get("plain") or rec.get("text") or ""
                text = " ".join(C.strip_diacritics(raw).split())
                clean = "".join(ch for ch in text if ch in ch2id)
                audit["dropped_chars"] += len(text) - len(clean)
                if len(clean) < 2:
                    continue
                if audit["pool_chars"] + len(clean) > args.pool_chars:
                    break
                audit["pool_chars"] += len(clean)
                texts.append(clean)
    seqs = []
    for text in texts:
        for i in range(0, len(text), args.seq):
            chunk = text[i:i + args.seq]
            if len(chunk) >= args.min_seq:
                seqs.append([ch2id[ch] for ch in chunk])
    audit["texts"] = len(texts)
    audit["seqs"] = len(seqs)
    audit["chars"] = sum(len(s) for s in seqs)
    return seqs, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--extra", nargs="*", default=DEFAULT_EXTRA)
    ap.add_argument("--extra-repeat", type=int, default=3)
    ap.add_argument("--pool", action="store_true", help="add data/ocr_textpool/texts.jsonl")
    ap.add_argument("--pool-chars", type=int, default=4000000)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--min-seq", type=int, default=8)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _, charset, _ = load_ctc_backbone(args.ckpt, torch.device("cpu"))
    id2ch = C.id_char_table(charset)
    ch2id = {}
    for i in sorted(id2ch):
        ch2id.setdefault(id2ch[i], i)
    print("charset vocab=%d usable=%d blank=%d" % (charset.vocab_size, len(ch2id), charset.blank_id))

    random.seed(args.seed)
    seqs, audit = build_corpus(args, ch2id)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    random.shuffle(seqs)
    n_val = max(1, int(len(seqs) * args.val_frac))
    val_seqs, train_seqs = seqs[:n_val], seqs[n_val:]
    print("corpus %s" % json.dumps(audit, ensure_ascii=False))

    model = C.CharLM(charset.vocab_size, args.dim, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_seqs)
        t0, run, seen = time.time(), 0.0, 0
        for start in range(0, len(train_seqs), args.batch):
            batch = train_seqs[start:start + args.batch]
            if len(batch) < 2:
                continue
            inp, tgt = C.make_batch(batch, model.bos)
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = lossf(logits.reshape(-1, charset.vocab_size), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach())
            seen += 1
            if seen % 200 == 0:
                print("ep%d step%d loss=%.4f %.1f seq/s" % (epoch, seen, run / seen,
                      seen / max(1e-6, time.time() - t0)), flush=True)
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_seqs), args.batch):
                batch = val_seqs[start:start + args.batch]
                inp, tgt = C.make_batch(batch, model.bos)
                logits = model(inp.to(device))
                vl += float(lossf(logits.reshape(-1, charset.vocab_size), tgt.reshape(-1).to(device)))
                vn += 1
        rec = {"epoch": epoch, "train_loss": round(run / max(1, seen), 4),
               "val_loss": round(vl / max(1, vn), 4),
               "val_ppl": round(float(torch.exp(torch.tensor(vl / max(1, vn)))), 3),
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print("EPOCH %s" % json.dumps(rec), flush=True)
        torch.save({"state_dict": model.state_dict(), "vocab": charset.vocab_size,
                    "dim": args.dim, "layers": args.layers, "id2ch": id2ch,
                    "charset": getattr(charset, "characters", None),
                    "history": history, "audit": audit}, args.out / "char_lm.pt")
        (args.out / "char_lm_meta.json").write_text(json.dumps(
            {"corpus": audit, "args": vars(args) | {"ckpt": str(args.ckpt), "out": str(args.out)},
             "history": history}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("DONE best_val_loss=%.4f" % min(h["val_loss"] for h in history))


if __name__ == "__main__":
    main()
