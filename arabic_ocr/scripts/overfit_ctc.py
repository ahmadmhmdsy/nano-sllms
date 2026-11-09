"""Overfit test: can a given architecture learn 32 lines at all?

Train on a tiny subset with CTC only and watch the loss, the blank fraction,
the encoder position-std and the batch CER. An architecture that cannot drive
CER down on 32 memorisable lines cannot be fixed by more data or more epochs.
"""
import argparse, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, "."); sys.path.insert(0, "arabic_ocr")
from arabic_ocr.src.models import create_model
from arabic_ocr.src.data.dataset import ArabicOCRDataset, collate_fn


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_ctc(log_probs, idx_to_char):
    ids = log_probs.argmax(-1)
    out = []
    for n in range(ids.size(1)):
        prev, chars = -1, []
        for t in ids[:, n].tolist():
            if t != prev and t != 0:
                chars.append(idx_to_char.get(t, ""))
            prev = t
        out.append("".join(chars))
    return out


def log_probs_of(model, images, mtype):
    if mtype == "crnn":
        logits = model.decode(model.encode(images))
        return F.log_softmax(logits.float(), dim=-1)
    return model.ctc_log_probs(model.encode(images))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-text", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ArabicOCRDataset("data/linecorpus_smoke/train", image_height=64, image_width=1024,
                          vocab_path="data/linecorpus_vocab.json", max_len=192,
                          min_text_len=2, max_text_len=a.max_text, keep_aspect=True)
    idx_to_char = {v: k for k, v in ds.char_to_idx.items()}
    batches = list(DataLoader(Subset(ds, list(range(a.n))), batch_size=a.batch,
                              shuffle=False, collate_fn=collate_fn))
    batches = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
               for b in batches]

    cfg = dict(type=a.model, vocab_size=len(ds.char_to_idx), image_height=64,
               image_width=1024, patch_size=16, vit_layers=6, vit_hidden=512,
               vit_heads=8, vit_ffn=2048, decoder_layers=6, decoder_hidden=512,
               decoder_heads=8, max_seq_len=192, dropout=0.0, hidden_dim=256,
               num_layers=2, ctc_weight=0.3)
    m = create_model(cfg).to(dev)
    print("model=%s  params=%.2fM  samples=%d  vocab=%d  device=%s"
          % (a.model, m.count_params() / 1e6, a.n, len(ds.char_to_idx), dev))
    with torch.no_grad():
        lp = log_probs_of(m, batches[0]["images"], a.model)
    print("CTC frames T = %d   (targets here <= %d chars)" % (lp.size(0), a.max_text))
    print()

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = batches[(step - 1) % len(batches)]
        out = m(b["images"], text=b["texts"], text_lens=b["text_lens"])
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            was = m.training
            m.eval()
            with torch.no_grad():
                f = m.encode(b["images"])
                lp = log_probs_of(m, b["images"], a.model)
                blank = (lp.argmax(-1) == 0).float().mean().item()
                pstd = f.std(dim=1).mean().item()
                hyps = greedy_ctc(lp, idx_to_char)
            m.train(was)
            ce = sum(edit_distance(r, h) for r, h in zip(b["raw_texts"], hyps))
            tot = sum(len(r) for r in b["raw_texts"])
            print("step %4d  loss=%.4f  blank=%.4f  pos_std=%.5f  batchCER=%.3f  %.0fs"
                  % (step, loss.item(), blank, pstd, ce / max(tot, 1), time.time() - t0))

    m.eval()
    with torch.no_grad():
        lp = log_probs_of(m, batches[0]["images"], a.model)
        hyps = greedy_ctc(lp, idx_to_char)
    print()
    print("--- final hypotheses ---")
    for r, h in list(zip(batches[0]["raw_texts"], hyps))[:4]:
        print("REF:", r[:60])
        print("HYP:", h[:60])


if __name__ == "__main__":
    main()