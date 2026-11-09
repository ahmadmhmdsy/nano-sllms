import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "arabic_ocr")
from arabic_ocr.src.models import create_model
from arabic_ocr.src.data.dataset import ArabicOCRDataset, collate_fn
from torch.utils.data import DataLoader, Subset

ds = ArabicOCRDataset("data/linecorpus_smoke/train", image_height=64, image_width=1024,
                      vocab_path="data/linecorpus_vocab.json", max_len=192,
                      min_text_len=2, max_text_len=48, keep_aspect=True)
idx_to_char = {v: k for k, v in ds.char_to_idx.items()}
dev = "cuda"
sub = Subset(ds, list(range(16)))
batches = list(DataLoader(sub, batch_size=8, shuffle=False, collate_fn=collate_fn))
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batches[0].items()}

print("texts[0,:12] :", b["texts"][0, :12].tolist())
print("text_lens    :", b["text_lens"].tolist())
L0 = int(b["text_lens"][0])
print("decoded[0]   :", "".join(idx_to_char.get(int(t), "?") for t in b["texts"][0, :L0]))
print("raw[0]       :", b["raw_texts"][0])
print("tgt used     :", "".join(idx_to_char.get(int(t), "?") for t in b["texts"][0, 1:1 + L0 - 2]))
print()

cfg = dict(type="vit_hybrid", patch_size=16, image_height=64, image_width=1024,
           vit_layers=6, vit_hidden=512, vit_heads=8, vit_ffn=2048,
           decoder_layers=6, decoder_hidden=512, decoder_heads=8,
           vocab_size=len(ds.char_to_idx), max_seq_len=192, dropout=0.0, ctc_weight=0.3)
m = create_model(cfg).to(dev)


def show(tag):
    m.eval()
    with torch.no_grad():
        f = m.encode(b["images"])
        print("%s feats[0,0,:5]   = %s" % (tag, [round(v, 5) for v in f[0, 0, :5].tolist()]))
        print("%s feats[0,1,:5]   = %s" % (tag, [round(v, 5) for v in f[0, 1, :5].tolist()]))
        print("%s feats[0,100,:5] = %s" % (tag, [round(v, 5) for v in f[0, 100, :5].tolist()]))
        print("%s feats[1,1,:5]   = %s" % (tag, [round(v, 5) for v in f[1, 1, :5].tolist()]))
        print("%s pos_std=%.7f  batch_std=%.7f" % (tag, f[:, 1:, :].std(dim=1).mean().item(),
                                                    f[:, 1:, :].std(dim=0).mean().item()))
        lp = m.ctc_log_probs(f)
        print("%s log_probs shape=%s" % (tag, tuple(lp.shape)))
        for t in (0, 128):
            top = lp[t, 0].topk(5)
            print("%s  frame %3d top5: %s" % (tag, t, [(idx_to_char.get(int(i), "?"), round(float(p), 3))
                                                       for p, i in zip(top.values, top.indices)]))
        print("%s  frame0 vs frame128 identical: %s" % (tag, bool(torch.allclose(lp[0, 0], lp[128, 0]))))
    m.train()


show("INIT ")
print()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(1, 61):
    out = m(b["images"], text=b["texts"], text_lens=b["text_lens"])
    loss = out["ctc_loss"]
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
print("after 60 CTC-only steps, loss=%.4f" % loss.item())
show("TRAIN")