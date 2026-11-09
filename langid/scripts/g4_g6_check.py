"""G4 int8-vs-fp32 accuracy delta + G6 leakage barrier, one quick pass."""
import pathlib, random, re, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.features import text_features
from langid.src.data import LANGS, LANG_TO_ID
from langid.src.infer import Int8LangID

model = torch.load("runs/langid_da1_e3/model_fp32.pt", map_location="cpu", weights_only=False)
emb = model.get("emb")
if emb is None:
    emb = model.get("embedding")
E = emb.detach().numpy() if hasattr(emb, "detach") else emb[0].detach().numpy()

def features_ids_offsets(pages):
    ids, offs = [], [0]
    for p in pages:
        ids.extend(p)
        offs.append(len(ids))
    return ids, offs[:-1]

rng = random.Random(0)
rows = []
with open("data/langid/eval.tsv", encoding="utf-8") as f:
    for line in f:
        lang, text = line.rstrip("\n").split("\t", 1)
        rows.append((lang, text))
rng.shuffle(rows)
sample = rows[:2000]
correct_i8 = correct_f32 = 0
bias = model.get("bias") if "bias" in model else None
# fp32 scores using torch module-free path
import torch.nn as nn

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(E.shape[0], E.shape[1], mode="sum")
        self.emb.weight.data.copy_(torch.from_numpy(E))
        self.bias = nn.Parameter(torch.zeros(len(LANGS)))
head = Head()
b = model.get("bias"); head.bias.data.copy_(b.detach().cpu() if hasattr(b, "detach") else torch.from_numpy(b))
m8 = Int8LangID("runs/langid_da1_e3/langid_int8.npz")
with torch.no_grad():
    for lang, text in sample:
        feats = text_features(text)
        if not feats:
            continue
        ids, offs = features_ids_offsets([feats])
        gi = LANG_TO_ID[lang]
        s32 = head.emb(torch.tensor(ids), torch.tensor([offs[0]] if isinstance(offs, int) else offs) ) + head.bias
        top2 = torch.topk(s32, 2, dim=1).values[0]
        margin = (top2[0] - top2[1]).item()
        # int8
        s8, _ = m8.scores(text)
        p8 = s8.argmax()
        pred8 = m8.langs[p8] if s8 is not None else None
        pred32 = LANGS[int(s32[0].argmax())]
        if pred8 == lang: correct_i8 += 1
        if pred32 == lang: correct_f32 += 1
n = len(sample)
print("sample n:", n, "fp32:", correct_f32 / n, "int8:", correct_i8 / n)

# G6 leakage: 10-word shingles train vs eval
def shingles(path, k=10, cap=200000):
    seen = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= cap: break
            toks = re.findall(r"[^\W\d_]+", line.lower())
            for j in range(0, max(0, len(toks)-k+1)):
                seen.setdefault(" ".join(toks[j:j+k]), set()).add(i)
    return seen
tr = shingles("data/langid/train.tsv")
ev = shingles("data/langid/eval.tsv")
inter = set(tr) & set(ev)
print("G6 shingle sets:", len(tr), len(ev), "overlap:", len(inter))
