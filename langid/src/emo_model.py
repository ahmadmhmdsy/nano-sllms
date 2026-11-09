"""DA-2 emo: hashed bag-of-n-grams emoji suggestion (Emo-analogue).

Two heads:
- HashedEmo: EmbeddingBag(65536 -> K, sum) + bias matrix factorization on
  hashed word/bigram/char features (E-44 arm; also the (c) arm carrier).
- TinyTransformerEmo: option (b), a 1-layer d=32 self-attention encoder
  over the concatenation of hashed word buckets + char-n-gram buckets
  (sequence truncated), mean-pooled to the K-way head.

Features (emo_features): word unigrams (W: prefix), adjacent-word bigrams
(B:), char 1..3 n-grams (C:) - all FNV-1a into 2^16 buckets with distinct
prefixes so no cross-family collision.
"""
import torch

from langid.src import features

MAX_LEN = 48


def emo_features(text: str):
    words = features._WORD_RE.findall(features.normalize(text))
    for i, w in enumerate(words):
        yield features.bucket("W:" + w)
        if i + 1 < len(words):
            yield features.bucket("B:" + w + "_" + words[i + 1])
    for w in words[:40]:
        for g in features.word_ngrams(w):
            yield features.bucket("C:" + g)


class HashedEmo(torch.nn.Module):
    def __init__(self, num_emo: int, num_buckets: int = 1 << 16):
        super().__init__()
        self.num_emo = num_emo
        self.num_buckets = num_buckets
        self.emb = torch.nn.EmbeddingBag(num_buckets, num_emo, mode="sum")
        self.bias = torch.nn.Parameter(torch.zeros(num_emo))
        torch.nn.init.zeros_(self.emb.weight)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.emb(ids, offsets) + self.bias


class TinyTransformerEmo(torch.nn.Module):
    """Option (b): tiny self-attention head over the token-bucket sequence."""

    def __init__(self, num_emo: int, num_buckets: int = 1 << 16, d: int = 32):
        super().__init__()
        self.num_emo = num_emo
        self.emb = torch.nn.Embedding(num_buckets, d)
        self.pos = torch.nn.Embedding(MAX_LEN, d)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=64, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.head = torch.nn.Linear(d, num_emo)
        torch.nn.init.zeros_(self.emb.weight)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        n = len(offsets)
        seqs = []
        for i in range(n):
            a = int(offsets[i])
            b = int(offsets[i + 1]) if i + 1 < n else int(ids.numel())
            cut = ids[a:b][:MAX_LEN]
            if cut.numel() == 0:
                cut = torch.zeros(1, dtype=torch.long)
            seqs.append(cut)
        L = max(s.numel() for s in seqs)
        x = torch.stack([
            torch.cat([s, torch.zeros(L - s.numel(), dtype=torch.long)])
            for s in seqs
        ])
        pos = torch.arange(L).unsqueeze(0).expand(x.shape[0], -1)
        h = self.emb(x) + self.pos(pos)
        pad = (x == 0)  # 0 is the dummy bucket (empty-bag marker)
        y = self.enc(h, src_key_padding_mask=pad)
        m = (~pad).float().unsqueeze(-1)
        pooled = (y * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def encode_batch(texts):
    """Flat bucket ids + offsets; empty bags get one dummy bucket."""
    ids, offsets = [], []
    for t in texts:
        offsets.append(len(ids))
        f = list(emo_features(t))
        if f:
            ids.extend(f)
        else:
            ids.append(0)
    if not ids:
        ids = [0]
        offsets = [0]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
