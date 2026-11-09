"""Hashed bag-of-n-grams multinomial logistic regression.

One EmbeddingBag(num_buckets -> num_langs, mode='sum') + per-language bias:
exactly the "feature hashing, no tokenizer" lexical classifier of the
Tongue recipe. Trains on CPU in minutes at DA-1 scale.
"""
import torch

from langid.src import features


class HashedLangID(torch.nn.Module):
    def __init__(self, num_langs: int, num_buckets: int = 1 << 16):
        super().__init__()
        self.num_langs = num_langs
        self.num_buckets = num_buckets
        self.emb = torch.nn.EmbeddingBag(num_buckets, num_langs, mode="sum")
        self.bias = torch.nn.Parameter(torch.zeros(num_langs))
        torch.nn.init.zeros_(self.emb.weight)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.emb(ids, offsets) + self.bias


def encode_batch(texts):
    """Flat hash ids + bag offsets for a list of texts (EmbeddingBag input).

    Texts with zero letter features get one dummy bucket so no bag is empty;
    the eval harness reports those as abstains.
    """
    ids, offsets = [], []
    for t in texts:
        offsets.append(len(ids))
        f = list(features.text_features(t))
        if f:
            ids.extend(f)
        else:
            ids.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
