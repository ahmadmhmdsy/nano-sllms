"""Stage 0.5e (TASKS 139) - shared char-LM pieces for CTC N-best rescoring.

The LM lives in the teacher's own acoustic charset, so its per-character
log-probabilities are directly comparable with CTC log-probs. Padding uses the
CTC blank id (0), which the LM never predicts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# U+064B..U+065F + U+0670 -- the 165-char checkpoint charset has none of these.
AR_DIACRITICS = set(chr(c) for c in list(range(0x064B, 0x0660)) + [0x0670])


def strip_diacritics(text: str) -> str:
    return "".join(ch for ch in text if ch not in AR_DIACRITICS)


def id_char_table(charset):
    """Authoritative id -> char map, whatever shape the Charset object has."""
    chars = list(getattr(charset, "characters", []) or [])
    if chars and len(chars) == charset.vocab_size:
        return {i: c for i, c in enumerate(chars) if i != charset.blank_id}
    if chars and len(chars) == charset.vocab_size - 1:
        ids = [i for i in range(charset.vocab_size) if i != charset.blank_id]
        return dict(zip(ids, chars))
    table = {}
    for i in range(charset.vocab_size):
        if i == charset.blank_id:
            continue
        s = charset.decode([i])
        if len(s) == 1:
            table[i] = s
    return table


def lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


class CharLM(nn.Module):
    """Next-character GRU LM; row vocab (acoustic ids 1..V-1), row V is BOS."""

    def __init__(self, vocab: int, dim: int = 256, layers: int = 2):
        super().__init__()
        self.vocab = vocab
        self.bos = vocab
        self.embed = nn.Embedding(vocab + 1, dim, padding_idx=0)
        self.rnn = nn.GRU(dim, dim, num_layers=layers, batch_first=True)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x):
        h, _ = self.rnn(self.embed(x))
        return self.out(h)


def make_batch(seqs, bos: int):
    length = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), length, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    inp = torch.cat([torch.full((len(seqs), 1), bos, dtype=torch.long), x[:, :-1]], dim=1)
    return inp, x


@torch.no_grad()
def score_sequences(model, seqs, device, batch: int = 256):
    """Total log P(seq) under the LM for each candidate (empty -> 0.0)."""
    model.eval()
    out = [0.0] * len(seqs)
    idx = [i for i, s in enumerate(seqs) if s]
    for start in range(0, len(idx), batch):
        take = idx[start:start + batch]
        chunk = [seqs[i] for i in take]
        x = torch.zeros(len(chunk), max(len(s) for s in chunk), dtype=torch.long, device=device)
        for j, s in enumerate(chunk):
            x[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        inp = torch.cat([torch.full((len(chunk), 1), model.bos, dtype=torch.long, device=device),
                         x[:, :-1]], dim=1)
        logp = torch.log_softmax(model(inp).float(), dim=-1)
        picked = logp.gather(2, x.clamp(min=1).unsqueeze(2)).squeeze(2)
        out_t = (picked * (x > 0).float()).sum(dim=1)
        for j, i in enumerate(take):
            out[i] = float(out_t[j])
    return out
