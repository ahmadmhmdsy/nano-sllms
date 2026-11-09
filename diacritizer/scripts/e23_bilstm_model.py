"""E-23d arm D1 - char-level BiLSTM(3x256, bidirectional) + Bahdanau attention
diacritizer, re-implemented to the SAME I/O contract as our transformer
models so bench.py/gate plumbing works unchanged: input_ids [B, ctx] int64
(our TK vocab, PAD-padded), forward -> {"logits": [B, ctx, 15]}.
Pads are masked by the training loop (labels == -1).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.W(hidden_states).unsqueeze(2)
        keys = self.U(hidden_states).unsqueeze(1)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, hidden_states)
        return context


class BiLSTMDiacritizer(nn.Module):
    """Embedding(128) -> BiLSTM(256x2, 3 layers) -> attention -> linear(15)."""

    def __init__(self, vocab_size: int, embed_dim=128, hidden_dim=256,
                 num_layers=3, num_classes=15, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> dict:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.classifier(x)
        return {"logits": x}


def build(vocab_size: int) -> BiLSTMDiacritizer:
    return BiLSTMDiacritizer(vocab_size)
