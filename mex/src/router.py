"""Tiny prompt->task classifier for the mu1 arm-C dispatch router.

DESIGN.md #4 arm C ('dispatch router'): the whole unlabeled input goes to
ONE expert. This module holds the shared router architecture + load/save so
train_router.py (training) and eval_mex.py (mixed eval) cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from mex.src.vocab import CharVocab

MAX_LEN = 96  # same context budget as the experts (model.ctx)
TASKS = ["x1", "x2", "x3", "x4"]


class PromptRouter(nn.Module):
    """Embedding(vocab,32) -> masked mean-pool -> MLP 32->32->4 logits."""

    def __init__(self, vocab_size: int = 97, dim: int = 32,
                 n_tasks: int = len(TASKS)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, n_tasks)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids [B, L<=MAX_LEN]; mask 1 = real char, 0 = pad."""
        x = self.emb(ids)
        m = mask.unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.fc2(torch.relu(self.fc1(pooled)))


def encode_batch(prompts: list[str], vocab: CharVocab,
                 max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """CharVocab ids per prompt, clip to max_len, pad to batch width."""
    pad = vocab.vocab["<pad>"]
    seqs = [vocab.encode(p)[:max_len] for p in prompts]
    width = max((len(s) for s in seqs), default=1)
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs],
                       dtype=torch.long)
    mask = torch.tensor([[1.0] * len(s) + [0.0] * (width - len(s))
                         for s in seqs], dtype=torch.float32)
    return ids, mask


@torch.no_grad()
def route_tasks(model: nn.Module, prompts: list[str],
                vocab: CharVocab) -> list[str]:
    """Greedy argmax routing; returns one TASKS name per prompt."""
    was_training = model.training
    model.eval()
    out: list[str] = []
    bs = 1024
    for s in range(0, len(prompts), bs):
        ids, mask = encode_batch(prompts[s:s + bs], vocab)
        pred = model(ids, mask).argmax(dim=1)
        out.extend(TASKS[i] for i in pred.tolist())
    if was_training:
        model.train()
    return out


def save_router(path: Path, model: nn.Module, meta: dict) -> None:
    """state_dict in router.pt, tiny JSON meta beside it (router_meta.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    (path.parent / "router_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def load_router(path: Path) -> tuple[nn.Module, dict]:
    """Rebuild the router from router.pt (+ its router_meta.json)."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vocab_size, dim = ckpt["emb.weight"].shape
    model = PromptRouter(vocab_size=vocab_size, dim=dim)
    model.load_state_dict(ckpt)
    model.eval()
    meta_path = path.parent / "router_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — train_router.py writes it beside router.pt")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta
