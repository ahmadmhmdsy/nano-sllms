"""Committee distillation helpers (MU1 arm D, EXPERIMENTS row E-29).

Two jobs:

1. TeacherEnsemble loads the four 12K expert finals
   (runs/mex/archive_12000/{x1..x4}/final), builds each one with
   src.model.build_model from its experiment config and averages their
   logits for the same input ids (fp32, eval mode, no_grad). Lazy CPU
   inference of 4 teachers per step would dominate training, so production
   training reads a PRECOMPUTED logits cache instead (see
   mex/scripts/distill_cache.py); the lazy path exists for tests/verification.
2. kd_loss -- the distillation objective:

   loss = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
        + (1 - alpha) * CE(student, labels)

   with both logits shifted by one (causal-LM convention, matching the
   internal label shift in src/model.py; packed data uses labels==input_ids).
   At a cache MISS (block not covered by the cache file)
   mex/scripts/train_distill.py substitutes teacher logits 0 AND masks the
   KL term for that block, so the loss degrades gracefully to pure CE; the
   id-0/pad fallback never touches the student input ids themselves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: E402  (used by TeacherLogitCache)
import torch
import torch.nn.functional as F


class TeacherEnsemble:
    """Mean of N teacher logits for identical input ids.

    Teachers share tokenizer/vocab (97) and ctx (96) but differ in hidden
    size; only logits are compared, so the ensemble is shape-safe. Each
    teacher is built with build_model from its experiment config plus the
    safetensors of its final dir, moved to *device*, set to eval() and
    called under torch.no_grad().
    """

    def __init__(self, configs: list[dict], final_dirs: list[str | Path],
                 device: str | torch.device = "cpu"):
        from src.model import build_model, load_finetune_init  # repo-root import

        if not configs:
            raise ValueError("TeacherEnsemble needs at least one teacher")
        self.device = torch.device(device)
        self.models = []
        for cfg, final_dir in zip(configs, final_dirs):
            vocab = int(cfg["tokenizer"]["vocab_size"])
            model = build_model(cfg, vocab_size=vocab)
            load_finetune_init(model, Path(final_dir))
            model.eval().to(self.device)
            self.models.append(model)

    @torch.no_grad()
    def ensemble_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-teacher logits, averaged -> float32 [B, L, vocab]."""
        logits = None
        with torch.no_grad():
            for model in self.models:
                out = model(input_ids=input_ids.to(self.device)).logits
                out = out.float()
                logits = out if logits is None else logits + out
        return logits / len(self.models)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            labels: torch.Tensor, alpha: float = 0.5,
            temperature: float = 1.0) -> torch.Tensor:
    """Combined distillation + CE loss over the SHIFTED positions.

    Mirrors the label shift of the causal LM (logits[:, :-1] vs
    labels[:, 1:]) so the packed labels == input_ids stream keeps the same
    CE signature. Label -100 positions are skipped in both terms. Student
    logits are upcast to fp32 before the softmaxes (fp16-safe). Both terms
    use batchmean over valid positions so their scales agree. At T = 1 the
    T^2 factor is 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    valid = shift_labels != -100
    if int(valid.sum()) == 0:
        return student_logits.sum() * 0.0
    s_logits = shift_logits.float()[valid]     # [N, vocab]
    t_logits = shift_teacher.float()[valid]    # [N, vocab]
    # mean over the N valid positions == batchmean here (flat rows are all
    # one token each, so no batch-size renorm is needed; "batchmean" is not
    # a legal cross_entropy reduction and fails at call time).
    ce = F.cross_entropy(s_logits, shift_labels[valid], reduction="mean")
    if alpha == 0.0:
        return ce
    T2 = float(temperature) * float(temperature)
    log_p_s = F.log_softmax(s_logits / temperature, dim=-1)
    p_t = F.softmax(t_logits / temperature, dim=-1)
    # F.kl_div(input=log p_student, target=p_teacher) == KL(teacher || student)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return alpha * kl * T2 + (1.0 - alpha) * ce


class TeacherLogitCache:
    """Memmap-backed teacher-logit cache aligned to PackedDataset blocks.

    Layout (built by mex/scripts/distill_cache.py): an .npy of shape
    [n_blocks, seq_len, vocab], row i == averaged fp32 teacher logits for
    dataset block i. Lookup is O(1) (memmap row); the row count always
    matches the PackedDataset built from the same sorted shards with the
    same seq_len, so cache rows and student blocks are position-aligned.
    On a MISS (index outside the cache rows, only possible with a partial
    sanity cache) block_logits returns zeros; train_distill masks the KL
    for such rows so the loss degrades to pure CE.
    """

    def __init__(self, path: str | Path, seq_len: int, vocab: int,
                 dtype: str = "float32"):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.vocab = int(vocab)
        if not self.path.is_file():
            raise FileNotFoundError(f"teacher logit cache not found: {self.path}")
        self.cache = np.load(self.path, mmap_mode="r")
        if self.cache.dtype != np.dtype(dtype):
            raise ValueError(
                f"cache dtype {self.cache.dtype} != expected {dtype} "
                f"({self.path})")
        if self.cache.ndim != 3:
            raise ValueError(f"cache must be 3-D, got {self.cache.ndim}d")
        if tuple(self.cache.shape[1:]) != (self.seq_len, self.vocab):
            raise ValueError(
                f"cache tail shape {tuple(self.cache.shape[1:])} != "
                f"{(self.seq_len, self.vocab)} ({self.path})")
        self.n_blocks = int(self.cache.shape[0])

    def block_logits(self, block_idx: int) -> np.ndarray:
        """O(1) memmap row; zeros [seq_len, vocab] on a miss."""
        if 0 <= block_idx < self.n_blocks:
            return self.cache[block_idx]
        return np.zeros((self.seq_len, self.vocab), dtype=self.cache.dtype)
