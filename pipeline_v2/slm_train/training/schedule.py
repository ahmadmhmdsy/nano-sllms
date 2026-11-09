"""Config-driven optimizer and LR-schedule factories.

Both exist because the config fields train.optimizer / train.scheduler /
train.warmup_steps / train.weight_decay are certifications: the run must do
what the file says. The scheduler advances exactly once per optimizer step, so
a resumed run reconstructs the same LR a never-interrupted run had — its real
state is carried in every checkpoint.
"""
from __future__ import annotations

import math

import torch

__all__ = ["build_optimizer", "build_scheduler"]


def build_optimizer(train_config, parameters, learning_rate: float):
    """Build the optimizer the config declared; refuse the undeclared ones."""
    optimizer_name = str(train_config.optimizer)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(getattr(train_config, "weight_decay", 0.0) or 0.0),
        )
    # train.optimizer is a Literal in config/models.py, so reaching this line
    # means the schema and the factory disagree; fail closed either way.
    raise ValueError("unsupported train.optimizer: " + optimizer_name)


def build_scheduler(train_config, optimizer, max_steps: int):
    """The declared schedule as a real torch LR scheduler.

    Warmup (train.warmup_steps > 0) is a linear ramp over the first
    warmup_steps optimizer steps; after warmup, the declared decay applies over
    max_steps. The scheduler is stepped once per optimizer step and the
    checkpoint stores its true state, so resume continues the same curve.
    """
    warmup = int(getattr(train_config, "warmup_steps", 0) or 0)
    schedule = str(train_config.scheduler)
    if schedule not in ("cosine", "linear", "constant"):
        raise ValueError("unsupported train.scheduler: " + schedule)
    total_steps = max(1, int(max_steps))

    def factor(step: int) -> float:
        step = max(0, int(step))
        if warmup and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == "linear":
            return max(1e-6, 1.0 - progress)
        raise ValueError("unsupported train.scheduler: " + schedule)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
