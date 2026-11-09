"""Utility functions for training: safe state loading and output path validation."""
from __future__ import annotations

from pathlib import Path
import torch

from slm_train.contracts.validation import require_repository_relative
from slm_train.errors import PipelineError

__all__ = [
    "load_state_dict_safely",
    "resolve_output_dir",
]


def resolve_output_dir(value: str | None) -> str:
    """Every output path is explicit and repository-relative; no defaults."""
    if value is None:
        raise PipelineError("output path is required: no silent output default exists")
    try:
        require_repository_relative(value)
    except ValueError as error:
        raise PipelineError("explicit output path must be repository-relative: " + str(error)) from error
    return value


def load_state_dict_safely(path: Path):
    """State loading always uses weights_only=True; never executes pickles."""
    return torch.load(path, map_location="cpu", weights_only=True)
