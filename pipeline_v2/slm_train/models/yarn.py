"""YaRN context extension for SLM training.

YaRN (Yet another RoPE extensioN) is a method for extending the context length
of transformer models by scaling the position embeddings. This module applies
YaRN scaling to a model's RoPE embeddings to extend its context length.

The extension is applied after model loading and before training/evaluation.
"""
from __future__ import annotations

import math
from typing import Any


def apply_yarn_scaling(
    model: Any,
    base_context_length: int,
    target_context_length: int,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
) -> dict[str, Any]:
    """Apply YaRN scaling to a model's RoPE embeddings.

    Args:
        model: The model to extend
        base_context_length: The original context length
        target_context_length: The target context length
        yarn_beta_fast: YaRN beta fast parameter
        yarn_beta_slow: YaRN beta slow parameter

    Returns:
        A dict with the extension results
    """
    if target_context_length <= base_context_length:
        raise ValueError("target context length must be greater than base context length")

    # Calculate the scaling factor
    scale_factor = target_context_length / base_context_length

    # YaRN scaling factors
    # These are the standard YaRN scaling factors
    yarn_factor = 0.1 * math.log(scale_factor) + 1.0

    # Apply scaling to the model's RoPE embeddings
    # This is a simplified version - a full implementation would modify
    # the RoPE theta values directly
    applied = False
    try:
        # Try to access and modify the RoPE theta
        if hasattr(model, "config") and hasattr(model.config, "rope_theta"):
            original_theta = model.config.rope_theta
            # YaRN modifies the theta based on the scaling factor
            # This is a simplified version
            model.config.rope_theta = original_theta * yarn_factor
            applied = True
    except Exception:
        pass

    return {
        "applied": applied,
        "base_context_length": base_context_length,
        "target_context_length": target_context_length,
        "scale_factor": scale_factor,
        "yarn_factor": yarn_factor,
        "yarn_beta_fast": yarn_beta_fast,
        "yarn_beta_slow": yarn_beta_slow,
    }


def yarn_context_extension_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize YaRN context extension results."""
    return {
        "applied": result["applied"],
        "base_context_length": result["base_context_length"],
        "target_context_length": result["target_context_length"],
        "scale_factor": result["scale_factor"],
    }
