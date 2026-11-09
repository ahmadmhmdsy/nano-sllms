"""Constrained R1 causal-LM construction (fp16 + SDPA GQA, CPU-only)."""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from slm_train.config.models import ModelConfig, R1_CAPABILITIES, R2_CAPABILITIES
from slm_train.errors import PipelineError

__all__ = ["ModelCapabilityError", "build_causal_lm", "validate_model_capabilities"]


class ModelCapabilityError(PipelineError):
    """Raised when a requested model capability is outside the R1 gate."""


SUPPORTED_ARCHITECTURES = frozenset({"llama"})
DEFAULT_VOCAB_SIZE = 64


def validate_model_capabilities(
    *,
    architecture: str,
    precision: str,
    attention: str,
    hidden: int,
    heads: int,
    kv_heads: int,
    layers: int,
    intermediate: int,
    context: int,
    capabilities: Sequence[str] = (),
    base_model: str | None = None,
    data_context: int | None = None,
) -> None:
    """Fail closed on every capability the R1 gate does not support."""
    # Every dimension must be a positive integer: the typed ModelConfig path is
    # protected by Pydantic 'gt=0', but this gate must stay fail-closed for any
    # bypass (e.g. model_construct) instead of building a broken model.
    for dimension, value in (
        ('hidden', hidden),
        ('heads', heads),
        ('kv_heads', kv_heads),
        ('layers', layers),
        ('intermediate', intermediate),
        ('context', context),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ModelCapabilityError(
                "model dimension {dimension!r} must be an integer >= 1; got {value!r}".format(dimension=dimension, value=value)
            )
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ModelCapabilityError(
            f"unsupported architecture for R1: {architecture!r} "
            f"(supported: {sorted(SUPPORTED_ARCHITECTURES)})"
        )
    if precision != "fp16":
        raise ModelCapabilityError(
            f"unsupported precision for R1: {precision!r} (R1 requires fp16)"
        )
    if attention != "sdpa":
        raise ModelCapabilityError(
            f"unsupported attention backend for R1: {attention!r} "
            "(R1 requires sdpa)"
        )
    requested = tuple(capabilities)
    unknown = sorted(set(requested) - R1_CAPABILITIES)
    if unknown:
        if set(unknown) <= R2_CAPABILITIES:
            raise ModelCapabilityError(
                f"R2-only model capabilities rejected under R1: {unknown}"
            )
        raise ModelCapabilityError(f"unknown model capabilities rejected: {unknown}")
    if base_model is not None:
        raise ModelCapabilityError(
            f"base_model mounting is an R2 capability, rejected under R1: {base_model!r}"
        )
    if hidden % heads:
        raise ModelCapabilityError(
            f"hidden size {hidden} is not divisible by heads {heads}"
        )
    if hidden % kv_heads:
        raise ModelCapabilityError(
            f"hidden size {hidden} is not divisible by kv_heads {kv_heads}"
        )
    if heads % kv_heads:
        raise ModelCapabilityError(
            f"heads {heads} must be divisible by kv_heads {kv_heads}"
        )
    if data_context is not None and data_context != context:
        raise ModelCapabilityError(
            f"context mismatch: model context {context} != data/alignment context {data_context}"
        )


def validate_config_capabilities(
    model_config: ModelConfig,
    *,
    capabilities: Sequence[str] = (),
    data_context: int | None = None,
) -> None:
    """Apply the R1 gate to a typed ModelConfig."""
    validate_model_capabilities(
        architecture=model_config.architecture,
        precision=model_config.precision,
        attention=model_config.attention,
        hidden=model_config.hidden,
        heads=model_config.heads,
        kv_heads=model_config.kv_heads,
        layers=model_config.layers,
        intermediate=model_config.intermediate,
        context=model_config.context,
        capabilities=capabilities,
        base_model=model_config.base_model,
        data_context=data_context,
    )


def build_causal_lm(
    model_config: ModelConfig,
    *,
    capabilities: Sequence[str] = (),
    data_context: int | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = 0,
):
    """Build an R1-constrained GQA causal-LM on CPU; never coerce a rejected capability."""
    validate_config_capabilities(
        model_config, capabilities=capabilities, data_context=data_context,
    )
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.hidden,
        intermediate_size=model_config.intermediate,
        num_hidden_layers=model_config.layers,
        num_attention_heads=model_config.heads,
        num_key_value_heads=model_config.kv_heads,
        max_position_embeddings=model_config.context,
        tie_word_embeddings=model_config.tied_embeddings,
        # fp16-only: the R1 gate above already rejected any other precision.
        torch_dtype=torch.float16,
    )
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(
        llama_config,
        attn_implementation=model_config.attention,
    )
    model.eval()
    return model
