"""Model manifest construction for the R1 model factory."""
from __future__ import annotations

from typing import Mapping, Optional

from slm_train.config.models import ModelConfig
from slm_train.contracts.hash import canonical_sha256
from slm_train.contracts.model import ModelManifest

__all__ = ["build_model_manifest"]


def build_model_manifest(
    *,
    model_config: ModelConfig,
    model,
    attn_implementation: str,
    source_hash: str,
    base_model: str | None = None,
    parent_checkpoint_identity: Optional[str] = None,
    parent_checkpoint_content_hash: Optional[str] = None,
    package_versions: Mapping[str, str] | None = None,
) -> ModelManifest:
    """Record the built model identity; absent base/parent identity stays null."""
    if package_versions is None:
        import torch
        import transformers

        package_versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    capability_metadata = {
        "hidden": model_config.hidden,
        "intermediate": model_config.intermediate,
        "layers": model_config.layers,
        "heads": model_config.heads,
        "kv_heads": model_config.kv_heads,
        "kv_head_groups": model_config.heads // model_config.kv_heads,
        "context": model_config.context,
        "vocab_size": int(model.config.vocab_size),
        "tied_embeddings": model_config.tied_embeddings,
        "precision": model_config.precision,
        "attention": model_config.attention,
        "attn_implementation": attn_implementation,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "package_versions": dict(package_versions),
        "source_hash": source_hash,
    }
    return ModelManifest(
        schema_version=1,
        architecture=model_config.architecture,
        config_hash=canonical_sha256(model_config.model_dump()),
        initialization="random",
        base_model_identity=base_model,
        parent_checkpoint_identity=parent_checkpoint_identity,
        parent_checkpoint_content_hash=parent_checkpoint_content_hash,
        capability_metadata=capability_metadata,
    )
