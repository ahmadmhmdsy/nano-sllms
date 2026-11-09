"""R1-constrained model construction and model manifests."""
from slm_train.models.factory import (
    ModelCapabilityError,
    build_causal_lm,
    build_model_manifest,
)

__all__ = ["ModelCapabilityError", "build_causal_lm", "build_model_manifest"]
