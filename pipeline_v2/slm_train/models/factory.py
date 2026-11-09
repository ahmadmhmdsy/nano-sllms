"""Re-exports for the R1 constrained causal-LM model factory."""
from slm_train.models.causal_lm import (
    SUPPORTED_ARCHITECTURES,
    ModelCapabilityError,
    build_causal_lm,
    validate_config_capabilities,
    validate_model_capabilities,
)
from slm_train.models.manifest import build_model_manifest

__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "ModelCapabilityError",
    "build_causal_lm",
    "build_model_manifest",
    "validate_config_capabilities",
    "validate_model_capabilities",
]
