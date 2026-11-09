"""Security: four rules with no exceptions.

paths          containment proved after resolving every symlink
model_loading  safetensors only, weights-only, and no config-named types
secrets        secret values removed by value, not only by key
export         an explicit allowlist, checked, and passed through unchanged
"""
from slm_train.security.export import ExportError, ExportRequest, export_allowlist
from slm_train.security.export import resolve_allowlist
from slm_train.security.model_loading import ModelSecurityError
from slm_train.security.model_loading import require_no_dynamic_type
from slm_train.security.model_loading import require_safe_weight_suffix
from slm_train.security.paths import PathSecurityError
from slm_train.security.paths import is_contained, require_contained, require_regular_file
from slm_train.security.secrets import REDACTED, redact, redact_mapping
from slm_train.security.secrets import sensitive_environment, value_is_sensitive

__all__ = [
    "ExportError",
    "ExportRequest",
    "ModelSecurityError",
    "PathSecurityError",
    "REDACTED",
    "export_allowlist",
    "is_contained",
    "redact",
    "redact_mapping",
    "require_contained",
    "require_no_dynamic_type",
    "require_regular_file",
    "require_safe_weight_suffix",
    "resolve_allowlist",
    "sensitive_environment",
    "value_is_sensitive",
]
