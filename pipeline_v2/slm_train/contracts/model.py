"""Model manifest (immutable)."""
from typing import ClassVar, Mapping, Optional

from slm_train.contracts.base import SemanticManifest


class ModelManifest(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    architecture: str
    config_hash: str
    initialization: str = "random"
    base_model_identity: Optional[str] = None
    parent_checkpoint_identity: Optional[str] = None
    parent_checkpoint_content_hash: Optional[str] = None
    capability_metadata: Mapping[str, object] = {}
