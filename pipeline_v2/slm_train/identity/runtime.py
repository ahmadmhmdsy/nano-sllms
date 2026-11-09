"""Runtime identity: records supplied platform facts, never invents them."""
from typing import ClassVar, Mapping

from pydantic import Field
from slm_train.contracts.base import SemanticManifest


class RuntimeIdentity(SemanticManifest):
    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    python_version: str
    venv_path: str
    package_lock: Mapping[str, object] = {}
    os: str
    cuda: str = "not-recorded"
    driver: str = "not-recorded"
    device_class: str = "not-recorded"
    device_name: str = "not-recorded"

    def runtime_hash(self) -> str:
        return self.semantic_manifest_hash()
