"""Data manifest (immutable semantic fields only)."""
from typing import ClassVar, Mapping, Optional

from slm_train.contracts.base import SemanticManifest


class DataManifest(SemanticManifest):
    """Immutable data manifest.

    Verification time, observed file hashes, freshness status, and policy
    evaluation are mutable evidence in the mutable status contract and
    immutable freshness reports; they are excluded from the semantic hash by
    never being fields here.
    """

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    source_ids: tuple[str, ...] = ()
    source_roles: Mapping[str, str] = {}
    source_identity: Optional[Mapping[str, str]] = None
    source_fingerprint: str = ""
    freshness_policy: str = "pinned"
    supersession_policy: str = "explicit-generation"
    dataset_config: Mapping[str, object] = {}
    split_policy: Mapping[str, object] = {}
    filters: Mapping[str, object] = {}
    normalization: str = ""
    dedupe_mode: str = "none"
    dedupe_salt: Optional[str] = None
    document_id_fields: tuple[str, ...] = ()
    document_counts: Mapping[str, int] = {}
    license_note: str = ""
    provenance_note: str = ""
    overlap_report: Optional[Mapping[str, object]] = None
    truncation_counts: Mapping[str, int] = {}
    rejection_counts: Mapping[str, int] = {}
    regeneration_command: tuple[str, ...] = ()
    content_hash: str
