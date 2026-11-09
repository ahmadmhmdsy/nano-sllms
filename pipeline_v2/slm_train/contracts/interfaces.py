"""Trusted interfaces: the R1 protocol set, its typed models, and the extension boundary.

This module is pure contract surface. It imports no torch, no transformers, and
nothing from V1. Every extension point the design allows is declared here as data
(typed models plus a Protocol), never as a discovery mechanism: there is no entry
point scan, no directory walk, no remote lookup, and no dynamic import.

Naming note: the design table gives the checkpoint-selection PROTOCOL the same name
as the policy CONFIG it consumes. The protocol keeps the plain name (SelectionPolicy)
and the configuration model is SelectionPolicyConfig, so the two never shadow one
another at the call site.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from pydantic import Field, model_validator

from slm_train.config.models import ModelConfig, PackingConfig, SourceDeclaration
from slm_train.contracts.base import ContractModel, SemanticManifest, ValidationLabel
from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.contracts.evaluation import EvaluationReport
from slm_train.contracts.final import FinalPublicationManifest
from slm_train.contracts.model import ModelManifest
from slm_train.contracts.source import SourceManifest
from slm_train.contracts.tokenized import TokenizedDataManifest
from slm_train.contracts.tokenizer import TokenizerManifest

if TYPE_CHECKING:  # keeps the training types out of the runtime import graph
    from slm_train.training.spec import TrainingContext, TrainingResult


# --- typed models the R1 protocols exchange ----------------------------------


class SourceRow(ContractModel):
    """One declared source row; text is the only untyped part of the pipeline."""
    schema_version: int = Field(default=1, ge=1)

    row_id: str = Field(min_length=1)
    text: str
    source_manifest_hash: str = Field(min_length=64, max_length=64)
    declared_role: str = Field(min_length=1)


class ProvenanceRecord(SemanticManifest):
    """How one prepared generation came from its declared source, never invented."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    dataset_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    source_manifest_hash: str = Field(min_length=64, max_length=64)
    source_kind: str = Field(min_length=1)
    declared_source_id: str = Field(min_length=1)
    declared_role: str = Field(min_length=1)
    transform_chain: tuple[str, ...] = ()
    tokenizer_content_hash: str = ""
    prepared_at: str = ""


class TransformReport(ContractModel):
    """Per-transform accounting; counts are observed, never assumed."""
    schema_version: int = Field(default=1, ge=1)

    transform: str = Field(min_length=1)
    rows_in: int = Field(ge=0)
    rows_out: int = Field(ge=0)
    rows_dropped: int = Field(ge=0)
    drop_reasons: dict[str, int] = {}

    @model_validator(mode="after")
    def _accounting_balances(self) -> "TransformReport":
        if self.rows_in != self.rows_out + self.rows_dropped:
            raise ValueError("rows_in must equal rows_out plus rows_dropped")
        if sum(self.drop_reasons.values()) != self.rows_dropped:
            raise ValueError("drop_reasons must account for every dropped row")
        return self


class DeduplicationResult(ContractModel):
    """Deduplication outcome; the kept set is a decision, not a silent default."""
    schema_version: int = Field(default=1, ge=1)

    strategy: str = Field(min_length=1)
    rows_in: int = Field(ge=0)
    rows_kept: int = Field(ge=0)
    rows_removed: int = Field(ge=0)
    exact_hash_collisions: int = Field(ge=0)
    content_merges: int = Field(ge=0)
    key_hash: str = ""

    @model_validator(mode="after")
    def _removed_matches_counts(self) -> "DeduplicationResult":
        if self.rows_in != self.rows_kept + self.rows_removed:
            raise ValueError("rows_in must equal rows_kept plus rows_removed")
        return self


class PreparedRows(ContractModel):
    """Prepared, deduplicated rows bound to one provenance record."""
    schema_version: int = Field(default=1, ge=1)

    provenance_hash: str = Field(min_length=64, max_length=64)
    rows: tuple[SourceRow, ...]
    total_rows: int = Field(ge=0)

    @model_validator(mode="after")
    def _count_matches(self) -> "PreparedRows":
        if self.total_rows != len(self.rows):
            raise ValueError("total_rows must equal the number of carried rows")
        return self


class SplitPolicy(ContractModel):
    """Declared split fractions; they must sum to exactly one."""
    schema_version: int = Field(default=1, ge=1)

    train: float = Field(gt=0, lt=1)
    validation: float = Field(ge=0, lt=1)
    test: float = Field(default=0, ge=0, lt=1)
    strategy: str = Field(default="hash_deterministic", min_length=1)

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "SplitPolicy":
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class SplitResult(ContractModel):
    """Split membership is content-addressed, so it is reproducible from the key."""
    schema_version: int = Field(default=1, ge=1)

    policy_hash: str = Field(min_length=64, max_length=64)
    key_hash: str = Field(min_length=64, max_length=64)
    train_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    test_row_ids: tuple[str, ...] = ()
    overlap_row_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _splits_are_disjoint(self) -> "SplitResult":
        if self.overlap_row_ids:
            raise ValueError("split membership must be disjoint")
        combined = set(self.train_row_ids)
        combined |= set(self.validation_row_ids)
        combined |= set(self.test_row_ids)
        declared = len(self.train_row_ids) + len(self.validation_row_ids)
        declared += len(self.test_row_ids)
        if len(combined) != declared:
            raise ValueError("a row may appear in at most one split")
        return self


class TokenBatch(ContractModel):
    """Token ids for prepared rows, with the per-row shapes needed to pack them."""
    schema_version: int = Field(default=1, ge=1)

    tokenizer_content_hash: str = Field(min_length=64, max_length=64)
    row_ids: tuple[str, ...]
    sequences: tuple[tuple[int, ...], ...]
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _shapes_align(self) -> "TokenBatch":
        if len(self.row_ids) != len(self.sequences):
            raise ValueError("row_ids and sequences must be the same length")
        if self.total_tokens != sum(len(sequence) for sequence in self.sequences):
            raise ValueError("total_tokens must equal the summed sequence lengths")
        return self


class ValidationReport(SemanticManifest):
    """Labeled outcome of a dataset validation; never a bare boolean."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    subject_manifest_hash: str = Field(min_length=64, max_length=64)
    label: ValidationLabel
    checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    artifact_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _label_agrees_with_findings(self) -> "ValidationReport":
        if self.label == "PASS" and (self.failed_checks or self.findings):
            raise ValueError("a PASS report cannot carry failed checks or findings")
        if self.label in {"FAIL", "BLOCKED"} and not (self.failed_checks or self.findings):
            raise ValueError("a failing report must name at least one failure")
        return self


class RuntimeProfile(ContractModel):
    """Supplied platform facts for a build; never probed behind the caller."""
    schema_version: int = Field(default=1, ge=1)

    python_version: str = Field(min_length=1)
    os: str = Field(min_length=1)
    device_class: str = Field(min_length=1)
    device_name: str = ""
    cuda: str = "not-recorded"
    driver: str = "not-recorded"
    venv_path: str = ""


class CheckpointState(ContractModel):
    """The resumable state of one checkpoint; weights stay in safetensors."""
    schema_version: int = Field(default=1, ge=1)

    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    weights_path: str = Field(min_length=1)
    weights_sha256: str = Field(min_length=64, max_length=64)
    optimizer_path: Optional[str] = None
    scheduler_path: Optional[str] = None
    sequence_length: int = Field(gt=0)
    consumed_batches: int = Field(ge=0)
    metric_values: dict[str, float] = {}


class EvaluationRequest(SemanticManifest):
    """What an evaluator is asked to measure, bound to its inputs."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    split_identity: str = Field(min_length=1)
    checkpoint_identity: str = Field(min_length=1)
    data_hash: str = Field(min_length=64, max_length=64)
    tokenizer_hash: str = Field(min_length=64, max_length=64)
    metric_schema_identity: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    seed: Optional[int] = None


class MetricSuiteReport(SemanticManifest):
    """A suite result always carrying an explicit label and its run."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    suite: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checkpoint_identity: str = Field(min_length=1)
    metric_definitions: dict[str, str] = {}
    metric_values: dict[str, float] = {}
    status: ValidationLabel
    expected_suite_gates: tuple[str, ...] = ()
    actual_suite_gates: tuple[str, ...] = ()


class GateRequest(SemanticManifest):
    """One gate check, with its identity and comparison declared."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    gate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    observed_value: float
    threshold: float
    direction: str = Field(pattern="^(min|max)$")
    artifact_paths: tuple[str, ...] = ()


class GateResult(SemanticManifest):
    """A gate outcome; an unmeasured gate is BLOCKED, never PASS."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    gate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: ValidationLabel
    metric_name: str = Field(min_length=1)
    observed_value: Optional[float] = None
    threshold: Optional[float] = None
    direction: Optional[str] = None
    reason: str = ""

    @model_validator(mode="after")
    def _label_matches_the_measurement(self) -> "GateResult":
        if self.status in {"PASS", "FAIL"} and self.observed_value is None:
            raise ValueError("a decided gate must record its observed value")
        if self.status != "PASS":
            return self
        if self.threshold is None or self.direction is None:
            raise ValueError("a PASS gate must record its threshold and direction")
        if self.direction == "min":
            met = self.observed_value <= self.threshold
        elif self.direction == "max":
            met = self.observed_value >= self.threshold
        else:
            raise ValueError("direction must be min or max")
        if not met:
            raise ValueError("PASS contradicts the recorded threshold and direction")
        return self


class SelectionPolicyConfig(ContractModel):
    """Which checkpoint wins; the comparator is declared, never inferred."""
    schema_version: int = Field(default=1, ge=1)

    metric_name: str = Field(min_length=1)
    direction: str = Field(pattern="^(min|max)$")
    split: str = Field(default="validation", min_length=1)
    tie_breaker: str = Field(default="lowest_step", min_length=1)
    require_finite: bool = True


class PublicationRequest(SemanticManifest):
    """A publication request, bound to the manifest it will publish."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    target_path: str = Field(min_length=1)
    immutable: bool = True
    payload_paths: tuple[str, ...] = ()
    source_manifest_hash: str = Field(min_length=64, max_length=64)


class PublicationReadBack(SemanticManifest):
    """What a read-back actually observed on disk, never what was intended."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    target_path: str = Field(min_length=1)
    observed_manifest_hash: str = Field(min_length=64, max_length=64)
    observed_payload_hashes: dict[str, str] = {}
    immutable: bool = True
    directory_durable: bool = True


class ResourceRequest(SemanticManifest):
    """A resource probe request; probing never changes the probed system."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    resources: tuple[str, ...] = ("memory", "disk")
    require_gpu: bool = False


class ResourceReport(SemanticManifest):
    """Observed resources. An unavailable resource is BLOCKED, never zero."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    status: ValidationLabel
    observations: dict[str, object] = {}
    unavailable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _blocked_names_its_unavailability(self) -> "ResourceReport":
        if self.status == "BLOCKED" and not self.unavailable:
            raise ValueError("a BLOCKED resource report must name what was unavailable")
        return self


class HealthSample(SemanticManifest):
    """One point-in-time health observation supplied by the monitor."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    loss: Optional[float] = None
    loss_delta: Optional[float] = None
    gradient_norm: Optional[float] = None
    throughput: Optional[float] = None


class HealthResult(SemanticManifest):
    """A health verdict; a divergence signal is never averaged away."""
    schema_version: int = Field(default=1, ge=1)

    MUTABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    status: ValidationLabel
    diverging: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def _divergence_is_labeled(self) -> "HealthResult":
        if self.diverging and self.status == "PASS":
            raise ValueError("a diverging sample cannot be labeled PASS")
        return self


# --- the R1 protocol set ------------------------------------------------------


@runtime_checkable
class SourceReader(Protocol):
    """Reads a declared source. Identity is proven, never assumed."""

    def identity(self) -> SourceManifest: ...

    def provenance(self) -> ProvenanceRecord: ...

    def iter_rows(self) -> Iterator[SourceRow]: ...


@runtime_checkable
class RowTransformer(Protocol):
    """Maps a row to a row and accounts for everything it dropped."""

    def transform(self, row: SourceRow) -> tuple[SourceRow, TransformReport]: ...


@runtime_checkable
class Deduplicator(Protocol):
    """Collapses duplicates under a declared, reported key."""

    def deduplicate(self, rows: Iterable[SourceRow]) -> DeduplicationResult: ...


@runtime_checkable
class Splitter(Protocol):
    """Partitions prepared rows into disjoint, content-addressed splits."""

    def split(self, rows: PreparedRows, policy: SplitPolicy) -> SplitResult: ...


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Encodes with a pinned tokenizer; there is no implicit fallback."""

    def encode(
        self, rows: PreparedRows, tokenizer: TokenizerManifest
    ) -> TokenBatch: ...


@runtime_checkable
class Packer(Protocol):
    """Packs token batches under a declared packing configuration."""

    def pack(self, batch: TokenBatch, config: PackingConfig) -> TokenizedDataManifest: ...


@runtime_checkable
class DatasetValidator(Protocol):
    """Validates a packed dataset and always returns a labeled report."""

    def validate(self, manifest: TokenizedDataManifest) -> ValidationReport: ...


@runtime_checkable
class ModelFactory(Protocol):
    """Builds a model for supplied runtime facts; it never probes the device."""

    def build(
        self, config: ModelConfig, runtime: RuntimeProfile
    ) -> tuple[ModelManifest, object]: ...


@runtime_checkable
class FeatureModule(Protocol):
    """A feature wraps training without owning any lifecycle authority."""

    def apply(self, context: "TrainingContext") -> "TrainingResult": ...


@runtime_checkable
class CheckpointAdapter(Protocol):
    """Publishes and reads back resumable checkpoint state."""

    def publish(self, state: CheckpointState) -> CheckpointManifest: ...

    def read_back(self, state: CheckpointState) -> CheckpointState: ...


@runtime_checkable
class Evaluator(Protocol):
    """Measures a checkpoint against a declared request."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationReport: ...


@runtime_checkable
class MetricSuite(Protocol):
    """Collects a named metric suite and always reports a status."""

    def collect(self, request: EvaluationRequest) -> MetricSuiteReport: ...


@runtime_checkable
class Gate(Protocol):
    """Applies one declared comparison and labels the outcome."""

    def check(self, request: GateRequest) -> GateResult: ...


@runtime_checkable
class SelectionPolicy(Protocol):
    """Picks a winning checkpoint under a declared comparator."""

    def select(
        self, candidates: Sequence[CheckpointManifest], config: SelectionPolicyConfig
    ) -> CheckpointManifest: ...


@runtime_checkable
class ArtifactPublisher(Protocol):
    """Publishes artifacts atomically and proves them by reading back."""

    def publish(self, request: PublicationRequest) -> FinalPublicationManifest: ...

    def read_back(self, request: PublicationRequest) -> PublicationReadBack: ...


@runtime_checkable
class ResourceProbe(Protocol):
    """Observes resources without changing them."""

    def probe(self, request: ResourceRequest) -> ResourceReport: ...


@runtime_checkable
class HealthMonitor(Protocol):
    """Turns samples into an explicit, labeled health verdict."""

    def check(self, sample: HealthSample) -> HealthResult: ...



# --- machine-readable extension boundary --------------------------------------



#: typed inputs each protocol declares, by name (resolved after class definition)
INPUT_MODELS: dict[str, tuple[str, ...]] = {
    "SourceReader": ("SourceDeclaration",),
    "RowTransformer": ("SourceRow",),
    "Deduplicator": ("SourceRow",),
    "Splitter": ("PreparedRows", "SplitPolicy"),
    "TokenizerAdapter": ("PreparedRows", "TokenizerManifest"),
    "Packer": ("TokenBatch", "PackingConfig"),
    "DatasetValidator": ("TokenizedDataManifest",),
    "ModelFactory": ("ModelConfig", "RuntimeProfile"),
    "FeatureModule": ("TrainingContext",),
    "CheckpointAdapter": ("CheckpointState",),
    "Evaluator": ("EvaluationRequest",),
    "MetricSuite": ("EvaluationRequest",),
    "Gate": ("GateRequest",),
    "SelectionPolicy": ("CheckpointManifest", "SelectionPolicyConfig"),
    "ArtifactPublisher": ("PublicationRequest",),
    "ResourceProbe": ("ResourceRequest",),
    "HealthMonitor": ("HealthSample",),
}


#: typed outputs each protocol declares, by name
OUTPUT_MODELS: dict[str, tuple[str, ...]] = {
    "SourceReader": ("SourceManifest", "ProvenanceRecord", "SourceRow"),
    "RowTransformer": ("SourceRow", "TransformReport"),
    "Deduplicator": ("DeduplicationResult",),
    "Splitter": ("SplitResult",),
    "TokenizerAdapter": ("TokenBatch",),
    "Packer": ("TokenizedDataManifest",),
    "DatasetValidator": ("ValidationReport",),
    "ModelFactory": ("ModelManifest",),
    "FeatureModule": ("TrainingResult",),
    "CheckpointAdapter": ("CheckpointManifest", "CheckpointState"),
    "Evaluator": ("EvaluationReport",),
    "MetricSuite": ("MetricSuiteReport",),
    "Gate": ("GateResult",),
    "SelectionPolicy": ("CheckpointManifest",),
    "ArtifactPublisher": ("FinalPublicationManifest", "PublicationReadBack"),
    "ResourceProbe": ("ResourceReport",),
    "HealthMonitor": ("HealthResult",),
}


#: every typed model the R1 protocols exchange, for structural checks
TYPED_MODEL_NAMES: tuple[str, ...] = (
    "SourceRow",
    "ProvenanceRecord",
    "TransformReport",
    "DeduplicationResult",
    "PreparedRows",
    "SplitPolicy",
    "SplitResult",
    "TokenBatch",
    "ValidationReport",
    "RuntimeProfile",
    "CheckpointState",
    "EvaluationRequest",
    "MetricSuiteReport",
    "GateRequest",
    "GateResult",
    "SelectionPolicyConfig",
    "PublicationRequest",
    "PublicationReadBack",
    "ResourceRequest",
    "ResourceReport",
    "HealthSample",
    "HealthResult",
)


#: the exact R1 method map, used to check every protocol and its declaration
R1_METHOD_MAP: dict[str, tuple[str, ...]] = {
    "SourceReader": ("identity", "provenance", "iter_rows"),
    "RowTransformer": ("transform",),
    "Deduplicator": ("deduplicate",),
    "Splitter": ("split",),
    "TokenizerAdapter": ("encode",),
    "Packer": ("pack",),
    "DatasetValidator": ("validate",),
    "ModelFactory": ("build",),
    "FeatureModule": ("apply",),
    "CheckpointAdapter": ("publish", "read_back"),
    "Evaluator": ("evaluate",),
    "MetricSuite": ("collect",),
    "Gate": ("check",),
    "SelectionPolicy": ("select",),
    "ArtifactPublisher": ("publish", "read_back"),
    "ResourceProbe": ("probe",),
    "HealthMonitor": ("check",),
}


#: capabilities no extension may ever hold, whatever it registers as
FORBIDDEN_CAPABILITIES: tuple[str, ...] = (
    "acquire_lease",
    "rewrite_immutable_manifest",
    "mark_run_successful",
    "publish_outside_allowlist",
)


#: capabilities an extension may request, per protocol
ALLOWED_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "SourceReader": ("read_rows", "report_provenance"),
    "RowTransformer": ("transform_rows",),
    "Deduplicator": ("deduplicate_rows",),
    "Splitter": ("assign_splits",),
    "TokenizerAdapter": ("encode_rows",),
    "Packer": ("pack_tokens",),
    "DatasetValidator": ("validate_dataset",),
    "ModelFactory": ("build_model",),
    "FeatureModule": ("wrap_training",),
    "CheckpointAdapter": ("write_checkpoint", "read_checkpoint"),
    "Evaluator": ("evaluate",),
    "MetricSuite": ("collect_metrics",),
    "Gate": ("check_gate",),
    "SelectionPolicy": ("select_checkpoint",),
    "ArtifactPublisher": ("publish_artifact",),
    "ResourceProbe": ("probe_resources",),
    "HealthMonitor": ("monitor_health",),
}


#: repository-relative roots each protocol may write under, per protocol
ARTIFACT_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "SourceReader": ("data/v2/prepared/",),
    "RowTransformer": ("data/v2/prepared/",),
    "Deduplicator": ("data/v2/prepared/",),
    "Splitter": ("data/v2/prepared/",),
    "TokenizerAdapter": ("data/v2/tokens/",),
    "Packer": ("data/v2/tokens/",),
    "DatasetValidator": ("artifacts/v2/reports/",),
    "ModelFactory": ("artifacts/v2/models/",),
    "FeatureModule": ("artifacts/v2/features/",),
    "CheckpointAdapter": ("artifacts/v2/checkpoints/",),
    "Evaluator": ("artifacts/v2/reports/",),
    "MetricSuite": ("artifacts/v2/reports/",),
    "Gate": ("artifacts/v2/reports/",),
    "SelectionPolicy": ("artifacts/v2/selection/",),
    "ArtifactPublisher": ("artifacts/v2/published/",),
    "ResourceProbe": ("artifacts/v2/probe/",),
    "HealthMonitor": ("artifacts/v2/health/",),
}


#: the label a protocol returns when it cannot complete its declared work
FAILURE_LABEL: dict[str, ValidationLabel] = {
    name: "FAIL" for name in R1_METHOD_MAP
}


@dataclass(frozen=True)
class ProtocolDeclaration:
    """The machine-readable extension boundary for one R1 protocol."""

    name: str
    methods: tuple[str, ...]
    input_models: tuple[str, ...]
    output_models: tuple[str, ...]
    failure_label: ValidationLabel
    artifact_allowlist: tuple[str, ...]
    allowed_capabilities: tuple[str, ...]
    forbidden_capabilities: tuple[str, ...] = FORBIDDEN_CAPABILITIES


def _declaration(name: str) -> ProtocolDeclaration:
    protocol = globals()[name]
    return ProtocolDeclaration(
        name=name,
        methods=R1_METHOD_MAP[name],
        input_models=INPUT_MODELS[name],
        output_models=OUTPUT_MODELS[name],
        failure_label=FAILURE_LABEL[name],
        artifact_allowlist=ARTIFACT_ALLOWLIST[name],
        allowed_capabilities=ALLOWED_CAPABILITIES[name],
    )


_DECLARATIONS: dict[str, ProtocolDeclaration] = {
    name: _declaration(name) for name in R1_METHOD_MAP
}


def protocol_declaration(name: str) -> ProtocolDeclaration:
    """Return the declaration for a protocol, or raise for an unknown name."""
    if not isinstance(name, str) or name not in _DECLARATIONS:
        raise ValueError("unknown protocol: " + repr(name))
    return _DECLARATIONS[name]


def request_capability(name: str, capability: str) -> bool:
    """Return True when an extension may hold a capability; raise when it may not.

    A forbidden capability is refused for every protocol, and an unknown
    capability is refused too: there is no permissive default to fall back on.
    """
    declaration = protocol_declaration(name)
    if not isinstance(capability, str) or not capability:
        raise ValueError("capability must be a non-empty string")
    if capability in declaration.forbidden_capabilities:
        raise ValueError(
            "capability " + repr(capability) + " is forbidden for " + name
        )
    if capability not in declaration.allowed_capabilities:
        raise ValueError(
            "capability " + repr(capability) + " is not granted to " + name
        )
    return True


def require_allowed_artifact(name: str, path: str) -> str:
    """Raise unless path is repository-relative and strictly inside the allowlist.

    An allowlist entry names a writable root directory, so the root itself is not
    an artifact. Dot segments and empty segments are refused outright rather than
    normalized away, so no two spellings of the same file can pass differently.
    """
    from slm_train.contracts.validation import require_repository_relative

    declaration = protocol_declaration(name)
    require_repository_relative(path)
    normalized = path.replace(chr(92), "/").rstrip("/")
    segments = normalized.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("path must not contain empty, dot, or traversal segments")
    for root in declaration.artifact_allowlist:
        root = root.rstrip("/")
        if normalized == root:
            raise ValueError(
                "path " + repr(path) + " is the allowlist root itself, not an artifact"
            )
        if normalized.startswith(root + "/"):
            return path
    raise ValueError(
        "path " + repr(path) + " is outside the allowlist for " + name
    )
