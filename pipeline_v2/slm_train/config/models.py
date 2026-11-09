"""Strict, frozen Pydantic models for R1 configuration documents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

R1_CAPABILITIES = frozenset({"pretrain"})

R2_CAPABILITIES = frozenset({"sft", "lora", "kd", "gdn", "mounting"})

SourceRole = Literal["train", "validation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RunConfig(StrictModel):
    name: str
    namespace: str
    stage: Literal["S", "P", "T"]
    seed: int = Field(ge=0)


class FreshnessPolicy(StrictModel):
    max_age: str
    supersession: str
    optional_reason: str | None = None


class SourceDeclaration(StrictModel):
    source_id: str
    kind: Literal["local", "huggingface"]
    path_or_repo: str
    revision: str | None
    file_set: tuple[str, ...]
    role: SourceRole
    license: str
    expected_rows: int = Field(ge=0)
    expected_tokens: int = Field(ge=0)
    freshness_policy: FreshnessPolicy
    optional_profile: str | None = None


class DataConfig(StrictModel):
    dataset_id: str
    sources: tuple[SourceDeclaration, ...]
    split_salt: int = Field(ge=0)
    validation_fraction: float = Field(gt=0, lt=1)
    dedupe: str
    # normalization is a CERTIFICATION: a value is legally usable only when the
    # data stage actually applies it. Supported: unicode_nfkc (NFKC then
    # whitespace collapse), ws-collapse (whitespace collapse only).
    normalization: Literal["unicode_nfkc", "ui_task_nl", "ws-collapse"]
    # filters are "field:value" min_chars / max_chars strings; unknown filter
    # names are rejected by validate_materialized hooks below.
    filters: tuple[str, ...]
    regeneration_command: tuple[str, ...]

    @field_validator("filters", mode="before")
    @classmethod
    def validate_filters(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        known = {"min_chars", "max_chars"}
        for item in value:
            if isinstance(item, str) and ":" in item:
                name, _sep, _rest = item.partition(":")
                if name.strip() not in known:
                    raise ValueError(
                        "unknown data filter: " + name.strip() + "; supported: "
                        + ", ".join(sorted(known))
                    )
        return value


class TokenizerConfig(StrictModel):
    source: str
    revision_or_files: str | tuple[str, ...]
    special_ids: dict[str, int]
    context: int = Field(gt=0)
    offsets: str


class PackingConfig(StrictModel):
    mode: Literal["concat", "document_masked"]
    sequence_length: int = Field(gt=0)
    overflow_policy: str
    dropped_tail_policy: str


class ModelConfig(StrictModel):
    architecture: Literal["llama"]
    hidden: int = Field(gt=0)
    intermediate: int = Field(gt=0)
    layers: int = Field(gt=0)
    heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    context: int = Field(gt=0)
    tied_embeddings: bool
    precision: Literal["fp16", "bf16"]
    attention: Literal["sdpa", "flash_attention_2"]
    base_model: str | None = None


class MonitorProbeConfig(StrictModel):
    """Native train monitor (E-005): multi-metric probes at the checkpoint
    cadence (honouring eval_fraction as the 1/N budget target), with the
    held-out probe TASKS file measured READ-ONLY. Decisions (selection,
    early stop) always come from the validation split, never the reserved
    test tasks."""

    enabled: bool = False
    eval_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    probe_tasks_file: str | None = None
    probe_tasks: int = Field(default=0, ge=0)
    probe_k: int = Field(default=1, ge=1)
    probe_pass_k: int = Field(default=1, ge=1)
    probe_temperature: float = Field(default=0.8, gt=0.0)
    probe_top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    overfit_patience: int = Field(default=2, ge=1)
    probe_max_new_tokens: int = Field(default=512, ge=64)


class KDConfig(StrictModel):
    """Native knowledge-distillation option (ui/E-008): teacher-logit KD
    wired into the train stage. Flat control arm = alpha/beta 0.5/0.5;
    distill-at-1/3-steps criterion = active_fraction 0.333."""

    enabled: bool = False
    teacher_checkpoint: str | None = None
    alpha_ce: float = Field(default=0.5, ge=0.0, le=1.0)
    beta_kl: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    active_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    fp32_logits: bool = True
    kl_chunk: int = Field(default=128, ge=16, le=1024)


class EarlyStopConfig(StrictModel):
    """Native early stop: patience in monitor PROBES on the validation loss,
    gated by a minimum trained fraction so a healthy run is never stopped
    mid-descent; a validity probe that still improves vetoes stopping."""

    enabled: bool = False
    patience_evals: int = Field(default=2, ge=1)
    min_steps_fraction: float = Field(default=0.5, gt=0.0, le=1.0)


class TrainingConfig(StrictModel):
    optimizer: Literal["adamw"]
    scheduler: Literal["cosine", "linear", "constant"]
    learning_rate: float = Field(gt=0)
    # Warmup steps and weight decay are real config inputs: the scheduler and
    # optimizer factories read them, and both honour the checkpoint boundary.
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    max_grad_norm: float = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    evaluation_cadence: int = Field(gt=0)
    checkpoint_policy: Literal["best", "last", "complete_only"]
    keep_last_checkpoints: int = Field(ge=0)
    keep_best_checkpoints: int = Field(ge=0)
    resume: bool = True
    # R2 adoption (2026-09-27 evidence: runs/v2/ui/u2g_arm_eos pass@4 35/40 ->
    # 40/40). When "final_n", the train stage averages the model weights of
    # the last _n complete checkpoints into checkpoints_avg/weight-average.pt
    # and the evaluation stages score THOSE weights.
    checkpoint_averaging: Literal["off", "final_n", "progressive"] = "off"
    checkpoint_averaging_n: int = Field(default=2, ge=1, le=8)
    # E-005 native monitor + early stop (user mandate 2026-09-27): declared,
    # reusable, never scrub code. Defaults keep every older config valid.
    monitor: MonitorProbeConfig = Field(default_factory=MonitorProbeConfig)
    kd: KDConfig = Field(default_factory=KDConfig)
    # E-007 native warm start (user mandate 2026-09-28): path to a
    # transplant .pt (embeddings + mask). Empty = disabled. The train
    # stage copies masked rows into the student embeddings at build time;
    # tied embeddings mean the lm_head is the same tensor.
    warm_start_embeddings: str = ""
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)

    @model_validator(mode="after")
    def validate_schedule(self) -> "TrainingConfig":
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


class EvaluatorSuiteConfig(StrictModel):
    ids: tuple[str, ...]
    versions: tuple[str, ...]
    suites: tuple[str, ...]
    metrics: tuple[str, ...]
    cadence_steps: int = Field(gt=0)
    thresholds: dict[str, float]
    directions: dict[str, Literal["min", "max"]]
    selection_policy: str
    # Decoding is a declared input, not a hidden constant: a run that declares
    # the generation suite states how many samples it drew and how long each
    # one is, so a recorded sample is reproducible from the config alone.
    generation_samples: int = Field(gt=0)
    generation_max_new_tokens: int = Field(gt=0)
    generation_temperature: float = Field(default=1.0, ge=0)
    generation_top_k: int = Field(default=50, gt=0)
    generation_top_p: float = Field(default=1.0, gt=0, le=1)


class RuntimeConfig(StrictModel):
    device: str
    precision: Literal["fp16", "bf16"]
    launch_profile: str
    vram_fraction: float = Field(gt=0, le=1)


class PromotionReference(StrictModel):
    predecessor_stage: Literal["S", "P"]
    predecessor_run_id: str
    checkpoint: str
    data_manifest: str
    evaluator_manifest: str
    gate_report: str


class FeaturesConfig(StrictModel):
    retention: Literal["manual"]
    promotion: PromotionReference | None


class ConfigDocument(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "$comment": (
                "Pydantic model_json_schema cannot express model-level cross-field validators "
                "(context, precision, evaluator, and promotion relationships). "
                "Runtime Pydantic validation is authoritative."
            )
        }
    )

    schema_version: Literal[2]
    release: Literal["r1"]
    capabilities: tuple[Literal["pretrain"], ...] = Field(min_length=1, max_length=1)
    run: RunConfig
    extends: tuple[str, ...] = ()
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    train: TrainingConfig
    evaluation: EvaluatorSuiteConfig
    runtime: RuntimeConfig
    features: FeaturesConfig
    packing: PackingConfig
    # Phase 3 (operator-friction plan, item 7): a single-key context alias.
    # The three real fields stay authoritative; this alias is only accepted
    # when it agrees with all of them.
    context: int | None = Field(default=None)

    @field_validator("capabilities", mode="before")
    @classmethod
    def validate_capabilities(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            unknown = set(value) - R1_CAPABILITIES
            if unknown:
                raise ValueError(f"R1 rejects capabilities: {sorted(unknown)}")
            if tuple(value) != ("pretrain",):
                raise ValueError("R1 requires exactly the pretrain capability")
        return value

    @model_validator(mode="after")
    def validate_release(self) -> "ConfigDocument":
        unknown = set(self.capabilities) - R1_CAPABILITIES
        if unknown:
            raise ValueError(f"R1 rejects capabilities: {sorted(unknown)}")
        if self.capabilities != ("pretrain",):
            raise ValueError("R1 requires exactly the pretrain capability")
        promotion = self.features.promotion
        if self.context is not None:
            alias = int(self.context)
            if (alias != self.model.context
                    or alias != self.tokenizer.context
                    or alias != self.packing.sequence_length):
                raise ValueError(
                    "top-level context " + str(alias) + " disagrees with the"
                    + " per-section values (model " + str(self.model.context)
                    + ", tokenizer " + str(self.tokenizer.context)
                    + ", packing " + str(self.packing.sequence_length) + ")"
                )
        for source in self.data.sources:
            if source.kind == "huggingface" and not source.revision:
                raise ValueError("Hugging Face sources require a pinned revision")
        if len(self.data.sources) > 1:
            # Declared but not consumable is a lie: a manifest that names
            # sources no stage ever opened is exactly the false evidence class
            # this release refuses. Fail here, honestly, until the
            # multi-source preparation lane exists.
            raise ValueError(
                "R1 preparation consumes exactly one declared source; "
                "declare one source and record the rest as not_migrated")
            if not source.file_set:
                raise ValueError("every source requires an explicit file set")
        if self.run.stage == "S" and promotion is not None:
            raise ValueError("S cannot reference a predecessor")
        if self.run.stage == "P" and promotion is not None and promotion.predecessor_stage != "S":
            raise ValueError("P promotion must reference S")
        if self.run.stage == "T" and promotion is not None and promotion.predecessor_stage != "P":
            raise ValueError("T promotion must reference P")
        if self.runtime.precision != self.model.precision:
            raise ValueError("runtime and model precision must match")
        if self.model.precision != "fp16" or self.model.attention != "sdpa":
            raise ValueError("R1 requires fp16 and sdpa")
        if self.model.context != self.tokenizer.context or self.packing.sequence_length != self.tokenizer.context:
            raise ValueError("model, tokenizer, and packing context must match")
        if self.model.hidden % self.model.heads or self.model.hidden % self.model.kv_heads:
            raise ValueError("hidden size must be divisible by heads and kv_heads")
        if self.model.heads % self.model.kv_heads:
            raise ValueError("attention heads must be divisible by kv heads")
        if len(self.evaluation.ids) != len(self.evaluation.versions) or len(self.evaluation.ids) != len(self.evaluation.suites):
            raise ValueError("evaluator ids, versions, and suites must align")
        if len(set(self.evaluation.ids)) != len(self.evaluation.ids) or len(set(self.evaluation.suites)) != len(self.evaluation.suites):
            raise ValueError("evaluator ids and suites must be unique")
        if set(self.evaluation.thresholds) != set(self.evaluation.metrics):
            raise ValueError("every metric requires one threshold")
        if set(self.evaluation.directions) != set(self.evaluation.metrics):
            raise ValueError("every metric requires one threshold direction")
        return self


def materialize_context(config: ConfigDocument) -> ConfigDocument:
    """Fill the real context fields from the one-key alias when declared.

    The alias is checked by the validator first, so the fill can only produce
    values the three sections already agree on.
    """
    if config.context is None:
        return config
    return config.model_copy(deep=True, update={
        "context": None,
        "model": config.model.model_copy(update={"context": int(config.context)}),
        "tokenizer": config.tokenizer.model_copy(update={"context": int(config.context)}),
        "packing": config.packing.model_copy(
            update={"sequence_length": int(config.context)}
        ),
    })


def validate_materialized(config: ConfigDocument) -> None:
    """Fail closed when the config is still a template-driven placeholder."""

    def contains_placeholder(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_placeholder(key) or contains_placeholder(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains_placeholder(item) for item in value)
        return value == "replace_me" or (isinstance(value, str) and value.startswith("replace_"))

    if contains_placeholder(config.model_dump()):
        raise ValueError("config is not materialized: replace_me values must be materialized before validation")
    if not config.data.sources:
        raise ValueError("config is not materialized: at least one source is required")
    if not config.evaluation.thresholds:
        raise ValueError("config is not materialized: evaluator thresholds are required")
