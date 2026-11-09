"""The R1 stage graph.

The graph is static data: an exact ordered sequence of stages, each declaring
the artifacts it consumes, the artifacts it produces, and the completion marker
that proves it finished. A graph is validated at construction, so an impossible
graph (a stage requiring an artifact no earlier stage produces, a duplicate stage,
an empty order) can never reach the engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from slm_train.orchestration.errors import GraphError

R1_ORDER: tuple[str, ...] = (
    "validate",
    "bootstrap",
    "gpu_preflight",
    "prepare",
    "tokenize",
    "dataset_validation",
    "model_validation",
    "evaluator_validation",
    "resolve_final_identity",
    "train",
    "internal_evaluation",
    "external_gates",
    "publish_final",
)

REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {
    "validate": (),
    "bootstrap": ("config/resolved.yaml", "config/resolved.json"),
    "gpu_preflight": ("attempt/bootstrap.json",),
    "prepare": ("attempt/preflight.json",),
    "tokenize": ("prepared_generation/COMPLETE.json", "manifests/data.json"),
    "dataset_validation": (
        "tokenized_generation/COMPLETE.json",
        "manifests/tokenizer.json",
    ),
    "model_validation": ("manifests/tokenized_data.json",),
    "evaluator_validation": ("config/resolved.json", "manifests/model.json"),
    "resolve_final_identity": (
        "manifests/data.json",
        "manifests/tokenizer.json",
        "manifests/tokenized_data.json",
        "manifests/model.json",
        "manifests/evaluator.json",
    ),
    "train": ("manifests/run_identity.json",),
    "internal_evaluation": ("checkpoints/COMPLETE.json",),
    "external_gates": ("evaluations/COMPLETE.json",),
    "publish_final": ("reports/COMPLETE.json",),
}

OUTPUTS: dict[str, tuple[str, ...]] = {
    "validate": ("config/resolved.yaml", "config/resolved.json"),
    "bootstrap": ("attempt/bootstrap.json",),
    "gpu_preflight": ("attempt/preflight.json",),
    "prepare": ("prepared_generation/COMPLETE.json", "manifests/data.json"),
    "tokenize": ("tokenized_generation/COMPLETE.json", "manifests/tokenizer.json"),
    "dataset_validation": ("manifests/tokenized_data.json",),
    "model_validation": ("manifests/model.json",),
    "evaluator_validation": ("manifests/evaluator.json",),
    "resolve_final_identity": ("manifests/run_identity.json",),
        "train": (
            "checkpoints/COMPLETE.json",
            "logs/resume_summary.json",
            # E-005 native monitor artifacts: always-written summaries (a run
            # that declared no monitor still publishes the regime summary).
            "logs/monitor_report.jsonl",
            "logs/monitor_summary.json",
        ),
    "internal_evaluation": ("evaluations/COMPLETE.json",),
    "external_gates": ("reports/COMPLETE.json",),
    "publish_final": ("manifests/final_publication.json", "COMPLETE.json"),
}


@dataclass(frozen=True)
class StageSpec:
    """One stage: what it needs, what it makes, and how it is marked complete."""

    name: str
    requires: tuple[str, ...]
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    completion_marker: str

    def allows_next(self, name: str) -> bool:
        """Only the immediately following stage may run next."""
        return name in self.requires



@dataclass(frozen=True)
class StageGraph:
    """An immutable, validated stage sequence."""

    order: tuple[str, ...] = R1_ORDER

    def __post_init__(self) -> None:
        if not isinstance(self.order, tuple) or not self.order:
            raise GraphError("a stage graph needs a non-empty order")
        if len(set(self.order)) != len(self.order):
            raise GraphError("a stage graph cannot repeat a stage")
        for name in self.order:
            if name not in REQUIRED_INPUTS or name not in OUTPUTS:
                raise GraphError("stage is not declared in the R1 contract: " + str(name))
        self._validate_sequence()
        object.__setattr__(self, "_specs", self._build_specs())

    def _validate_sequence(self) -> None:
        """Every required input must come from an earlier stage of THIS graph."""
        produced: set[str] = set()
        for name in self.order:
            missing = set(REQUIRED_INPUTS[name]) - produced
            if missing:
                raise GraphError(
                    "stage " + name + " requires unproduced inputs: "
                    + ", ".join(sorted(missing)),
                )
            produced |= set(OUTPUTS[name])

    def _build_specs(self) -> tuple[StageSpec, ...]:
        return tuple(
            StageSpec(
                name=name,
                requires=() if index == 0 else (self.order[index - 1],),
                required_inputs=REQUIRED_INPUTS[name],
                outputs=OUTPUTS[name],
                completion_marker=name + ".COMPLETE.json",
            )
            for index, name in enumerate(self.order)
        )

    @classmethod
    def r1(cls) -> "StageGraph":
        """The exact R1 sequence. A fresh object every call, never a shared one."""
        return cls()

    def specs(self) -> tuple[StageSpec, ...]:
        return self._specs

    def spec_by_name(self, name: str) -> StageSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise GraphError("stage is not in this graph: " + str(name))

    def index_of(self, name: str) -> int:
        if name not in self.order:
            raise GraphError("stage is not in this graph: " + str(name))
        return self.order.index(name)

