"""The stage engine: verify inputs, dispatch, verify outputs, never fall back.

For every stage the engine checks that each declared required input exists under
an EARLIER stage directory, dispatches the handler, then checks that every
declared output actually appeared. A stage that does not declare all of its
inputs up front fails closed before the handler runs.

On any exception the engine writes the status and diagnostic evidence and
propagates. It never skips ahead, never retries a stage automatically, and never
substitutes a fallback stage for the one that failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from slm_train.orchestration.errors import (
    MissingStageInput,
    StageBlocked,
    StageFailed,
)
from slm_train.orchestration.graph import StageGraph, StageSpec
from slm_train.orchestration.stages import RunResult, StageContext, StageOutcome
from slm_train.orchestration.status import write_status

Handler = Callable[[StageContext], StageOutcome]


@dataclass
class Engine:
    """Runs a stage graph in order, failing closed at the first problem."""

    graph: StageGraph
    handlers: Mapping[str, Handler]
    run_root: Path
    #: the resolved configuration every handler reads; one object, one truth
    config: object = None
    #: the repository root, so a handler never has to guess the current directory
    repo_root: Path = field(default_factory=Path.cwd)

    def _missing_inputs(self, spec: StageSpec) -> list[str]:
        """Required inputs not present under any earlier stage directory."""
        earlier_stages = self.graph.order[: self.graph.index_of(spec.name)]
        missing: list[str] = []
        for relative in spec.required_inputs:
            found = any(
                (Path(self.run_root) / earlier / relative).exists()
                for earlier in earlier_stages
            )
            if not found:
                missing.append(relative)
        return missing

    def _missing_outputs(self, spec: StageSpec) -> list[str]:
        return [
            relative
            for relative in spec.outputs
            if not (Path(self.run_root) / spec.name / relative).is_file()
        ]

    def _record(self, result: RunResult) -> None:
        """Status is best effort: it must never mask the original failure."""
        try:
            write_status(Path(self.run_root), result)
        except OSError:
            pass

    def _decided(
        self,
        completed: list[str],
        spec: StageSpec,
        status: str,
        reason: str,
    ) -> RunResult:
        """Record the label the stage actually reached, never a flattened one."""
        result = RunResult(
            status=status,
            completed_stages=tuple(completed),
            failed_stage=spec.name,
            reason=reason,
        )
        self._record(result)
        return result

    def _blocked(self, completed: list[str], spec: StageSpec, reason: str) -> RunResult:
        return self._decided(completed, spec, "BLOCKED", reason)

    def _failed(self, completed: list[str], spec: StageSpec, reason: str) -> RunResult:
        return self._decided(completed, spec, "FAIL", reason)


    def run(self) -> RunResult:
        """Run every stage in order. Returns PASS, or raises after recording."""
        completed: list[str] = []
        for spec in self.graph.specs():
            if spec.name not in self.handlers:
                self._blocked(
                    completed, spec, "no handler is registered for stage " + spec.name,
                )
                raise MissingStageInput("no handler is registered for " + spec.name)
            missing = self._missing_inputs(spec)
            if missing:
                listed = ", ".join(sorted(missing))
                self._blocked(completed, spec, "missing required inputs: " + listed)
                raise MissingStageInput(
                    "stage " + spec.name + " is missing required inputs: " + listed,
                )
            context = StageContext(
                spec=spec,
                graph=self.graph,
                run_root=Path(self.run_root),
                config=self.config,
                repo_root=Path(self.repo_root),
            )
            try:
                outcome = self.handlers[spec.name](context)
            except StageBlocked as exc:
                self._blocked(completed, spec, str(exc))
                raise
            except Exception as exc:
                detail = type(exc).__name__ + ": " + str(exc)
                self._failed(completed, spec, detail)
                raise StageFailed(
                    "stage " + spec.name + " failed: " + detail
                ) from exc
            if not isinstance(outcome, StageOutcome):
                self._failed(
                    completed,
                    spec,
                    "handler returned " + type(outcome).__name__ + " instead of StageOutcome",
                )
                raise StageFailed("stage " + spec.name + " returned no StageOutcome")
            if outcome.label == "BLOCKED":
                self._blocked(completed, spec, outcome.reason)
                raise StageBlocked(
                    "stage " + spec.name + " is blocked: " + outcome.reason,
                )
            if outcome.label != "PASS":
                self._decided(completed, spec, outcome.label, outcome.reason)
                raise StageFailed(
                    "stage " + spec.name + " returned " + outcome.label,
                )
            absent = self._missing_outputs(spec)
            if absent:
                listed = ", ".join(sorted(absent))
                self._failed(completed, spec, "stage did not produce: " + listed)
                raise StageFailed(
                    "stage " + spec.name + " did not produce: " + listed,
                )
            completed.append(spec.name)
        result = RunResult(status="PASS", completed_stages=tuple(completed))
        self._record(result)
        return result

