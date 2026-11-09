"""Stage context, outcome, and the default status writer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from slm_train.artifacts.atomic import atomic_write_json
from slm_train.contracts.base import ValidationLabel
from slm_train.orchestration.graph import StageGraph, StageSpec

VALID_LABELS = set(ValidationLabel.__args__)
DECIDED = frozenset({"PASS", "FAIL"})


@dataclass(frozen=True)
class StageOutcome:
    """What one stage concluded. A non-PASS outcome always carries a reason."""

    label: ValidationLabel
    reason: str = ""
    detail: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError("invalid validation label: " + repr(self.label))
        if self.label != "PASS" and not self.reason.strip():
            raise ValueError("a " + self.label + " outcome must state a reason")


@dataclass(frozen=True)
class StageContext:
    """What a stage handler is given: its spec, the graph, and its run root.

    A handler receives the resolved configuration and the repository root in
    addition to its own directory. Without the config a handler could only ever
    re-read what an earlier stage wrote, which would make every decision
    indirect. The config is the same immutable object the validate stage
    published, so there is exactly one source of truth.
    """

    spec: StageSpec
    graph: StageGraph
    run_root: Path
    config: object = None
    repo_root: Optional[Path] = None

    def output_path(self, relative: str) -> Path:
        """Resolve a declared output under this stage directory only."""
        if relative not in self.spec.outputs:
            raise ValueError(
                "stage " + self.spec.name + " did not declare the output " + relative
            )
        if relative.startswith("/") or ".." in relative.replace("\\", "/").split("/"):
            raise ValueError("stage outputs are repository-relative paths")
        return self.run_root / self.spec.name / relative

    def stage_root(self) -> Path:
        return Path(self.run_root) / self.spec.name

    def certifiable_outputs(self) -> "tuple[str, ...]":
        """The declared outputs a completion marker is expected to certify.

        Two cases are excluded, and both are structural rather than convenient:

        * the marker itself, which is written AFTER the hashes it certifies and
          so can never contain its own hash; and
        * the certified manifest, but ONLY when the marker's payload map does not
          already contain it. A manifest that is also a payload (most stages)
          stays in the set. A manifest that is not (publish_final, where the run
          marker certifies the staged final files instead) is already covered,
          because verify_completion_marker re-reads the manifest and compares its
          bytes against the marker's manifest_hash.

        Declared outputs are still checked for EXISTENCE by the engine either
        way. Only the payload certification is narrowed, and never below the
        coverage the marker actually provides.
        """
        from slm_train.artifacts.markers import MARKER_NAME

        certified: "dict[str, str]" = {}
        manifest_name = ""
        marker_path = self.stage_root() / MARKER_NAME
        if marker_path.is_file():
            try:
                document = json.loads(marker_path.read_text(encoding="utf-8"))
                if isinstance(document, dict):
                    name = document.get("manifest")
                    manifest_name = name if isinstance(name, str) else ""
                    hashes = document.get("payload_hashes")
                    if isinstance(hashes, dict):
                        certified = hashes
            except (OSError, ValueError):
                manifest_name = ""
        return tuple(
            relative
            for relative in self.spec.outputs
            if relative != MARKER_NAME
            and (relative != manifest_name or relative in certified)
        )

    def already_complete(self) -> bool:
        """True when this exact stage already finished and its evidence verifies.

        An exactly repeated run must not republish create-once artifacts, and
        making every handler idempotent by hand would be a per-stage chance to
        get it subtly wrong. So the question is asked once, here, and answered
        by re-verifying the stage's own marker against the bytes on disk.

        A marker that is absent, incomplete, or does not verify is NOT complete,
        so a partial or tampered stage is never silently skipped.
        """
        from slm_train.artifacts.markers import verify_completion_marker

        if not (self.stage_root() / "COMPLETE.json").is_file():
            return False
        try:
            expected = self.certifiable_outputs()
            if not expected:
                # publish_final declares exactly the two files that cannot be
                # certified (the marker and the manifest it binds), so the
                # derived set is empty. The run marker still certifies the
                # staged final files, and verifying those is the strongest check
                # available: every certified payload is re-hashed from disk.
                expected = self._certified_payload_names()
                if not expected:
                    return False
            return verify_completion_marker(self.stage_root(), expected) is True
        except (OSError, ValueError):
            return False

    def _certified_payload_names(self) -> "tuple[str, ...]":
        """The payload names the stage marker actually certifies."""
        from slm_train.artifacts.markers import MARKER_NAME

        marker_path = self.stage_root() / MARKER_NAME
        if not marker_path.is_file():
            return ()
        try:
            document = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ()
        if not isinstance(document, dict):
            return ()
        hashes = document.get("payload_hashes")
        if not isinstance(hashes, dict):
            return ()
        return tuple(sorted(str(name) for name in hashes))

    def published(self, stage: str, relative: str) -> Path:
        """The path of an artifact an EARLIER stage published.

        Reading forwards is refused. A stage that could read its own output
        would be able to satisfy its own required-input check, and a stage that
        could read a LATER stage's output would be depending on work that has
        not happened yet.
        """
        if stage not in self.graph.order:
            raise ValueError("stage is not in this graph: " + str(stage))
        if self.graph.index_of(stage) >= self.graph.index_of(self.spec.name):
            raise ValueError(
                "stage " + self.spec.name + " may not read from " + stage
            )
        return Path(self.run_root) / stage / relative

    def read_published_json(self, stage: str, relative: str) -> dict:
        """Read one earlier stage's JSON output, or say plainly that it is absent."""
        path = self.published(stage, relative)
        if not path.is_file():
            raise FileNotFoundError(
                "stage " + self.spec.name + " expected " + stage + "/" + relative
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(stage + "/" + relative + " is not a JSON object")
        return document

    def output_bytes(self) -> dict:
        """Every declared output, read back from disk, for the marker to bind.

        A marker written from memory would attest to bytes that were never
        stored. Every hash here is computed from the file on disk.
        """
        # On the WRITE side every declared output is bound, including the
        # certified manifest. certifiable_outputs() narrows only the VERIFY
        # side, and only for a manifest the marker structurally cannot carry.
        payloads = {}
        for relative in self.spec.outputs:
            path = self.output_path(relative)
            if not path.is_file():
                raise FileNotFoundError(
                    "stage " + self.spec.name + " did not produce " + relative
                )
            payloads[relative] = path.read_bytes()
        return payloads

    def write_stage_marker(
        self,
        intended: Optional[dict] = None,
        extra_evidence: Optional[dict] = None,
    ) -> Path:
        """Write this stage's declared completion marker, binding its outputs.

        Every stage writes one. A stage that completes without a marker is a
        stage whose work cannot be proven, which is exactly what the marker
        exists to prevent.

        `intended` maps an output to the exact bytes the handler meant to
        publish. Supplying it is what makes the marker a real check: the marker
        re-reads the file from disk and compares against those bytes, so a
        truncated, partially written, or substituted output cannot be marked
        complete. Omitting it falls back to hashing what is on disk, which still
        binds the manifest bytes but only proves the file did not change between
        the read and the marker write.
        """
        from slm_train.artifacts.markers import write_completion_marker
        from slm_train.contracts.hash import sha256_bytes

        payloads = self.output_bytes()
        root = self.stage_root()
        for relative, blob in (extra_evidence or {}).items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    "evidence " + relative + " is not on disk under " + str(root)
                )
            payloads[relative] = path.read_bytes()

        source = intended if intended is not None else payloads
        certified = {name: sha256_bytes(source[name]) for name in payloads}
        return write_completion_marker(
            root, payloads, self.json_manifest(), certified=certified
        )

    def json_manifest(self) -> str:
        """The declared output the marker is built on: the first JSON one.

        Taking outputs[0] would depend on declaration order, and the validate
        stage declares its YAML first. A marker cannot bind a YAML file, so the
        choice is made by content and fails closed when there is no JSON output
        rather than parsing whatever happens to be first.
        """
        for relative in self.spec.outputs:
            if relative.endswith(".json"):
                return relative
        raise ValueError(
            "stage " + self.spec.name + " declares no JSON output to build a marker on"
        )


@dataclass(frozen=True)
class RunResult:
    """The engine outcome, recorded whether the run passed or not."""

    status: ValidationLabel
    completed_stages: tuple[str, ...]
    failed_stage: Optional[str] = None
    reason: str = ""
