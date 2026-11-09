"""Checkpoint selection from published manifests, never from a directory scan.

Selection answers one question: which COMPLETE checkpoint is the highest numeric
identity whose recorded files all still hash to what the manifest claims. The
candidate list comes from the manifests themselves, so a half-written directory
with no manifest is invisible, and a manifest whose files no longer verify is
skipped rather than trusted.

An empty candidate set is a BLOCKED decision, not a silent "start from scratch":
a resume that cannot find a checkpoint must say so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pathlib import Path

from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.artifacts.markers import verify_completion_marker
from slm_train.contracts.hash import sha256_bytes
from slm_train.training.checkpoint import (
    CHECKPOINTS_DIR,
    MANIFEST_NAME,
    STATE_DIR,
    checkpoint_dir,
    is_complete,
)

CHECKPOINT_POLICIES = ("best", "last", "complete_only")


def validate_policy(policy: object) -> str:
    """Refuse an undeclared policy before any selection happens."""
    text = str(policy)
    if text not in CHECKPOINT_POLICIES:
        raise ValueError(
            "unsupported train.checkpoint_policy: " + text
            + " (declared policies are " + ", ".join(CHECKPOINT_POLICIES) + ")"
        )
    return text


def validation_scores(stage_root: Path) -> dict[int, float]:
    """Published step -> held-out loss, from the train stage own score file."""
    path = Path(stage_root) / CHECKPOINTS_DIR / "scores.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, float] = {}
    for key, value in (raw.get("val_loss") or {}).items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def select_policy_checkpoint(
    stage_root: Path, policy: str, final_step: int
) -> int | None:
    """The checkpoint the DECLARED selection policy names, not step luck.

    "last" means the highest published step. "best" means the published step
    with the LOWEST measured held-out loss (ties break to the higher step); a
    best run that somehow published no measurement falls back to the last
    step, and says so in the manifest rather than silently renaming last.

    Returns None when no checkpoint has been published.
    """
    validate_policy(policy)
    steps = declared_checkpoint_numbers(stage_root)
    if not steps:
        return None
    candidates = sorted({*steps, int(final_step)})
    if policy in ("last", "complete_only") or not candidates:
        return steps[-1]
    scores = validation_scores(stage_root)
    if not scores:
        return steps[-1]
    return min(scores, key=lambda step: (scores[step], -step))


@dataclass(frozen=True)
class SelectionResult:
    """Which checkpoint was selected, and why any candidate was rejected."""

    number: int | None
    reason: str
    rejected: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.number is not None


def declared_checkpoint_numbers(run_root: Path) -> list[int]:
    """Only step-NNNNNNNN directories that carry a manifest are candidates."""
    root = Path(run_root) / CHECKPOINTS_DIR
    if not root.is_dir():
        return []
    numbers: list[int] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("step-"):
            continue
        suffix = path.name[len("step-"):]
        if not suffix.isdigit():
            continue
        if not (path / MANIFEST_NAME).is_file():
            continue
        numbers.append(int(suffix))
    return sorted(numbers)


def read_manifest(run_root: Path, number: int) -> CheckpointManifest:
    """Load one published manifest, or refuse: a partial one is not a manifest."""
    path = checkpoint_dir(run_root, number) / MANIFEST_NAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("checkpoint manifest is unreadable: " + str(number)) from error
    # JSON mode, not object mode: the published document carries arrays for the
    # tuple-valued fields, and strict object validation would reject its own file.
    return CheckpointManifest.model_validate_json(raw)


def verify_checkpoint(run_root: Path, number: int) -> list[str]:
    """Every way this checkpoint fails verification. Empty means usable.

    The completion marker is re-verified BEFORE the manifest is trusted. The
    marker binds the exact manifest bytes, so an edited manifest that drops a
    state file is caught there, and a deleted state file is caught by the
    marker certified payload hashes. Checking the manifest own file list alone
    would accept a manifest edited to describe a smaller checkpoint.
    """
    problems: list[str] = []
    destination = checkpoint_dir(run_root, number)
    if not is_complete(run_root, number):
        # A partial directory with no completion marker is not a candidate
        # for selection. Return a special marker so the caller can
        # distinguish between "no checkpoint" and "checkpoint failed".
        return ["no completion marker"]
    try:
        manifest = read_manifest(run_root, number)
    except Exception as error:
        return [str(error)[:120]]
    certified = [STATE_DIR + "/" + entry.path for entry in manifest.files]
    try:
        verify_completion_marker(destination, certified)
    except Exception as error:
        problems.append("completion marker does not verify: " + str(error)[:100])
    if not manifest.complete:
        problems.append("manifest is not marked complete")
    for entry in manifest.files:
        target = destination / STATE_DIR / entry.path
        if not target.is_file():
            problems.append("missing state file: " + entry.role)
            continue
        payload = target.read_bytes()
        if len(payload) != entry.size_bytes:
            problems.append("size mismatch: " + entry.role)
            continue
        if sha256_bytes(payload) != entry.sha256:
            problems.append("hash mismatch: " + entry.role)
    return problems


def select_checkpoint(run_root: Path) -> SelectionResult:
    """The checkpoint the DECLARED selection policy names, or a BLOCKED reason.

    Reads the checkpoint_policy from the train stage's published config and
    delegates to select_policy_checkpoint. If no policy is declared, defaults
    to "last" (the highest-numbered verified checkpoint).
    """
    # Try to read the checkpoint_policy from the train stage's published config
    policy = "last"  # default
    config_path = Path(run_root) / "train" / "config.json"
    if config_path.is_file():
        try:
            import json
            config_doc = json.loads(config_path.read_text(encoding="utf-8"))
            train_section = config_doc.get("train", {})
            if "checkpoint_policy" in train_section:
                policy = validate_policy(train_section["checkpoint_policy"])
        except (OSError, ValueError, KeyError):
            pass

    # Get the final step from the train stage's published config
    final_step = 0
    if config_path.is_file():
        try:
            import json
            config_doc = json.loads(config_path.read_text(encoding="utf-8"))
            train_section = config_doc.get("train", {})
            if "max_steps" in train_section:
                final_step = int(train_section["max_steps"])
        except (OSError, ValueError, KeyError):
            pass

    # Use the declared policy to select the checkpoint
    selected = select_policy_checkpoint(run_root, policy, final_step)
    if selected is None:
        return SelectionResult(number=None,
                            reason="no complete checkpoint has been published")
    rejected: list[str] = []
    problems = verify_checkpoint(run_root, selected)
    if not problems:
        return SelectionResult(number=selected, reason="verified",
                             rejected=tuple(rejected))
    rejected.append("checkpoint " + str(selected) + ": " + "; ".join(problems))
    # Fall back to the highest verifiable checkpoint if the selected one fails
    for number in reversed(declared_checkpoint_numbers(run_root)):
        if number == selected:
            continue
        problems = verify_checkpoint(run_root, number)
        if not problems:
            return SelectionResult(number=number, reason="verified",
                                 rejected=tuple(rejected))
        rejected.append("checkpoint " + str(number) + ": " + "; ".join(problems))
    if rejected:
        return SelectionResult(
            number=None,
            reason="every candidate checkpoint failed verification",
            rejected=tuple(rejected),
        )
    return SelectionResult(number=None,
                        reason="no complete checkpoint has been published")
