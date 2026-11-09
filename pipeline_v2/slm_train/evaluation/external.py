"""External gates: claims the run made, re-derived from the published bytes.

Each gate here deliberately does NOT trust the run's own bookkeeping. The
internal evaluation reads a record the run wrote; these read the files that
record describes. That is the whole point of an external gate, and the reason
the previous implementation was worse than no gate at all: it copied the
internal suite reports and relabelled them as external verification, which
manufactured assurance out of the very thing it was meant to check.

A gate returns (name, status, detail). A gate that cannot run is BLOCKED,
never PASS, and never silently skipped.
"""
from __future__ import annotations

__all__ = ["EXTERNAL_GATES", "GateResult", "run_external_gates"]


class GateResult:
    """One gate outcome: a name, a validation label, and what it actually did."""

    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: dict) -> None:
        self.name = name
        self.status = status
        self.detail = detail

    def as_dict(self) -> dict:
        return {"gate": self.name, "status": self.status, "detail": self.detail}


def _artifact_integrity(context, document) -> GateResult:
    """Re-hash the selected checkpoint against its own manifest.

    The stage order puts external_gates BEFORE publish_final, so this verifies
    the checkpoint the run is about to publish rather than a publication that
    does not exist yet. A run that certified bytes it no longer holds, or that
    never wrote them at all, fails here.
    """
    import json
    from pathlib import Path

    from slm_train.contracts.hash import sha256_file

    train_record = context.read_published_json("train", "checkpoints/COMPLETE.json")
    checkpoint = str(train_record.get("checkpoint", ""))
    if not checkpoint:
        return GateResult(
            "artifact_integrity", "BLOCKED",
            {"reason": "the train record names no checkpoint to verify"},
        )
    root = Path(context.run_root) / "train" / "checkpoints" / checkpoint
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return GateResult(
            "artifact_integrity", "FAIL",
            {"reason": "the selected checkpoint has no manifest", "checkpoint": checkpoint},
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("files", []))
    if not entries:
        return GateResult(
            "artifact_integrity", "FAIL",
            {"reason": "the checkpoint manifest lists no payloads", "checkpoint": checkpoint},
        )
    checked = 0
    problems = []
    roles = set()
    for entry in entries:
        name = str(entry.get("path", ""))
        claimed = str(entry.get("sha256", ""))
        claimed_size = entry.get("size_bytes")
        roles.add(str(entry.get("role", "")))
        # The manifest records the payload name; the bytes live under state/.
        target = root / "state" / name
        if not target.is_file():
            target = root / name
        if not target.is_file():
            problems.append(name + " (absent)")
            continue
        checked += 1
        if claimed and sha256_file(target) != claimed:
            problems.append(name + " (digest differs)")
        elif claimed_size is not None and int(target.stat().st_size) != int(claimed_size):
            problems.append(name + " (size differs)")
    if "model" not in roles:
        problems.append("no model payload is declared")
    if problems:
        return GateResult(
            "artifact_integrity", "FAIL",
            {"checkpoint": checkpoint, "checked": checked,
             "problems": sorted(problems)},
        )
    return GateResult(
        "artifact_integrity", "PASS",
        {"checkpoint": checkpoint, "checked": checked, "roles": sorted(roles)},
    )


def _retention_policy(context, document) -> GateResult:
    """Re-derive the retention decision from the checkpoints on disk."""
    from pathlib import Path

    stage_root = Path(context.run_root) / "train" / "checkpoints"
    present = sorted(
        path.name for path in stage_root.iterdir()
        if path.is_dir() and path.name.startswith("step-")
    )
    keep_last = int(document.train.keep_last_checkpoints)
    keep_best = int(document.train.keep_best_checkpoints)
    expected_max = keep_last + keep_best
    scores_path = stage_root / "scores.json"
    if not scores_path.is_file():
        return GateResult(
            "retention_policy", "BLOCKED",
            {"reason": "no scores.json records which checkpoint was best"},
        )
    import json

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    val = scores.get("val_loss", {})
    if not val:
        return GateResult(
            "retention_policy", "BLOCKED",
            {"reason": "no held-out loss was ever recorded"},
        )
    best = min(val, key=lambda key: val[key])
    best_dir = "step-" + str(best).zfill(8)
    if best_dir not in present:
        return GateResult(
            "retention_policy", "FAIL",
            {"reason": "the best checkpoint " + best_dir + " is not on disk",
             "present": present},
        )
    if len(present) > expected_max:
        return GateResult(
            "retention_policy", "FAIL",
            {"reason": "retention kept " + str(len(present))
             + " checkpoints, more than the declared maximum of "
             + str(expected_max),
             "present": present},
        )
    return GateResult(
        "retention_policy", "PASS",
        {"present": present, "best": best_dir, "maximum": expected_max},
    )


def _checkpoint_reload(context, document) -> GateResult:
    """Reload the published weights and recompute the held-out loss.

    This is the gate that catches a checkpoint which hashes correctly but does
    not contain the weights its own record claims.
    """
    import torch

    from slm_train.models.causal_lm import build_causal_lm
    from slm_train.orchestration.handlers import _held_out_loss, _load_trained_weights, _split_blocks

    train_record = context.read_published_json(
        "train", "checkpoints/COMPLETE.json"
    )
    # The BEST checkpoint and the BEST loss are a pair. Comparing the best
    # recorded loss against the reloaded FINAL checkpoint measures two
    # different models and fails a run that is actually consistent.
    selected = str(train_record.get("best_checkpoint", ""))
    recorded = train_record.get("best_val_loss")
    if recorded is None:
        return GateResult(
            "checkpoint_reload", "BLOCKED",
            {"reason": "the train record names no best checkpoint to reload"},
        )
    tokenized = context.read_published_json(
        "dataset_validation", "manifests/tokenized_data.json"
    )
    tokenizer_manifest = context.read_published_json(
        "tokenize", "manifests/tokenizer.json"
    )
    _unused, held_out, _train_masks_unused, held_masks_ext = _split_blocks(
        context, tokenized, float(document.data.validation_fraction)
    )
    if not held_out:
        return GateResult(
            "checkpoint_reload", "BLOCKED",
            {"reason": "the run published no held-out block to re-measure"},
        )
    device = str(document.runtime.device)
    # The device question is asked through gpu_probe, the only module allowed to
    # to ask the runtime about an accelerator. Asking it here broke the device
    # boundary that the release-boundary contract test enforces by text scan.
    from slm_train.resources.gpu_probe import device_available

    if not device_available(device):
        return GateResult(
            "checkpoint_reload", "BLOCKED",
            {"reason": "the run declared " + device + " but no device is present"},
        )

    model = build_causal_lm(
        model_config=document.model,
        capabilities=list(document.capabilities),
        data_context=int(document.packing.sequence_length),
        vocab_size=int(tokenizer_manifest["vocabulary_size"]),
        seed=int(document.run.seed),
    )
    if _load_trained_weights(context, model, selected) is None:
        return GateResult(
            "checkpoint_reload", "FAIL",
            {"reason": "the published checkpoint carries no loadable model state"},
        )
    # The reloaded model must live where the blocks do. Building it on cpu and
    # then feeding it device blocks is a mismatch, not a measurement.
    model = model.to(torch.float32).to(device)
    compute = torch.float32 if device == "cpu" else torch.float16
    measured = _held_out_loss(
        model, held_out, device, compute, torch.device(device).type,
        device != "cpu",
        masks=held_masks_ext,
    )
    drift = abs(float(measured) - float(recorded))
    tolerance = 0.05 * abs(float(recorded)) + 1e-6
    if drift > tolerance:
        return GateResult(
            "checkpoint_reload", "FAIL",
            {"reason": "the reloaded checkpoint does not reproduce its recorded loss",
             "checkpoint": selected,
             "recorded": float(recorded), "remeasured": float(measured),
             "drift": drift, "tolerance": tolerance},
        )
    return GateResult(
        "checkpoint_reload", "PASS",
        {"checkpoint": selected,
         "recorded": float(recorded), "remeasured": float(measured),
         "drift": drift, "tolerance": tolerance},
    )


EXTERNAL_GATES = {
    "artifact_integrity": _artifact_integrity,
    "checkpoint_reload": _checkpoint_reload,
    "retention_policy": _retention_policy,
}


def run_external_gates(context, document) -> tuple:
    """Every REGISTERED gate, in a fixed order, each independently labelled."""
    from slm_train.evaluation.suites import EXTERNAL_GATE_NAMES

    results = []
    for name in EXTERNAL_GATE_NAMES:
        gate = EXTERNAL_GATES.get(name)
        if gate is None:
            results.append(GateResult(
                name, "BLOCKED", {"reason": "gate is registered but not implemented"}
            ))
            continue
        try:
            results.append(gate(context, document))
        except Exception as error:
            results.append(GateResult(
                name, "BLOCKED",
                {"reason": "the gate could not run: " + type(error).__name__
                 + ": " + str(error)},
            ))
    return tuple(results)
