"""Post-hoc weight averaging over retained checkpoints (R2 adoption).

The U-2 paired-arm evidence (research/ui_u2_full_train_plan.md, 2026-09-27)
showed averaging the model weights of the last surviving checkpoints beats
the single best checkpoint on pass@k with no extra training. This module
publishes the average as a first-class artifact: the model-state average of
the last N COMPLETE checkpoints, a manifest binding every source checkpoint
by step and content hash, and a completion marker, so a downstream stage can
reload it and VERIFY what it loaded.

The average is over model weights only; optimizer/scheduler roles cannot be
averaged meaningfully and are intentionally absent.
"""
from __future__ import annotations

import io
from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.artifacts.manifest import write_manifest
from slm_train.artifacts.markers import MARKER_NAME, write_completion_marker
from slm_train.contracts.hash import canonical_sha256, sha256_bytes
from slm_train.artifacts.markers import verify_completion_marker
from slm_train.training.checkpoint import is_complete


AVERAGING_MARKER = "averaged"


class AveragingError(RuntimeError):
    """Raised when an average could not be published from complete sources."""


def average_steps(run_root: Path, n: int) -> "list[int]":
    """The last n complete checkpoint numbers, ascending, or a shorter list."""
    root = Path(run_root) / "checkpoints"
    numbers = sorted(int(directory.name.split("-")[-1])
                     for directory in root.iterdir()
                     if directory.is_dir() and directory.name.startswith("step-")
                     and is_complete(run_root, int(directory.name.split("-")[-1])))
    return numbers[-n:]


def load_state(callable_loader, run_root: Path, step: int) -> dict:
    """Load one checkpoint's model state through the caller's loader.

    The loader indirection keeps torch out of this module's import graph so
    the CPU-only contract tests do not need the device stack.
    """
    path = Path(run_root) / "checkpoints" / ("step-" + str(step).zfill(8)) / "state" / "model.pt"
    if not path.is_file():
        raise AveragingError("complete checkpoint lacks a model state file: step " + str(step))
    return callable_loader(path)


def average_states(states: "list[dict]") -> dict:
    """Floatwise mean over matching state keys; buffers use the last value.

    A missing or non-tensor key is a real failure, not a silent skip: the
    published average must be exactly the weights the sources claim.
    """
    if not states:
        raise AveragingError("no states to average")
    keys = set(states[0])
    for state in states[1:]:
        if set(state) != keys:
            raise AveragingError("checkpoint state keys differ; refusing to average")
    averaged: dict = {}
    for key in sorted(states[0]):
        values = []
        for state in states:
            value = state[key]
            if not hasattr(value, "dtype"):
                raise AveragingError("non-tensor state entry: " + str(key))
            values.append(value)
        import torch

        if values[0].dtype.is_floating_point or values[0].dtype.is_complex:
            averaged[key] = torch.stack(values).mean(dim=0).to(values[0].dtype)
        else:
            # Integer and boolean buffers (norm counters, cache slices) are
            # copied rather than averaged: mean() on a bool tensor is a
            # runtime error, and a copied last value is the honest choice.
            averaged[key] = values[-1]
    return averaged


def torch_float_dtype(dtype):
    import torch

    return torch.float32 if dtype.is_floating_point else dtype


def publish_weight_average(
    run_root: Path,
    identity: dict,
    n: int,
    load_one,
) -> dict:
    """Average the last n complete checkpoints and publish the artifact.

    Returns the record that stage COMPLETE documents embed. Any failure to
    complete the publish raises, and the run must FAIL (labeled), because a
    declared averaging run whose average is absent is not evidence.
    """
    import torch

    out = Path(run_root) / "checkpoints_avg"
    existing = out / "weight-average.pt"
    if existing.is_file() and verify_completion_marker(out, ["weight-average.pt"]) is True:
        # The publish path is immutable: an existing verified average is the
        # artifact of record, whatever produced it. A rerun REUSES it instead
        # of failing the whole stage on the second publication.
        from slm_train.artifacts.manifest import load_manifest

        prior = load_manifest(out, "manifest.json")
        if isinstance(prior, dict) and "record" in prior:
            return prior["record"]
    steps = average_steps(run_root, int(n))
    if len(steps) < 1:
        raise AveragingError("no complete checkpoint to average")
    for step in steps:
        if not ((Path(run_root) / "checkpoints" /
                 ("step-" + str(step).zfill(8)) / "state" / "model.pt").is_file()):
            raise AveragingError(
                "complete checkpoint lacks a model state file: step " + str(step))
    states = [load_one(run_root, step) for step in steps]
    averaged = average_states(states)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "weight-average.pt"
    buffer = io.BytesIO()
    torch.save(averaged, buffer)
    payload = buffer.getvalue()
    atomic_write_bytes(path, payload)
    sources = [
        {
            "step": int(step),
            "model_sha256": sha256_bytes(
                (Path(run_root) / "checkpoints" / ("step-" + str(step).zfill(8))
                 / "state" / "model.pt").read_bytes()
            ),
        }
        for step in steps
    ]
    record = {
        "kind": "weight-average",
        "schema_version": 1,
        "n": len(steps),
        "steps": steps,
        "sources": sources,
        "path": "checkpoints_avg/weight-average.pt",
        "sha256": sha256_bytes(payload),
        "content_hash": canonical_sha256({
            "identity": identity,
            "sources": sources,
            "mode": "final_n",
        }),
        "run_id": str(identity.get("run_id", "")),
    }
    write_manifest(out, "manifest.json", {
        "record": record,
        "manifest": {
            "weight-average.pt": record["sha256"],
        },
    })
    write_completion_marker(out, {"weight-average.pt": payload}, "manifest.json")
    return record
