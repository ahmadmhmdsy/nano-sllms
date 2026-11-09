"""Weight-average publication tests (R2 adoption)."""
from pathlib import Path

import torch

from slm_train.artifacts.markers import verify_completion_marker
from slm_train.training.averaging import (
    average_states,
    publish_weight_average,
)


def _fake_checkpoint(root: Path, step: int, weights: dict) -> None:
    import io

    from slm_train.artifacts.markers import write_completion_marker
    from slm_train.training.checkpoint import checkpoint_dir

    directory = checkpoint_dir(root, step)
    (directory / "state").mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    payload = buffer.getvalue()
    (directory / "state" / "model.pt").write_bytes(payload)
    # A publish_checkpoint would write its manifest; the average reads only
    # the state file and checks completeness, so a minimal hand-written
    # marker keeps the fixture honest about what a COMPLETE label means.
    manifest = directory / "manifest.json"
    manifest.write_text(
        '{"manifest": {"state/model.pt": "'
        + __import__("hashlib").sha256(payload).hexdigest()
        + '"}}',
        encoding="utf-8",
    )
    write_completion_marker(
        directory, {"state/model.pt": payload}, "manifest.json"
    )


def test_average_states_is_the_floatwise_mean() -> None:
    averaged = average_states([
        {"w": torch.tensor([2.0, 4.0]), "b": torch.tensor([True])},
        {"w": torch.tensor([4.0, 6.0]), "b": torch.tensor([False])},
    ])
    assert torch.equal(averaged["w"], torch.tensor([3.0, 5.0]))
    # Non-floating buffers keep the newest source's value, so normalization
    # flag tensors survive round-trip loads.
    assert torch.equal(averaged["b"], torch.tensor([False]))


def test_publish_weight_average_writes_a_verified_artifact(tmp_path) -> None:
    root = Path(tmp_path)
    _fake_checkpoint(root, 1, {"w": torch.tensor([1.0])})
    _fake_checkpoint(root, 2, {"w": torch.tensor([3.0])})
    record = publish_weight_average(
        root,
        {"run_id": "test-run"},
        2,
        lambda _pin, step: torch.load(
            root / "checkpoints" / ("step-" + str(step).zfill(8))
            / "state" / "model.pt",
            map_location="cpu",
            weights_only=True,
        ),
    )
    assert record["steps"] == [1, 2]
    assert record["n"] == 2
    out_dir = root / "checkpoints_avg"
    assert verify_completion_marker(out_dir, ["weight-average.pt"]) is True
    state = torch.load(out_dir / "weight-average.pt", weights_only=True)
    assert torch.equal(state["w"], torch.tensor([2.0]))
    # Sources bind to the bytes that were averaged.
    recorded = [
        source["model_sha256"]
        for source in record["sources"]
    ]
    import hashlib

    for step, sha in zip((1, 2), recorded):
        blob = (root / "checkpoints" / ("step-" + str(step).zfill(8))
                / "state" / "model.pt").read_bytes()
        assert hashlib.sha256(blob).hexdigest() == sha


def test_publish_weight_average_refuses_a_missing_state(tmp_path) -> None:
    import pytest as _pytest

    from slm_train.training.averaging import AveragingError

    root = Path(tmp_path)
    _fake_checkpoint(root, 1, {"w": torch.tensor([1.0])})
    (root / "checkpoints" / "step-00000001" / "state" / "model.pt").unlink()
    with _pytest.raises(AveragingError):
        publish_weight_average(root, {"run_id": "x"}, 1, lambda _pin, _step: {})
