

def test_progressive_average_loads_into_a_live_model() -> None:
    """Progressive mode: live params become the mean of the last checkpoints."""
    import torch

    import sys
    sys.path.insert(0, "pipeline_v2")
    from slm_train.orchestration.handlers import _progressive_average
    from slm_train.training.checkpoint import checkpoint_dir
    import io
    from slm_train.artifacts.markers import write_completion_marker
    import hashlib
    from pathlib import Path

    root = Path("scratch/progressive_avg_fixture")
    if root.exists():
        import shutil
        shutil.rmtree(root)
    for step, val in ((1, 1.0), (2, 5.0)):
        directory = checkpoint_dir(root, step)
        (directory / "state").mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        torch.save({"w": torch.tensor([val])}, buffer)
        payload = buffer.getvalue()
        (directory / "state" / "model.pt").write_bytes(payload)
        (directory / "manifest.json").write_text(
            '{"manifest": {"state/model.pt": "' + hashlib.sha256(payload).hexdigest() + '"}}',
            encoding="utf-8")
        write_completion_marker(directory, {"state/model.pt": payload}, "manifest.json")

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([0.0]))

    model = FakeModel()
    _progressive_average(model, root, 2, None, 2)
    assert torch.equal(model.w.detach(), torch.tensor([3.0]))
