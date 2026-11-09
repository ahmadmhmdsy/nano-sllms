r"""Regression test for scripts/sft.py --vram-cap-gib (AGENTS 20.3 allocator ceiling).

Dependency-free runner (repo convention), run directly with the venv python:
    & .\.venv\Scripts\python.exe tests\test_sft_vram_cap.py
CPU-only: torch.cuda is monkeypatched, so no GPU allocation ever happens. The
test also asserts the call SITE ordering (cap before the model reaches the GPU)
and that the default (flag unset) is a no-op.
"""
import subprocess
import sys
from pathlib import Path

import torch

from _common import run

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sft.py"

import sft  # noqa: E402  (scripts/ is on sys.path via _common)


class _Props:
    total_memory = 8 * 1024**3  # pretend the device is 8 GiB


class _CudaPatch:
    """Make torch.cuda look like an 8 GiB CUDA device and record cap calls."""

    _names = ("is_available", "current_device", "get_device_properties",
              "set_per_process_memory_fraction")

    def __enter__(self):
        self.calls = []
        self._saved = {name: getattr(torch.cuda, name) for name in self._names}
        torch.cuda.is_available = lambda: True
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_properties = lambda device: _Props()

        def _set(fraction, device=None):
            self.calls.append((fraction, str(device)))

        torch.cuda.set_per_process_memory_fraction = _set
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(torch.cuda, name, value)
        return False


def t_unset_cap_is_a_noop():
    with _CudaPatch() as patch:
        assert sft.apply_vram_cap(0.0) == {}, "0.0 must be a no-op"
        assert sft.apply_vram_cap(None) == {}, "None must be a no-op"
        assert patch.calls == [], f"unset cap still called the allocator: {patch.calls}"


def t_negative_cap_is_a_noop():
    with _CudaPatch() as patch:
        assert sft.apply_vram_cap(-1.0) == {}, "negative must be a no-op"
        assert patch.calls == []


def t_cap_sets_the_fraction_on_the_cuda_device():
    with _CudaPatch() as patch:
        out = sft.apply_vram_cap(2.0)
        assert out == {"vram_cap_gib": 2.0, "vram_cap_fraction": 0.25}, out
        assert len(patch.calls) == 1, patch.calls
        fraction, device = patch.calls[0]
        assert abs(fraction - 0.25) < 1e-12, fraction
        assert "cuda" in device, device


def t_cap_larger_than_the_device_clamps_to_one():
    with _CudaPatch() as patch:
        out = sft.apply_vram_cap(100.0)
        assert out["vram_cap_fraction"] == 1.0, out
        assert patch.calls[0][0] == 1.0, patch.calls


def t_cap_is_applied_before_the_model_reaches_the_gpu():
    src = SCRIPT.read_text(encoding="utf-8")
    call = src.index("apply_vram_cap(args.vram_cap_gib)")
    load = src.index("AutoModelForCausalLM.from_pretrained")
    trainer = src.index("trainer = Trainer(")
    assert call < load < trainer, (
        "the allocator ceiling must be set before the model is moved to the GPU "
        f"(call={call}, load={load}, trainer={trainer})")


def t_cli_exposes_the_flag():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--vram-cap-gib" in proc.stdout, proc.stdout


if __name__ == "__main__":
    run({
        "unset cap is a no-op": t_unset_cap_is_a_noop,
        "negative cap is a no-op": t_negative_cap_is_a_noop,
        "cap sets the fraction on the CUDA device": t_cap_sets_the_fraction_on_the_cuda_device,
        "cap larger than the device clamps to 1.0": t_cap_larger_than_the_device_clamps_to_one,
        "cap is applied before the model reaches the GPU": t_cap_is_applied_before_the_model_reaches_the_gpu,
        "CLI exposes --vram-cap-gib": t_cli_exposes_the_flag,
    })
