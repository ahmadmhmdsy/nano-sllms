"""Generation smoke evaluator over decoded text the caller actually produced.

The evaluator is deliberately dumb: it scores exactly what it is given and
refuses anything it cannot score. It never calls a model, never samples, and
never treats an empty decode as a pass.
"""
from __future__ import annotations


def generation_metrics(samples) -> dict:
    """generation_smoke is 1 only when every sample is non-empty text."""
    if not isinstance(samples, (list, tuple)):
        raise ValueError("generation samples must be a sequence")
    if not samples:
        raise ValueError("generation smoke needs at least one sample")
    for index, sample in enumerate(samples):
        if not isinstance(sample, str):
            raise ValueError("sample " + str(index) + " is not text")
    produced = [sample for sample in samples if sample.strip()]
    return {"generation_smoke": 1.0 if len(produced) == len(samples) else 0.0}


def smoke_prompt() -> str:
    """The one fixed prompt, so the smoke is comparable across runs."""
    return "The capital of France is"


def generation_details(samples) -> dict:
    """What was decoded, recorded verbatim rather than summarised."""
    return {"prompt": smoke_prompt(), "samples": list(samples)}
