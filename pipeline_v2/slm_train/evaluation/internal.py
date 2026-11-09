"""Internal validation evaluator: loss and perplexity from real numbers.

This module computes; it does not load a model. The caller supplies the mean
cross-entropy it actually measured, and the evaluator refuses anything it
cannot verify: a non-finite loss, a non-positive loss that would produce an
undefined perplexity, or a sample count of zero.
"""
from __future__ import annotations

from math import exp, isfinite

__all__ = ["EvaluationError", "validation_metrics"]


class EvaluationError(ValueError):
    """An evaluation could not be computed from the values supplied."""


def validation_metrics(mean_loss: object, sample_count: object) -> dict:
    """The exact metric set the validation suite declares, or a refusal."""
    if isinstance(mean_loss, bool) or not isinstance(mean_loss, (int, float)):
        raise EvaluationError("mean loss is not a number: " + repr(mean_loss))
    loss = float(mean_loss)
    if not isfinite(loss):
        raise EvaluationError("mean loss is not finite: " + repr(mean_loss))
    if loss <= 0.0:
        reason = "mean loss must be strictly positive for a finite perplexity: "
        raise EvaluationError(reason + repr(mean_loss))
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise EvaluationError("sample count must be an integer: " + repr(sample_count))
    if sample_count < 1:
        raise EvaluationError("sample count must be positive: " + repr(sample_count))
    try:
        perplexity = exp(loss)
    except OverflowError as error:
        raise EvaluationError("perplexity overflowed for loss " + repr(mean_loss)) from error
    if not isfinite(perplexity):
        raise EvaluationError("perplexity is not finite for loss " + repr(mean_loss))
    return {"val_loss": loss, "val_perplexity": perplexity}
