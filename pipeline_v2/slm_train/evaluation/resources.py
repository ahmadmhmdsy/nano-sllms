"""Throughput and resource reporting.

Every value here is what the caller measured. Nothing is estimated, and a
measurement that was not taken is a refusal rather than a zero, because a
fabricated zero would read as a real measurement of no memory use.
"""
from __future__ import annotations

from math import isfinite

__all__ = ["resource_metrics"]


def _non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " was not measured: " + repr(value))
    number = float(value)
    if not isfinite(number):
        raise ValueError(name + " is not finite: " + repr(value))
    if number < 0.0:
        raise ValueError(name + " must not be negative: " + repr(value))
    return number


def resource_metrics(*, tokens: object, seconds: object, peak_vram_bytes: object) -> dict:
    """Throughput and peak memory from real measurements."""
    token_count = _non_negative("tokens", tokens)
    elapsed = _non_negative("seconds", seconds)
    peak = _non_negative("peak_vram_bytes", peak_vram_bytes)
    if token_count <= 0.0:
        raise ValueError("tokens must be positive to compute throughput")
    if elapsed <= 0.0:
        raise ValueError("seconds must be positive to compute throughput")
    return {
        "tokens_per_second": token_count / elapsed,
        "peak_vram_bytes": peak,
    }
