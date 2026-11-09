"""Shared eval statistics for the mex harness (mu1 plan: binomial 95% CI).

wilson_ci() implements the Wilson score interval on a binomial proportion -
the interval the mu1 plan pins every exact-match / routing claim to (gate
discipline: never report a bare rate on small held-out sets).
"""
from __future__ import annotations

import math

# Two-sided 95%: norm.ppf(0.975), quantised once so results are byte-stable.
Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> list[float]:
    """Wilson score interval [lo, hi] for k successes in n Bernoulli trials.

    Extremes are exact (never clamped): k=0 gives lo=0, k=n gives hi=1 because
    the sqrt term equals z*z/(2n) and the center cancels it.
    """
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"invalid k={k} for n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return [(center - half) / denom, (center + half) / denom]
