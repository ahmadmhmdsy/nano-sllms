# mex/tests/test_metrics.py — Wilson interval known-value pins + guard rails.
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.metrics import Z95, wilson_ci

# Independent hand computation (z = norm.ppf(0.975)), not a re-run of the
# implementation: e.g. (3,10): p=.3, center=.39208, half=1.96*sqrt(.02265)
# = .29477, denom=1.38416 -> [.10779, .60322].
KNOWN = [
    ((3, 10), (0.10779127, 0.60322185)),
    ((0, 10), (0.00000000, 0.27753280)),
    ((10, 10), (0.72246720, 1.00000000)),
    ((1, 5), (0.03622411, 0.62446537)),
]


class TestWilsonCI(unittest.TestCase):
    def test_known_values_95(self):
        for (k, n), (lo, hi) in KNOWN:
            got = wilson_ci(k, n)
            self.assertAlmostEqual(got[0], lo, places=7, msg=str((k, n)))
            self.assertAlmostEqual(got[1], hi, places=7, msg=str((k, n)))

    def test_extremes_touch_bounds_clamp_free(self):
        lo, hi = wilson_ci(0, 9)
        self.assertAlmostEqual(lo, 0.0, places=10)
        self.assertLess(hi, 0.375, "0 successes in 9: hi = 3.8416/13.8416")
        lo, hi = wilson_ci(9, 9)
        self.assertGreater(lo, 0.7, "9/9: lo = 9/13.8416")
        self.assertAlmostEqual(hi, 1.0, places=10)

    def test_interval_contains_phat_and_is_narrow(self):
        lo, hi = wilson_ci(650, 2000)
        self.assertLessEqual(lo, 0.325)
        self.assertGreaterEqual(hi, 0.325)
        self.assertLess(hi - lo, 0.05)

    def test_biased_z_passthrough(self):
        self.assertAlmostEqual(wilson_ci(3, 10, z=1.96)[0],
                               wilson_ci(3, 10, z=1.959963984540054)[0],
                               delta=2e-4)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            wilson_ci(0, 0)
        with self.assertRaises(ValueError):
            wilson_ci(11, 10)

    def test_z_is_95_percent_quantized_constant(self):
        import math
        # Z95^2 == chi2.ppf(0.95, 1) == 3.841458820694124 (norm.ppf(0.975)^2).
        self.assertAlmostEqual(Z95, math.sqrt(3.841458820694124), places=8)
