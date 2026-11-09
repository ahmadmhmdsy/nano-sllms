import pathlib
import tempfile
import unittest

import numpy as np

from langid.src import features
from langid.src.infer import Int8LangID, quantize_columns


class TestQuantize(unittest.TestCase):
    def test_round_trip_error_bound(self):
        rng = np.random.RandomState(0)
        W = rng.randn(64, 3) * 0.1
        Wq, scale = quantize_columns(W)
        self.assertEqual(Wq.dtype, np.int8)
        max_abs = np.max(np.abs(W), axis=0)
        err = np.abs(Wq.astype(np.float64) / scale - W)
        self.assertTrue(np.all(err <= max_abs / 254.0 + 1e-12))


class TestInt8LangID(unittest.TestCase):
    def test_scores_and_predict(self):
        # shrink the hash space to match the synthetic model's buckets
        # (bucket() reads the module global at call time) and restore after
        saved = features.NUM_BUCKETS
        features.NUM_BUCKETS = 1 << 12
        try:
            rng = np.random.RandomState(1)
            W = rng.randn(1 << 12, 2) * 0.05
            Wq, scale = quantize_columns(W)
            bias = np.array([0.1, -0.1])
            with tempfile.TemporaryDirectory() as td:
                p = pathlib.Path(td) / "m.npz"
                np.savez(p, W=Wq, scale=scale, bias=bias,
                         langs=np.array(["en", "fr"]))
                m = Int8LangID(str(p))
                s, langs = m.scores("bonjour le monde")
                self.assertEqual(s.shape, (2,))
                self.assertEqual(langs, ["en", "fr"])
                self.assertIsNone(m.scores("123 !!!")[0])
                pred, tie = m.predict("hello")
                self.assertIsInstance(tie, bool)
                self.assertIn(pred, ("en", "fr", None))
        finally:
            features.NUM_BUCKETS = saved
