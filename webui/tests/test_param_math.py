"""Unit tests for the shared param-math package (packages/param-math/param_math.py).

The Python module must produce EXACTLY the same anchors as the TypeScript
package (packages/param-math/src/index.ts) and the viz/flow suites:
nano_SLMs 226,526,208 · SmolLM2-135M 134,515,008.

Run from the repo root:
    .venv/Scripts/python.exe -m unittest discover -s webui/tests -t .
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAM_MATH_DIR = REPO_ROOT / "packages" / "param-math"
if str(PARAM_MATH_DIR) not in sys.path:
    sys.path.insert(0, str(PARAM_MATH_DIR))

import param_math  # noqa: E402  (path inserted above)

# Shape fixtures mirroring viz/models/nano-target.yaml and
# viz/models/smollm2-135m.yaml (smoke/pilot shapes from the flow suite).
NANO = dict(layers=16, d=1024, heads=16, kv=4, ffn=3072, vocab=32768)
SMOLLM2 = dict(layers=30, d=576, heads=9, kv=3, ffn=1536, vocab=49152)


class TestAnchors(unittest.TestCase):
    def test_nano_total(self):
        self.assertEqual(param_math.p_total_dense(**NANO), 226_526_208)

    def test_smollm2_total(self):
        self.assertEqual(param_math.p_total_dense(**SMOLLM2), 134_515_008)

    def test_smoke_and_pilot_totals(self):
        self.assertEqual(
            param_math.p_total_dense(layers=4, d=256, heads=4, kv=2, ffn=1024, vocab=32768),
            12_323_072)
        self.assertEqual(
            param_math.p_total_dense(layers=12, d=768, heads=12, kv=4, ffn=2048, vocab=32768),
            100_682_496)


class TestComponents(unittest.TestCase):
    def test_head_dim(self):
        self.assertEqual(param_math.head_dim(1024, 16), 64)
        self.assertEqual(param_math.head_dim(576, 9), 64)

    def test_kv_dim(self):
        self.assertEqual(param_math.kv_dim(1024, 16, 4), 256)

    def test_p_embed(self):
        self.assertEqual(param_math.p_embed(1024, 32768), 33_554_432)

    def test_p_attention_dense(self):
        self.assertEqual(param_math.p_attention_dense(1024, 16, 4), 2_621_440)

    def test_p_ffn_dense(self):
        self.assertEqual(param_math.p_ffn_dense(1024, 3072), 9_437_184)

    def test_p_norms_counted_once(self):
        self.assertEqual(param_math.p_norms(1024, 16), 33_792)
        self.assertEqual(param_math.p_norms(576, 30), 35_136)


class TestConsumerWiring(unittest.TestCase):
    """explorer.py and status.py must agree with the shared package."""

    def test_explorer_projected_params_matches_shared(self):
        sys.path.insert(0, str(REPO_ROOT / "webui"))
        import explorer
        dims = {"layers": 16, "hidden": 1024, "heads": 16, "kv_heads": 4,
                "ffn": 3072, "vocab": 32768, "tie": True, "ctx": 1024,
                "rope_theta": 10000, "rms_eps": 1e-5}
        self.assertEqual(explorer.projected_params(dims), 226_526_208)

    def test_status_estimate_params_matches_shared(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import status
        model_cfg = {"layers": 16, "hidden": 1024, "heads": 16, "kv_heads": 4,
                     "ffn": 3072, "tie_embeddings": True}
        self.assertEqual(status.estimate_params(model_cfg, 32768), 226_526_208)
        # untied: + one more vocab*h for the separate lm_head
        untied = dict(model_cfg, tie_embeddings=False)
        self.assertEqual(status.estimate_params(untied, 32768),
                         226_526_208 + 32768 * 1024)


if __name__ == "__main__":
    unittest.main()
