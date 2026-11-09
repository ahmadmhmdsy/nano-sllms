"""T0 golden-diff safety net for the flow registry promotion.

Every committed flow/flows/*.flow.json has a golden under
flow/tests/goldens/ (captured with the PRE-refactor dict-based registry by
flow/tests/goldens/create_goldens.py, see its docstring). This test asserts
flow.server.config_gen.build_config reproduces each golden exactly —
proving the NodeDefinition refactor is shape-preserving, not a behavior
change. Error goldens lock the exact refusal wording for branching or
unknobbed graphs.
"""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from flow.server import config_gen  # noqa: E402

FLOWS_DIR = REPO / "flow" / "flows"
GOLDENS_DIR = REPO / "flow" / "tests" / "goldens"


def load_golden(name: str):
    """Parse the golden (config dict, or {"error": message})."""
    with open(GOLDENS_DIR / (name + ".golden.json"), "r", encoding="utf-8") as f:
        return json.load(f)


class GoldenDiffTests(unittest.TestCase):
    """Every committed flow must reproduce its golden exactly."""

    def test_every_committed_flow_has_a_golden(self):
        flows = sorted(p.name[: -len(".flow.json")]
                       for p in FLOWS_DIR.glob("*.flow.json"))
        self.assertTrue(flows, "the flows/ fixtures disappeared")
        for name in flows:
            self.assertTrue((GOLDENS_DIR / (name + ".golden.json")).exists(),
                            "no golden for flow %r - re-run "
                            "flow/tests/goldens/create_goldens.py" % (name,))

    def test_valid_flows_reproduce_the_golden_config(self):
        for flow_path in sorted(FLOWS_DIR.glob("*.flow.json")):
            name = flow_path.name[: -len(".flow.json")]
            golden = load_golden(name)
            if "error" in golden:
                continue  # covered by the refusal test below
            with open(flow_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            self.assertEqual(config_gen.build_config(graph), golden,
                             "flow %r no longer maps to its golden config"
                             % (name,))

    def test_branching_or_unknobbed_flows_raise_the_golden_message(self):
        for flow_path in sorted(FLOWS_DIR.glob("*.flow.json")):
            name = flow_path.name[: -len(".flow.json")]
            golden = load_golden(name)
            if "error" not in golden:
                continue  # covered by the config test above
            with open(flow_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            with self.assertRaises(ValueError) as cm:
                config_gen.build_config(graph)
            self.assertEqual(
                str(cm.exception), golden["error"],
                "flow %r refusal wording drifted from its golden" % (name,))


if __name__ == "__main__":
    unittest.main()
