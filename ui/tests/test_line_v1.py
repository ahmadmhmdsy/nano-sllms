import unittest

from ui.src.chain import spec_to_chain, chain_to_spec, CATALOG_FROZEN
from ui.src.generator import ARCHETYPES, build
from ui.src.validate import validate_spec, validate_chain, _norm


class Generator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build(42, "todo_list"), build(42, "todo_list"))

    def test_all_archetypes_valid(self):
        for seed in (0, 7, 99, 12345):
            for arch in ARCHETYPES:
                try:
                    task, spec = build(seed, arch)
                except RuntimeError as e:
                    raise AssertionError(str(e))


class RoundTrip(unittest.TestCase):
    def test_roundtrip_all(self):
        for seed in (1, 2, 5000):
            for arch in ARCHETYPES:
                task, spec = build(seed, arch)
                chain = spec_to_chain(spec, task)
                spec2, issues = chain_to_spec(chain)
                self.assertEqual(issues, [])
                self.assertEqual(validate_chain(chain), [], "seed={0} arch={1}".format(seed, arch))
                chain2 = spec_to_chain(spec2, task)
                self.assertEqual(chain2, chain)

    def test_frozen_id(self):
        self.assertEqual(CATALOG_FROZEN, "UI1-2026-09-24-41c")


class Validator(unittest.TestCase):
    def setUp(self):
        self.base = {"root": "r", "elements": {"r": {"type": "Card", "props": {}, "children": []}}}

    def test_unknown_type(self):
        spec = self.base
        spec["elements"]["x"] = {"type": "NoSuch", "props": {}, "children": []}
        errs = validate_spec(spec)
        self.assertTrue(any(e[1] == "unknown_type" for e in errs))

    def test_unknown_event(self):
        spec = self.base
        spec["elements"]["x"] = {"type": "Stack", "props": {"direction": "vertical"}, "children": [],
                                  "on": {"press": {"action": "setState", "params": {}}}}
        self.assertTrue(any(e[1] == "unknown_event_for_type" for e in validate_spec(spec)))

    def test_unknown_action(self):
        spec = self.base
        spec["elements"]["x"] = {"type": "Button", "props": {"label": "Go"}, "children": [],
                                  "on": {"press": {"action": "explode", "params": {}}}}
        self.assertTrue(any(e[1] == "unknown_action" for e in validate_spec(spec)))

    def test_missing_state_path(self):
        spec = self.base
        spec["elements"]["x"] = {"type": "Heading", "props": {"text": {"$state": "/nope/x"}}, "children": []}
        spec["state"] = {"real": 1}
        self.assertTrue(any(e[1] == "binding_path_missing" for e in validate_spec(spec)))

    def test_item_outside_repeat(self):
        spec = self.base
        spec["elements"]["x"] = {"type": "Heading", "props": {"text": {"$item": "title"}}, "children": []}
        spec["state"] = {"todos": [{"title": "x"}]}
        self.assertTrue(any(e[1] == "repeat_item_outside_scope" for e in validate_spec(spec)))

    def test_ok_minimal(self):
        self.assertEqual(validate_spec(self.base), [])


if __name__ == "__main__":
    unittest.main()
