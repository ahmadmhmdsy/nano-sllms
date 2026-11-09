"""Tests for the promoted flow.server.nodes package (promotion T1).

The registry now carries BEHAVIOR (NodeDefinition classes in
flow/server/nodes/builtin/) instead of dict entries. This file pins the
package-level API: canonical registration order, NodeRegistry.get
raising NodeRegistryError on unknown kinds, the derived compat surface
matching the exact legacy dict shapes (the dict-entry registry of the
pre-promotion flow/server/nodes.py), required_upstream declarations,
and the label_semantic / features fields the T3 frontend expects.
Kind-port-table and validate_props behavior live in the Task-3 tests
(flow/tests/server/test_registry.py) and stay green unchanged.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from flow.server import nodes, config_gen  # noqa: E402
from flow.server.nodes import NodeRegistry, NodeRegistryError  # noqa: E402
from flow.server.nodes.base import NodeDefinition  # noqa: E402
from flow.server.nodes.ports import port_compatible  # noqa: E402


class NodeRegistryApiTests(unittest.TestCase):
    """The registry object and its canonical order."""

    def test_canonical_kind_order(self):
        self.assertEqual(
            nodes.REGISTRY.kinds(),
            ("dataset", "prepare", "tokenize", "train", "eval", "infer"))

    def test_definitions_have_display_metadata(self):
        for kind in nodes.REGISTRY.kinds():
            defn = nodes.REGISTRY.get(kind)
            self.assertIsInstance(defn, NodeDefinition)
            self.assertEqual(defn.kind, kind)
            self.assertTrue(defn.title)
            self.assertTrue(defn.category)

    def test_get_unknown_kind_raises(self):
        with self.assertRaises(NodeRegistryError):
            nodes.REGISTRY.get("quantum")

    def test_register_rejects_duplicate_and_foreign(self):
        registry = NodeRegistry()
        registry.register(nodes.REGISTRY.get("dataset"))
        with self.assertRaises(NodeRegistryError):
            registry.register(nodes.REGISTRY.get("dataset"))
        with self.assertRaises(NodeRegistryError):
            registry.register("not a definition")


class DerivedCompatSurfaceTests(unittest.TestCase):
    """The derived surface equals the legacy dict-entry shapes."""

    def test_valid_kinds(self):
        self.assertEqual(
            nodes.VALID_KINDS,
            frozenset({"dataset", "prepare", "tokenize", "train",
                       "eval", "infer"}))

    def test_port_specs_match_the_legacy_table(self):
        legacy = {
            "dataset": {
                "cleaned": ("out", "raw-dir: dataset name/rows",
                            "row-batch")},
            "prepare": {
                "raw-dir": ("in", "raw-dir", "file-list"),
                "cleaned-dir": ("out", "cleaned-dir", "file-list")},
            "tokenize": {
                "cleaned-dir": ("in", "cleaned-dir", "file-list"),
                "shard-dir": ("out", "shard-dir", "file-list")},
            "train": {
                "shard-dir": ("in", "shard-dir", "file-list"),
                "ckpt-dir": ("out", "ckpt-dir", "checkpoint")},
            "eval": {
                "ckpt-dir": ("in", "ckpt-dir", "checkpoint"),
                "report": ("out", "report", "single-file")},
            "infer": {
                "ckpt-dir": ("in", "ckpt-dir", "checkpoint")},
        }
        self.assertEqual(set(nodes.PORT_SPECS), set(legacy))
        for kind, expected in legacy.items():
            specs = nodes.PORT_SPECS[kind]
            self.assertEqual(len(specs), len(expected), (kind, specs))
            for spec in specs:
                self.assertEqual(spec["name"] in expected, True, spec)
                direction, ptype, burst = expected[spec["name"]]
                self.assertEqual(spec, {
                    "name": spec["name"],
                    "direction": direction,
                    "type": ptype,
                    "burst-format": burst,
                })

    def test_ports_shape_for_graph_schema(self):
        legacy = {
            "dataset": {"in": [], "out": ["cleaned"]},
            "prepare": {"in": ["raw-dir"], "out": ["cleaned-dir"]},
            "tokenize": {"in": ["cleaned-dir"], "out": ["shard-dir"]},
            "train": {"in": ["shard-dir"], "out": ["ckpt-dir"]},
            "eval": {"in": ["ckpt-dir"], "out": ["report"]},
            "infer": {"in": ["ckpt-dir"], "out": []},
        }
        self.assertEqual(nodes.PORTS, legacy)

    def test_props_numeric_props_gates(self):
        self.assertEqual(
            nodes.PROPS,
            {"dataset": frozenset(),
             "prepare": frozenset({"rows", "val_fraction", "min_chars"}),
             "tokenize": frozenset({"seq_len", "vocab"}),
             "train": frozenset({"preset", "steps", "lr_preset"}),
             "eval": frozenset(),
             "infer": frozenset()})
        self.assertEqual(
            nodes.NUMERIC_PROPS,
            {"dataset": frozenset(),
             "prepare": frozenset({"rows", "val_fraction", "min_chars"}),
             "tokenize": frozenset({"seq_len", "vocab"}),
             "train": frozenset({"steps"}),
             "eval": frozenset(),
             "infer": frozenset()})
        self.assertEqual(
            nodes.GATES,
            {"dataset": None, "prepare": None, "tokenize": None,
             "train": "gpu", "eval": "gpu", "infer": "gpu/cpu"})

    def test_nodes_view_sorted_props_with_new_snapshot_fields(self):
        self.assertEqual(nodes.NODES["train"], {
            "ports": nodes.PORT_SPECS["train"],
            "props": ["lr_preset", "preset", "steps"],
            "numeric_props": ["steps"],
            "gate": "gpu",
            "label_semantic": None,
            "features": [],
        })
        self.assertEqual(nodes.NODES["dataset"]["ports"],
                         nodes.PORT_SPECS["dataset"])

    def test_config_gen_reexports_node_consts_for_parity_tests(self):
        from flow.server.nodes.builtin import prepare, tokenize, train
        self.assertIs(config_gen.MAX_ROWS, prepare.MAX_ROWS)
        self.assertIs(config_gen.PRESETS, train.PRESETS)
        self.assertIs(config_gen.LR_PRESETS, train.LR_PRESETS)
        self.assertIs(config_gen.TOKENIZER_NAME, tokenize.TOKENIZER_NAME)
        self.assertIs(config_gen.SHARD_TOKENS, tokenize.SHARD_TOKENS)


class DefinitionDeclarationTests(unittest.TestCase):
    """Per-definition declarative data (required_upstream, features)."""

    def setUp(self):
        self.defn = nodes.REGISTRY.get

    def test_required_upstream_declares_the_linear_chain(self):
        self.assertEqual(self.defn("dataset").required_upstream, {})
        self.assertEqual(self.defn("prepare").required_upstream,
                         {"raw-dir": ("dataset", "cleaned")})
        self.assertEqual(self.defn("tokenize").required_upstream,
                         {"cleaned-dir": ("prepare", "cleaned-dir")})
        self.assertEqual(self.defn("train").required_upstream,
                         {"shard-dir": ("tokenize", "shard-dir")})
        # Non-participants declare nothing yet (future F5 execution).
        self.assertEqual(self.defn("eval").required_upstream, {})
        self.assertEqual(self.defn("infer").required_upstream, {})

    def test_config_participants_are_the_mapping_chain_only(self):
        participants = {kind for kind in nodes.REGISTRY.kinds()
                        if self.defn(kind).config_participant}
        self.assertEqual(participants,
                         {"dataset", "prepare", "tokenize", "train"})

    def test_label_semantics_and_features(self):
        self.assertEqual(self.defn("dataset").label_semantic,
                         "hf-dataset-name")
        for kind in ("prepare", "tokenize", "train", "eval", "infer"):
            self.assertIsNone(self.defn(kind).label_semantic)
        for kind in ("dataset", "prepare", "tokenize", "train", "eval"):
            self.assertEqual(self.defn(kind).features, ())
        self.assertEqual(self.defn("infer").features,
                         ("webui-chat-button",))


class BaseDefaultsTests(unittest.TestCase):
    """The abstract base contract itself."""

    def test_base_validate_semantic_is_empty_and_section_is_abstract(self):
        defn = NodeDefinition()
        self.assertEqual(defn.validate_semantic({"id": "x"}, None), [])
        with self.assertRaises(NotImplementedError):
            defn.build_section(None)


class PortCompatRuleTests(unittest.TestCase):
    """port_compatible: the strict-equality rule named once."""

    def test_strict_equality(self):
        self.assertTrue(port_compatible("shard-dir", "shard-dir"))
        self.assertFalse(port_compatible("shard-dir", "ckpt-dir"))
        self.assertFalse(port_compatible("shard-dir", None))


if __name__ == "__main__":
    unittest.main()
