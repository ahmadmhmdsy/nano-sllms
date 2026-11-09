"""Tests for the flow.server.nodes registry (Task 3).

The registry owns kinds, per-kind ports (the 6-node MVP table), editable
props and gates; flow.server.graph_schema lazy-imports VALID_KINDS/PORTS
from it and must reject edges whose toPort OR fromPort is not a real port
of the endpoint kind.
"""
import unittest

from flow.server import nodes, graph_schema


def make_node(id="n1", kind="dataset", position=(0, 0), props=None):
    return {
        "id": id,
        "kind": kind,
        "props": props if props is not None else {},
        "position": {"x": position[0], "y": position[1]},
    }


def make_edge(id="e1", frm="n1", frmPort="cleaned", to="n2", toPort="cleaned-dir"):
    return {"id": id, "from": frm, "fromPort": frmPort, "to": to, "toPort": toPort}


def doc_for(nodes_list, edges=None):
    return {
        "schema": "flow/0.1",
        "meta": {"name": "t"},
        "graph": {"nodes": nodes_list, "edges": [] if edges is None else edges},
    }


class TestRegistryTable(unittest.TestCase):
    def test_six_kinds_exactly(self):
        self.assertEqual(
            nodes.VALID_KINDS,
            frozenset({"dataset", "prepare", "tokenize", "train", "eval", "infer"}),
        )

    def test_ports_match_the_six_node_table(self):
        expected = {
            "dataset": {"in": [], "out": ["cleaned"]},
            "prepare": {"in": ["raw-dir"], "out": ["cleaned-dir"]},
            "tokenize": {"in": ["cleaned-dir"], "out": ["shard-dir"]},
            "train": {"in": ["shard-dir"], "out": ["ckpt-dir"]},
            "eval": {"in": ["ckpt-dir"], "out": ["report"]},
            "infer": {"in": ["ckpt-dir"], "out": []},
        }
        self.assertEqual(nodes.PORTS, expected)

    def test_each_kind_exposes_rich_port_specs(self):
        for kind, specs in nodes.PORT_SPECS.items():
            for port in specs:
                self.assertEqual(
                    sorted(port),
                    sorted(["name", "direction", "type", "burst-format"]),
                    (kind, port),
                )

    def test_props_keys_and_gates_match_the_table(self):
        self.assertEqual(nodes.PROPS["prepare"],
                         frozenset({"rows", "val_fraction", "min_chars"}))
        self.assertEqual(nodes.PROPS["tokenize"], frozenset({"seq_len", "vocab"}))
        self.assertEqual(nodes.PROPS["train"],
                         frozenset({"preset", "steps", "lr_preset"}))
        self.assertEqual(nodes.GATES, {
            "dataset": None,
            "prepare": None,
            "tokenize": None,
            "train": "gpu",
            "eval": "gpu",
            "infer": "gpu/cpu",
        })

    def test_node_view_contains_ports_props_gate(self):
        for kind in nodes.VALID_KINDS:
            view = nodes.NODES[kind]
            self.assertIn("ports", view)
            self.assertIn("props", view)
            self.assertIn("gate", view)


class TestValidateProps(unittest.TestCase):
    def test_known_numeric_props_accept_finite_numbers(self):
        self.assertEqual(nodes.validate_props("prepare",
                                              {"rows": 1000, "val_fraction": 0.1, "min_chars": 3}), [])
        self.assertEqual(nodes.validate_props("tokenize", {"seq_len": 512, "vocab": 8000}), [])
        self.assertEqual(nodes.validate_props("train", {"steps": 100, "preset": "smoke",
                                                        "lr_preset": "s1"}), [])

    def test_unknown_prop_key_is_rejected(self):
        self.assertEqual(len(nodes.validate_props("prepare", {"bogus": 1})), 1)
        self.assertEqual(len(nodes.validate_props("train", {"bogus": 1})), 1)

    def test_non_finite_and_bool_numbers_are_rejected(self):
        self.assertEqual(len(nodes.validate_props("prepare", {"rows": float("inf")})), 1)
        self.assertEqual(len(nodes.validate_props("prepare", {"rows": float("nan")})), 1)
        self.assertEqual(len(nodes.validate_props("tokenize", {"seq_len": True})), 1)
        self.assertEqual(len(nodes.validate_props("tokenize", {"seq_len": "512"})), 1)

    def test_empty_props_pass_for_kinds_with_no_props(self):
        for kind in ("dataset", "eval", "infer"):
            self.assertEqual(nodes.validate_props(kind, {}), [])

    def test_props_must_be_a_dict(self):
        self.assertEqual(len(nodes.validate_props("prepare", [1, 2])), 1)

    def test_unknown_kind_is_rejected(self):
        self.assertEqual(len(nodes.validate_props("quantum", {})), 1)


class TestGraphSchemaUsesRegistryPorts(unittest.TestCase):
    """The registry is the source of truth for BOTH toPort and fromPort."""

    def chain(self, edges):
        nodes_list = [
            make_node("a", kind="dataset"),
            make_node("b", kind="tokenize"),
        ]
        return doc_for(nodes_list, edges)

    def test_valid_registry_port_edge_is_accepted(self):
        doc = self.chain([make_edge("e1", "a", "cleaned", "b", "cleaned-dir")])
        self.assertEqual(graph_schema.validate(doc), [])

    def test_to_port_not_in_kind_inputs_is_rejected(self):
        doc = self.chain([make_edge("e1", "a", "cleaned", "b", "ckpt-dir")])
        errors = graph_schema.validate(doc)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("toPort", errors[0])

    def test_from_port_not_in_kind_outputs_is_rejected(self):
        doc = self.chain([make_edge("e1", "a", "raw-dir", "b", "cleaned-dir")])
        errors = graph_schema.validate(doc)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("fromPort", errors[0])

    def test_missing_edges_key_is_treated_as_empty(self):
        doc = doc_for([make_node("n1", kind="dataset"), make_node("n2")])
        del doc["graph"]["edges"]
        self.assertEqual(graph_schema.validate(doc), [])

if __name__ == "__main__":
    unittest.main()
