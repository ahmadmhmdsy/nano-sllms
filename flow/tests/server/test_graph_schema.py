"""Tests for flow.server.graph_schema (Task 2 of the flow/ MVP plan).

Written before the implementation (strict TDD). validate(g) -> list[str]
returns [] for valid documents and one human-readable reason string per
violation.
"""
import unittest

from flow.server.graph_schema import validate

SAFE = "flow/0.1"


def make_node(id="n1", kind="dataset", position=(0, 0), props=None):
    return {
        "id": id,
        "kind": kind,
        "props": props if props is not None else {},
        "position": {"x": position[0], "y": position[1]},
    }


def make_edge(id="e1", frm="n1", frmPort="cleaned", to="n2", toPort="shard-dir"):
    return {"id": id, "from": frm, "fromPort": frmPort, "to": to, "toPort": toPort}


def valid_doc(nodes=None, edges=None, schema=SAFE, meta=None):
    if meta is None:
        meta = {"name": "test-graph"}
    return {
        "schema": schema,
        "meta": meta,
        "graph": {
            "nodes": [
                make_node("n1", "dataset"),
                make_node("n2", "train"),
            ] if nodes is None else nodes,
            "edges": [] if edges is None else edges,
        },
    }


class TestValidDocuments(unittest.TestCase):
    def test_completely_valid_document_returns_empty_list(self):
        doc = valid_doc(
            edges=[make_edge("e1", "n1", "cleaned", "n2", "shard-dir")],
        )
        self.assertEqual(validate(doc), [])

    def test_nodes_without_edges_are_valid(self):
        self.assertEqual(validate(valid_doc()), [])


class TestSchemaField(unittest.TestCase):
    def test_missing_schema_is_rejected(self):
        doc = valid_doc()
        del doc["schema"]
        errors = validate(doc)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("schema", errors[0])

    def test_wrong_schema_is_rejected(self):
        errors = validate(valid_doc(schema="flow/0.2"))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("schema", errors[0])

    def test_wrong_schema_mentions_forward_migration(self):
        errors = validate(valid_doc(schema="flow/1.0"))
        joined = " ".join(errors)
        self.assertIn("flow/0.1", joined)
        self.assertIn("migration", joined.lower())


class TestMeta(unittest.TestCase):
    def test_missing_meta_name_is_rejected(self):
        doc = valid_doc(meta={"version": 1})
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("meta.name", joined)

    def test_empty_meta_name_is_rejected(self):
        doc = valid_doc(meta={"name": ""})
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("meta.name", joined)

    def test_non_string_meta_name_is_rejected(self):
        doc = valid_doc(meta={"name": 42})
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("meta.name", joined)


class TestEmptyGraph(unittest.TestCase):
    def test_empty_nodes_list_is_rejected(self):
        doc = valid_doc(nodes=[], edges=[])
        errors = validate(doc)
        self.assertEqual(len(errors), 1, errors)

    def test_missing_nodes_key_is_rejected(self):
        doc = valid_doc(nodes=[])
        del doc["graph"]["nodes"]
        errors = validate(doc)
        self.assertTrue(errors, errors)


class TestNodes(unittest.TestCase):
    def test_duplicate_node_ids_are_rejected(self):
        doc = valid_doc(
            nodes=[make_node("n1"), make_node("n1")],
            edges=[],
        )
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("n1", joined)
        self.assertIn("duplicate", joined.lower())

    def test_unknown_kind_is_rejected(self):
        doc = valid_doc(
            nodes=[make_node("n1", kind="quantum")]
        )
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("n1", joined)
        self.assertIn("quantum", joined)

    def test_missing_node_id_is_rejected(self):
        node = make_node()
        del node["id"]
        errors = validate(valid_doc(nodes=[node], edges=[]))
        self.assertTrue(errors, errors)

    def test_missing_kind_is_rejected(self):
        node = make_node()
        del node["kind"]
        errors = validate(valid_doc(nodes=[node], edges=[]))
        self.assertTrue(errors, errors)
        joined = " ".join(errors)
        self.assertIn("kind", joined)

    def test_missing_position_is_rejected(self):
        node = make_node()
        del node["position"]
        errors = validate(valid_doc(nodes=[node], edges=[]))
        self.assertTrue(errors, errors)

    def test_position_missing_x_or_y_is_rejected(self):
        node = make_node()
        del node["position"]["y"]
        errors = validate(valid_doc(nodes=[node], edges=[]))
        self.assertTrue(errors, errors)
        joined = " ".join(errors)
        self.assertIn("position", joined)

    def test_position_non_numeric_is_rejected(self):
        node = make_node()
        node["position"]["x"] = "0"
        errors = validate(valid_doc(nodes=[node], edges=[]))
        joined = " ".join(errors)
        self.assertIn("position", joined)

    def test_missing_props_is_rejected(self):
        node = make_node()
        del node["props"]
        errors = validate(valid_doc(nodes=[node], edges=[]))
        self.assertTrue(errors, errors)

    def test_each_known_kind_is_accepted(self):
        for kind in ("dataset", "prepare", "tokenize", "train", "eval", "infer"):
            doc = valid_doc(nodes=[make_node(kind, kind=kind)])
            self.assertEqual(
                validate(doc), [], "kind %r should be valid" % kind
            )


class TestEdges(unittest.TestCase):
    def test_edge_to_unknown_node_id_is_rejected(self):
        doc = valid_doc(edges=[make_edge("e1", "n1", "cleaned", "ghost", "cleaned-dir")])
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("ghost", joined)
        self.assertIn("e1", joined)

    def test_edge_from_unknown_node_id_is_rejected(self):
        doc = valid_doc(edges=[make_edge("e1", "ghost", "cleaned", "n2", "cleaned-dir")])
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("ghost", joined)

    def test_duplicate_edge_ids_are_rejected(self):
        doc = valid_doc(
            edges=[make_edge("e1"), make_edge("e1")],
        )
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("e1", joined)
        self.assertIn("duplicate", joined.lower())

    def test_missing_edge_id_is_rejected(self):
        edge = make_edge()
        del edge["id"]
        errors = validate(valid_doc(edges=[edge]))
        self.assertTrue(errors, errors)
        joined = " ".join(errors)
        self.assertIn("id", joined)

    def test_missing_from_port_is_rejected(self):
        edge = make_edge()
        del edge["fromPort"]
        errors = validate(valid_doc(edges=[edge]))
        joined = " ".join(errors)
        self.assertIn("fromPort", joined)

    def test_missing_to_port_is_rejected(self):
        edge = make_edge()
        del edge["toPort"]
        errors = validate(valid_doc(edges=[edge]))
        joined = " ".join(errors)
        self.assertIn("toPort", joined)

    def test_empty_or_placeholder_port_names_are_rejected(self):
        for bad in ("", "<from_port>", "TODO", "?"):
            with self.subTest(bad=bad):
                errors = validate(
                    valid_doc(edges=[make_edge("e1", "n1", bad, "n2", "shard-dir")])
                )
                self.assertTrue(errors, errors)

    def test_mismatched_port_names_are_rejected(self):
        # fromPort must be a source-side output port, toPort an input-side port
        errors = validate(
            valid_doc(edges=[make_edge("e1", "n1", "in", "n2", "in")])
        )
        joined = " ".join(errors)
        self.assertIn("port", joined.lower())
        self.assertIn("out", joined.lower())


    def test_self_edge_is_rejected(self):
        doc = valid_doc(edges=[make_edge("e1", "n1", "cleaned", "n1", "cleaned-dir")])
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("itself", joined.lower())


class TestCycles(unittest.TestCase):
    def _nodes_for(self):
        return [
            make_node("a", kind="dataset"),
            make_node("b", kind="tokenize"),
            make_node("c", kind="train"),
        ]

    def test_simple_cycle_is_rejected(self):
        edges = [make_edge("e1", "a", "cleaned", "b", "cleaned-dir"),
                 make_edge("e2", "b", "shard-dir", "c", "shard-dir"),
                 make_edge("e3", "c", "ckpt-dir", "a", "cleaned-dir")]
        doc = valid_doc(nodes=self._nodes_for(), edges=edges)
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("cycle", joined.lower())

    def test_two_node_cycle_is_rejected(self):
        edges = [make_edge("e1", "a", "cleaned", "b", "cleaned-dir"),
                 make_edge("e2", "b", "shard-dir", "a", "cleaned-dir")]
        doc = valid_doc(nodes=self._nodes_for(), edges=edges)
        errors = validate(doc)
        joined = " ".join(errors)
        self.assertIn("cycle", joined.lower())

    def test_acyclic_chain_is_accepted(self):
        edges = [make_edge("e1", "a", "cleaned", "b", "cleaned-dir"),
                 make_edge("e2", "b", "shard-dir", "c", "shard-dir")]
        doc = valid_doc(nodes=self._nodes_for(), edges=edges)
        self.assertEqual(validate(doc), [])

    def test_diamond_is_accepted(self):
        edges = [make_edge("e1", "a", "cleaned", "b", "cleaned-dir"),
                 make_edge("e2", "a", "cleaned", "c", "shard-dir"),
                 make_edge("e3", "b", "shard-dir", "c", "shard-dir")]
        doc = valid_doc(nodes=self._nodes_for(), edges=edges)
        self.assertEqual(validate(doc), [])




class TestMultipleErrors(unittest.TestCase):
    def test_multiple_violations_are_all_reported(self):
        doc = valid_doc(
            nodes=[make_node("n1"), make_node("n1", kind="quantum")],
            edges=[make_edge("e1", "n1", "cleaned", "ghost", "cleaned-dir")],
        )
        errors = validate(doc)
        self.assertGreaterEqual(len(errors), 3, errors)


if __name__ == "__main__":
    unittest.main()
