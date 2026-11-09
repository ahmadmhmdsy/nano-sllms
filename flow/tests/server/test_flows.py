"""Tests for the flow.server.flows store (Task 4).

The store persists .flow.json documents under flow/flows/ (git-tracked, Q2).
Contract (documented in flow/server/flows.py):

- save(name, g) validates the graph FIRST via BOTH validators
  (graph_schema.validate AND nodes.validate_props per node — the Task-3
  reviewer-noted wiring) and raises ValueError with the newline-joined
  reason list on any violation; on success it writes
  flows/<slug>.flow.json, fsyncs, and returns the written Path.
- load(name) returns the parsed dict or raises a clean FileNotFoundError.
- list_flows() returns slugs (without the .flow.json suffix), sorted.
- names are slugs: lowercase [a-z0-9-]{1,64}; anything containing path
  separators (both / and \\) or ".." is rejected — no path traversal.

Tests run the store against a tempfile directory, never the repo's
flow/flows/ (no test artifacts are committed).
"""
import json
import tempfile
import unittest
from pathlib import Path

from flow.server import flows

BACKSLASH = chr(92)


def make_node(id="n1", kind="dataset", position=(0, 0), props=None):
    return {
        "id": id,
        "kind": kind,
        "props": props if props is not None else {},
        "position": {"x": position[0], "y": position[1]},
    }


def make_edge(id="e1", frm="n1", frmPort="cleaned", to="n2", toPort="raw-dir"):
    return {"id": id, "from": frm, "fromPort": frmPort, "to": to, "toPort": toPort}


def valid_doc():
    return {
        "schema": "flow/0.1",
        "meta": {"name": "demo"},
        "graph": {
            "nodes": [make_node("n1", "dataset"), make_node("n2", "prepare")],
            "edges": [make_edge()],
        },
    }


class FlowsStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # --- save ----------------------------------------------------------

    def test_save_writes_slug_file_and_returns_path(self):
        p = flows.save("demo-flow", valid_doc(), flows_dir=self.dir)
        self.assertEqual(p, self.dir / "demo-flow.flow.json")
        self.assertTrue(p.exists())
        self.assertEqual(json.loads(p.read_text(encoding="utf-8")), valid_doc())

    def test_save_overwrite_is_atomic_no_tmp_residue(self):
        """Atomic write (retro MUST-ADD): an overwrite replaces the file via
        os.replace so readers never observe a torn .flow.json, and no .tmp
        residue survives a successful save (or a failed one)."""
        flows.save("demo", valid_doc(), flows_dir=self.dir)
        p = self.dir / "demo.flow.json"
        before = p.read_text(encoding="utf-8")
        changed = valid_doc()
        changed["meta"]["name"] = "demo-2"
        flows.save("demo", changed, flows_dir=self.dir)
        self.assertEqual(json.loads(p.read_text(encoding="utf-8")), changed)
        self.assertNotEqual(before, p.read_text(encoding="utf-8"))
        residue = [q.name for q in self.dir.iterdir()
                   if not q.name.endswith(".flow.json")]
        self.assertEqual(residue, [], "no temp files may survive a save")
        # A failed save (validation error) also leaves no residue and keeps
        # the previous file byte-identical.
        bad = valid_doc()
        bad["schema"] = "flow/9.9"
        with self.assertRaises(ValueError):
            flows.save("demo", bad, flows_dir=self.dir)
        self.assertIn("demo-2", p.read_text(encoding="utf-8"))

    def test_save_creates_missing_dir(self):
        nested = self.dir / "flows-sub"
        p = flows.save("demo", valid_doc(), flows_dir=nested)
        self.assertTrue(p.exists())

    def test_save_rejects_invalid_graph_with_reason_list(self):
        bad = valid_doc()
        bad["schema"] = "flow/9.9"  # unsupported schema
        with self.assertRaises(ValueError) as cm:
            flows.save("demo", bad, flows_dir=self.dir)
        msg = str(cm.exception)
        self.assertIn("schema", msg)
        self.assertFalse(list(self.dir.glob("*.flow.json")),
                         "nothing must be written for an invalid graph")

    def test_save_rejects_bad_node_props_via_registry_validator(self):
        # graph_schema.validate alone PASSES this document (props just has to
        # exist and be an object); nodes.validate_props must reject it, so
        # save must call BOTH validators (Task-3 reviewer-noted wiring).
        from flow.server import graph_schema
        bad = valid_doc()
        bad["graph"]["nodes"][1]["props"] = {"rows": "many"}  # not a number
        self.assertEqual(graph_schema.validate(bad), [])
        with self.assertRaises(ValueError) as cm:
            flows.save("demo", bad, flows_dir=self.dir)
        self.assertIn("rows", str(cm.exception))
        self.assertFalse(list(self.dir.glob("*.flow.json")))

    def test_save_rejects_path_traversal_names(self):
        for bad_name in ("../evil", "..", "a/b", "a" + BACKSLASH + "b", ".",
                         "sub/dir/name", ".." + BACKSLASH + "evil"):
            with self.assertRaises(ValueError, msg=repr(bad_name)):
                flows.save(bad_name, valid_doc(), flows_dir=self.dir)
        self.assertFalse(list(self.dir.glob("*.flow.json")))

    def test_save_rejects_non_slug_names(self):
        for bad_name in ("", "UPPER", "has space", "has_underscore",
                         "x" * 65, "caf" + chr(233)):
            with self.assertRaises(ValueError, msg=repr(bad_name)):
                flows.save(bad_name, valid_doc(), flows_dir=self.dir)

    def test_save_accepts_max_length_slug(self):
        name = "a" * 64
        p = flows.save(name, valid_doc(), flows_dir=self.dir)
        self.assertEqual(p.name, name + ".flow.json")

    def test_save_overwrites_existing_flow(self):
        flows.save("demo", valid_doc(), flows_dir=self.dir)
        changed = valid_doc()
        changed["meta"] = {"name": "changed"}
        flows.save("demo", changed, flows_dir=self.dir)
        self.assertEqual(flows.load("demo", flows_dir=self.dir)["meta"]["name"],
                         "changed")

    # --- load ----------------------------------------------------------

    def test_load_returns_parsed_json(self):
        flows.save("demo", valid_doc(), flows_dir=self.dir)
        self.assertEqual(flows.load("demo", flows_dir=self.dir), valid_doc())

    def test_load_missing_raises_clean_filenotfound(self):
        with self.assertRaises(FileNotFoundError) as cm:
            flows.load("nope", flows_dir=self.dir)
        self.assertIn("nope", str(cm.exception))

    def test_load_rejects_path_traversal_names(self):
        with self.assertRaises(ValueError):
            flows.load("../evil", flows_dir=self.dir)
        with self.assertRaises(ValueError):
            flows.load("a/b", flows_dir=self.dir)

    # --- delete_flow -----------------------------------------------------

    def test_delete_flow_removes_file_and_handles_missing(self):
        flows.save("demo", valid_doc(), flows_dir=self.dir)
        target = flows.delete_flow("demo", flows_dir=self.dir)
        self.assertEqual(target, self.dir / "demo.flow.json")
        self.assertFalse(target.exists())
        with self.assertRaises(FileNotFoundError):
            flows.delete_flow("demo", flows_dir=self.dir)

    def test_delete_flow_rejects_non_slug_names(self):
        with self.assertRaises(ValueError):
            flows.delete_flow("../evil", flows_dir=self.dir)

    # --- list_flows ------------------------------------------------------

    def test_list_flows_sorted_slugs(self):
        for name in ("b-flow", "a-flow", "c-flow"):
            flows.save(name, valid_doc(), flows_dir=self.dir)
        self.assertEqual(flows.list_flows(flows_dir=self.dir),
                         ["a-flow", "b-flow", "c-flow"])

    def test_list_flows_empty_when_dir_missing(self):
        self.assertEqual(flows.list_flows(flows_dir=self.dir / "absent"), [])


if __name__ == "__main__":
    unittest.main()
