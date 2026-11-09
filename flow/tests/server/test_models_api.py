"""Tests for the model-graph persistence endpoints (F2 Task 4).

POST /api/models/{name} and GET /api/models are thin wrappers around
pure helpers in flow/server/app.py (save_model / list_models /
validate_model_graph), backed by the shared atomic writer
flow.server.flows.atomic_write. No network calls, no TestClient/httpx
(not in the venv): mirroring test_app_contracts.py, this file exercises
the helpers the routes call and the documented error mapping
(ValueError -> 400 {"errors": [...]}) without HTTP.

Documents are model/0.1 (backward-open like flow/0.1):

    {"format": "model/0.1", "name": str, "meta": {},
     "nodes": [{id, kind, props, position}],
     "edges": [{id, from, to, fromPort, toPort}]}

saved as <slug>.modelgraph.json under flow/models/ (git-tracked like
flow/flows/). Model names share the flow slug rule (single validator,
flows._validate_slug); tests run against a tempfile directory, never
the repo's flow/models/ (no test artifacts are committed).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fastapi.responses import JSONResponse  # noqa: E402

from flow.server import app as app_mod  # noqa: E402

BACKSLASH = chr(92)


def model_node(id="n1", kind="input", position=(0, 0), props=None):
    return {
        "id": id,
        "kind": kind,
        "props": props if props is not None else {},
        "position": {"x": position[0], "y": position[1]},
    }


def model_edge(id="e1", frm="n1", to="n2", frmPort="out", toPort="in"):
    return {"id": id, "from": frm, "fromPort": frmPort,
            "to": to, "toPort": toPort}


def valid_model_doc():
    return {
        "format": "model/0.1",
        "name": "demo-model",
        "meta": {},
        "nodes": [
            model_node("n1", "input",
                       props={"vocab_size": 32768, "ctx": 1024}),
            model_node("n2", "embedding",
                       props={"vocab_size": 32768, "d": 256}),
        ],
        "edges": [model_edge()],
    }


class ModelsApiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # --- save (POST /api/models/{name} backend) -------------------------

    def test_save_model_writes_slug_file_and_json_roundtrips(self):
        p = app_mod.save_model("demo-model", valid_model_doc(),
                               models_dir=self.dir)
        self.assertEqual(p, self.dir / "demo-model.modelgraph.json")
        self.assertTrue(p.exists())
        self.assertEqual(json.loads(p.read_text(encoding="utf-8")),
                         valid_model_doc())

    def test_save_model_overwrite_is_atomic_no_tmp_residue(self):
        """Atomic write via flows.atomic_write: an overwrite replaces the
        file via os.replace (readers never observe a torn document) and
        no .tmp residue survives a successful save."""
        app_mod.save_model("demo", valid_model_doc(), models_dir=self.dir)
        p = self.dir / "demo.modelgraph.json"
        before = p.read_text(encoding="utf-8")
        changed = valid_model_doc()
        changed["name"] = "demo-2"
        app_mod.save_model("demo", changed, models_dir=self.dir)
        self.assertEqual(json.loads(p.read_text(encoding="utf-8")), changed)
        self.assertNotEqual(before, p.read_text(encoding="utf-8"))
        residue = [q.name for q in self.dir.iterdir()
                   if not q.name.endswith(".modelgraph.json")]
        self.assertEqual(residue, [], "no temp files may survive a save")

    def test_save_model_rejects_bad_format_with_honest_400_message(self):
        bad = valid_model_doc()
        bad["format"] = "model/9.9"  # unsupported format
        with self.assertRaises(ValueError) as cm:
            app_mod.save_model("demo", bad, models_dir=self.dir)
        msg = str(cm.exception)
        self.assertIn("format", msg)
        self.assertIn("model/0.1", msg)
        # The route maps this ValueError to the documented 400 errors
        # shape (same flows_validation_response the /api/flows PUT uses).
        resp = app_mod.flows_validation_response(cm.exception)
        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(any("format" in e
                            for e in app_mod.error_list(cm.exception)))
        self.assertFalse(list(self.dir.glob("*.modelgraph.json")),
                         "nothing must be written for an invalid body")

    def test_save_model_rejects_malformed_nodes_and_edges(self):
        bads = (
            {"format": "model/0.1"},                                  # no nodes/edges
            {"format": "model/0.1", "nodes": "nope", "edges": []},    # nodes not a list
            {"format": "model/0.1", "nodes": [], "edges": "nope"},    # edges not a list
            {"format": "model/0.1", "nodes": [{"id": "n1"}],          # node missing keys
             "edges": []},
            {"format": "model/0.1", "nodes": [],                      # edge missing keys
             "edges": [{"id": "e1", "from": "a"}]},
        )
        for bad in bads:
            with self.assertRaises(ValueError, msg=repr(bad)):
                app_mod.save_model("demo", bad, models_dir=self.dir)
        self.assertFalse(list(self.dir.glob("*.modelgraph.json")),
                         "nothing must be written for an invalid body")

    def test_save_model_rejects_path_traversal_names_maps_to_400(self):
        """Traversal names are rejected BEFORE any write, with the shared
        flow slug rule relabeled for the model endpoint."""
        for bad_name in ("../evil", "..", "a/b", "a" + BACKSLASH + "b",
                         "sub/dir/name", ".." + BACKSLASH + "evil"):
            with self.assertRaises(ValueError, msg=repr(bad_name)) as cm:
                app_mod.save_model(bad_name, valid_model_doc(),
                                   models_dir=self.dir)
            msg = str(cm.exception)
            self.assertIn("model name", msg)
            self.assertIn("traversal", msg)
        self.assertFalse(list(self.dir.glob("*.modelgraph.json")))
        # and the route maps the ValueError to the 400 errors shape:
        resp = app_mod.flows_validation_response(
            ValueError("model name: '../evil' is not a valid slug"))
        self.assertEqual(resp.status_code, 400)

    def test_save_model_shares_flow_slug_rule_non_slug_names(self):
        """Same slug rule as flows.save (single validator): lowercase
        [a-z0-9-]{1,64}; anything else is a 400-shaped ValueError."""
        for bad_name in ("", "UPPER", "has space", "has_underscore",
                         "x" * 65, "caf" + chr(233)):
            with self.assertRaises(ValueError, msg=repr(bad_name)) as cm:
                app_mod.save_model(bad_name, valid_model_doc(),
                                   models_dir=self.dir)
            self.assertIn("slug", str(cm.exception))

    def test_save_model_accepts_max_length_slug(self):
        name = "a" * 64
        p = app_mod.save_model(name, valid_model_doc(), models_dir=self.dir)
        self.assertEqual(p.name, name + ".modelgraph.json")

    # --- list (GET /api/models backend) ---------------------------------

    def test_list_models_returns_sorted_saved_slugs(self):
        for name in ("b-model", "a-model", "c-model"):
            app_mod.save_model(name, valid_model_doc(), models_dir=self.dir)
        # Exact shape the GET /api/models route returns (the /api/flows
        # list shape — sorted slugs — wrapped in {"models": ...}).
        self.assertEqual({"models": app_mod.list_models(self.dir)},
                         {"models": ["a-model", "b-model", "c-model"]})

    def test_list_models_empty_when_dir_missing(self):
        self.assertEqual(app_mod.list_models(self.dir / "absent"), [])

    def test_list_models_ignores_non_modelgraph_files(self):
        app_mod.save_model("demo", valid_model_doc(), models_dir=self.dir)
        (self.dir / "notes.txt").write_text("not a model", encoding="utf-8")
        self.assertEqual(app_mod.list_models(self.dir), ["demo"])

    # --- route registration ----------------------------------------------

    def test_models_routes_registered_on_app(self):
        routes = {r.path for r in app_mod.app.routes}
        self.assertIn("/api/models", routes)
        self.assertIn("/api/models/{name}", routes)

    def test_read_model_returns_saved_doc_and_404_message(self):
        app_mod.save_model("demo-get", valid_model_doc(), models_dir=self.dir)
        doc = app_mod.read_model("demo-get", models_dir=self.dir)
        self.assertEqual(doc["format"], "model/0.1")
        self.assertEqual(doc["name"], "demo-model")
        missing = app_mod.read_model("absent", models_dir=self.dir)
        self.assertEqual(missing.status_code, 404)
        self.assertIn("model 'absent' does not exist", missing.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
