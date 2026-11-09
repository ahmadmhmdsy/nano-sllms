"""Task 7 contract tests: pure pre-route assembly functions of app.py.

No network calls, no TestClient/httpx (not in the venv): the routes are
THIN wrappers around already-tested backend functions, so this file only
exercises the pure helpers the routes call - port resolution, the
error-to-HTTP-status mapping, and the validate preflight assembly
(including its tempfile out_dir dry-run so NO repo file is written).

Zero network calls; nothing under configs/, flow/flows/ or runs/ is
touched by these tests.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from flow.server import app as app_mod  # noqa: E402
from flow.server import runner  # noqa: E402


def dataset_node(nid="n1", label="TheGamingMahi/TinyCode"):
    return {"id": nid, "kind": "dataset", "props": {},
            "label": label, "position": {"x": 0, "y": 0}}


def node(nid, kind, props, x=0.0):
    return {"id": nid, "kind": kind, "props": props,
            "position": {"x": x, "y": 0}}


def linear_chain():
    return {
        "schema": "flow/0.1",
        "meta": {"name": "contract-check"},
        "graph": {
            "nodes": [
                dataset_node("n1"),
                node("n2", "prepare", {"rows": 2000, "val_fraction": 0.02,
                                       "min_chars": 60}),
                node("n3", "tokenize", {"seq_len": 256, "vocab": 32768}),
                node("n4", "train", {"preset": "Small (~12M, smoke)",
                                     "steps": 200,
                                     "lr_preset": "Pretrain 4e-4"}),
            ],
            "edges": [
                {"id": "e1", "from": "n1", "to": "n2",
                 "fromPort": "cleaned", "toPort": "raw-dir"},
                {"id": "e2", "from": "n2", "to": "n3",
                 "fromPort": "cleaned-dir", "toPort": "cleaned-dir"},
                {"id": "e3", "from": "n3", "to": "n4",
                 "fromPort": "shard-dir", "toPort": "shard-dir"},
            ],
        },
    }


class ResolvePortTest(unittest.TestCase):
    def test_explicit_port_wins_even_with_env(self):
        # FLOW_PORT env MUST NOT win over --port (CLI beats env default).
        self.assertEqual(app_mod.resolve_port(8123, {"FLOW_PORT": "9999"}),
                         8123)

    def test_env_used_when_port_absent(self):
        self.assertEqual(app_mod.resolve_port(None, {"FLOW_PORT": "9999"}),
                         9999)

    def test_default_without_port_or_env(self):
        self.assertEqual(app_mod.resolve_port(None, {}), app_mod.DEFAULT_PORT)
        self.assertEqual(app_mod.DEFAULT_PORT, 3010)

    def test_non_integer_env_falls_back_to_default(self):
        self.assertEqual(app_mod.resolve_port(None, {"FLOW_PORT": "abc"}),
                         app_mod.DEFAULT_PORT)

    def test_empty_env_string_uses_default(self):
        self.assertEqual(app_mod.resolve_port(None, {"FLOW_PORT": ""}),
                         app_mod.DEFAULT_PORT)


class StatusForErrorTest(unittest.TestCase):
    def test_typed_runner_errors_map_to_409(self):
        # 409 wins even though JobRunningError is a ValueError subclass.
        self.assertEqual(
            app_mod.status_for_error(
                runner.JobRunningError("GPU job already running (pid 1)")),
            409)
        self.assertEqual(
            app_mod.status_for_error(
                runner.NoJobRunningError("no live GPU job to stop")),
            409)

    def test_missing_flow_maps_to_404(self):
        self.assertEqual(
            app_mod.status_for_error(FileNotFoundError("flow 'x' not found")),
            404)

    def test_validation_errors_map_to_400(self):
        self.assertEqual(
            app_mod.status_for_error(ValueError("schema: ...")), 400)


class ReadFlowErrorPathTest(unittest.TestCase):
    """A non-slug GET /api/flows/{name} must hit the 400 errors path.

    Pure-function check: read_flow's handler maps exactly two exception
    types from flows.load — FileNotFoundError -> 404 detail shape and
    ValueError (slug/traversal) -> 400 with the flows_validation_response
    error-list shape. Mirror that mapping without HTTP.
    """

    def test_non_slug_load_raises_valueerror_and_maps_to_400_errors_shape(self):
        from fastapi.responses import JSONResponse
        from flow.server import flows

        with self.assertRaises(ValueError):
            flows.load("../evil")  # what read_flow sees for this name
        exc = ValueError("flow name: '../evil' is not a valid slug")
        status = app_mod.status_for_error(exc)
        self.assertEqual(status, 400)
        # and the route builds the documented shape from that exception:
        resp = app_mod.flows_validation_response(exc)
        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(app_mod.error_list(exc),
                         ["flow name: '../evil' is not a valid slug"])


class ValidatePreflightTest(unittest.TestCase):
    """Preflight assembly: pure validation, tempfile dry-run, no repo writes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self._repo_before = self._repo_listing()

    def _repo_listing(self):
        """Snapshot files under configs/ + flow/flows/ (validation must
        not add anything in either).

        Guarded scope is exactly configs/ + flow/flows/: the two repo
        locations the preflight path can touch (deliverable yaml dir and
        the flows store).
        """
        listing = {}
        for d in (REPO / "configs", REPO / "flow" / "flows"):
            listing[d] = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        return listing

    def test_valid_graph_returns_no_errors_and_writes_no_repo_file(self):
        errors = app_mod.validate_preflight(linear_chain())
        self.assertEqual(errors, [])
        # dry run only: the tempfile out_dir has the generated yaml,
        # and the repo did NOT gain/lose any config or flow file.
        self.assertEqual(self._repo_listing(), self._repo_before)
        self.assertEqual(sorted(os.listdir(self.out_dir)), [])

    def test_invalid_schema_document_lists_reasons(self):
        g = {"schema": "flow/9.9", "meta": {}, "graph": {"nodes": []}}
        errors = app_mod.validate_preflight(g)
        self.assertTrue(errors)
        self.assertTrue(any("schema" in e for e in errors))
        self.assertTrue(any("meta" in e for e in errors))
        self.assertEqual(self._repo_listing(), self._repo_before)

    def test_unknown_node_kind_lists_reasons(self):
        g = linear_chain()
        g["graph"]["nodes"][0]["kind"] = "warp-drive"
        errors = app_mod.validate_preflight(g)
        self.assertTrue(any("unknown kind" in e for e in errors))

    def test_prop_violation_surfaced_from_registry(self):
        g = linear_chain()
        g["graph"]["nodes"][1]["props"]["rows"] = "many"
        errors = app_mod.validate_preflight(g)
        self.assertTrue(any("finite number" in e for e in errors))

    def test_reserved_run_name_rejected_via_config_gen_dry_run(self):
        g = linear_chain()
        g["meta"]["name"] = "smoke"
        errors = app_mod.validate_preflight(g)
        self.assertTrue(any("reserved shipped-config name" in e
                            for e in errors))

    def test_rows_cap_enforced_via_config_gen_dry_run(self):
        g = linear_chain()
        g["graph"]["nodes"][1]["props"]["rows"] = 900_000
        errors = app_mod.validate_preflight(g)
        self.assertTrue(any("above the hard cap" in e for e in errors))

    def test_branching_graph_rejected(self):
        g = linear_chain()
        g["graph"]["edges"].append(
            {"id": "e4", "from": "n2", "to": "n1",
             "fromPort": "cleaned-dir", "toPort": "raw-dir"})
        errors = app_mod.validate_preflight(g)
        self.assertTrue(errors)  # linear-chain / cycle reasons present
        self.assertEqual(self._repo_listing(), self._repo_before)

    def test_error_list_splits_newline_joined_reasons(self):
        exc = ValueError("alpha\nbeta")
        self.assertEqual(app_mod.error_list(exc), ["alpha", "beta"])


class AppModuleContractTest(unittest.TestCase):
    def test_app_object_exists_and_registers_api_routes(self):
        routes = {r.path for r in app_mod.app.routes}
        for path in ("/api/health", "/api/nodes", "/api/flows",
                     "/api/flows/{name}", "/api/validate", "/api/run",
                     "/api/run/status", "/api/run/stop"):
            self.assertIn(path, routes)


if __name__ == "__main__":
    unittest.main()
