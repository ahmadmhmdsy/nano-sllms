"""Task 5 tests: flow.server.config_gen — graph -> configs/flow_<slug>.yaml.

TDD step 1: these tests FAIL before flow/server/config_gen.py exists.
Key names validated against the REAL pipeline configs by loading
configs/smoke.yaml (not invented keys). Tests never write into the repo's
configs/ or flow/flows/: out_dir is injected as a tempfile directory.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from flow.server import config_gen, graph_schema, nodes  # noqa: E402


def dataset_node(nid="n1", label="TheGamingMahi/TinyCode"):
    node = {"id": nid, "kind": "dataset", "props": {},
            "position": {"x": 0, "y": 0}}
    if label is not None:
        node["label"] = label
    return node


def node(nid, kind, props, x=0.0):
    return {"id": nid, "kind": kind, "props": props,
            "position": {"x": x, "y": 0}}


def linear_chain():
    """dataset -> prepare -> tokenize -> train chain with full props."""
    return {
        "schema": "flow/0.1",
        "meta": {"name": "my-run"},
        "graph": {
            "nodes": [
                dataset_node("n1", "TheGamingMahi/TinyCode"),
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


def real_train_keys():
    """Real train-block key names parsed from configs/smoke.yaml."""
    with open(REPO / "configs" / "smoke.yaml", "r", encoding="utf-8") as f:
        smoke = yaml.safe_load(f)
    return set(smoke["train"].keys()), set(smoke["data"].keys())


class ConfigGenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    # --- happy path ----------------------------------------------------
    def test_generate_returns_path_writes_flow_yaml(self):
        path = config_gen.generate(linear_chain(), out_dir=self.out_dir)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(Path(path).name, "flow_my-run.yaml")
        self.assertEqual(Path(path).parent.name, Path(self.out_dir).name)

    def test_yaml_has_real_key_names_from_smoke_yaml(self):
        cfg = config_gen.build_config(linear_chain())
        train_keys, data_keys = real_train_keys()
        self.assertEqual(set(cfg["train"].keys()), train_keys)
        self.assertEqual(set(cfg["data"].keys()), data_keys)
        self.assertEqual(set(cfg.keys()), {"name", "tokenizer", "model",
                                           "data", "train", "eval"})
        # eval block mirrors the shipped key names
        self.assertIn("max_new_tokens", cfg["eval"])

    def test_values_map_from_node_props(self):
        cfg = config_gen.build_config(linear_chain())
        data = cfg["data"]
        train = cfg["train"]
        model = cfg["model"]
        # dataset name from the dataset node label
        self.assertEqual(data["dataset_candidates"],
                         [{"name": "TheGamingMahi/TinyCode"}])
        self.assertEqual(data["rows"], 2000)
        self.assertEqual(data["val_fraction"], 0.02)
        self.assertEqual(data["min_chars"], 60)
        # tokenize props
        self.assertEqual(cfg["tokenizer"]["vocab_size"], 32768)
        self.assertEqual(model["ctx"], 256)
        # train preset/lr_preset strings -> real config values
        self.assertEqual(train["max_steps"], 200)
        self.assertEqual(train["lr"], 4.0e-4)
        self.assertEqual(model["layers"], 4)
        self.assertEqual(train["accum"], 32)
        # derived dirs named after the run
        self.assertEqual(data["tokens_dir"], "data/my-run/tokens")
        self.assertEqual(train["output_dir"], "runs/my-run")

    def test_writes_yaml_equivalent_roundtrip(self):
        path = config_gen.generate(linear_chain(), out_dir=self.out_dir)
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded, config_gen.build_config(linear_chain()))

    def test_generated_file_never_overwrites_shipped_configs(self):
        # Strong form: the test-written file is resolved BELOW the injected
        # tempfile dir and NOT below the repo's configs/ directory.
        path = config_gen.generate(linear_chain(), out_dir=self.out_dir)
        resolved = Path(path).resolve()
        repo_configs = config_gen.DEFAULT_CONFIGS_DIR.resolve()
        self.assertTrue(resolved.is_relative_to(Path(self.tmp.name).resolve()))
        self.assertFalse(resolved.is_relative_to(repo_configs))
        self.assertFalse(resolved == repo_configs)
        # A real attempt at a shipped run name is blocked by the run-name
        # reservation (build_config ValueError) - and the repo configs/ dir
        # stays untouched because generate() writes only after validation.
        shipped = repo_configs / "flow_smoke.yaml"
        self.assertFalse(shipped.exists())  # repo-readonly evidence base
        g = linear_chain()
        g["meta"]["name"] = "smoke"
        with self.assertRaises(ValueError) as cm:
            config_gen.generate(g, out_dir=config_gen.DEFAULT_CONFIGS_DIR)
        self.assertIn("reserved shipped-config name", str(cm.exception))
        self.assertFalse(shipped.exists())  # still untouched after the try

    # --- validation reuse ------------------------------------------------
    def test_invalid_graph_rejected_with_graph_schema_errors(self):
        g = linear_chain()
        g["schema"] = "flow/9.9"
        with self.assertRaises(ValueError) as cm:
            config_gen.generate(g, out_dir=self.out_dir)
        self.assertIn("schema", str(cm.exception))

    def test_unknown_props_rejected(self):
        g = linear_chain()
        g["graph"]["nodes"][3]["props"]["epochs"] = 5  # unknown for train
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("epochs", str(cm.exception))

    def test_nothing_written_on_invalid_graph(self):
        g = linear_chain()
        g["graph"]["nodes"][1]["kind"] = "nope"
        try:
            config_gen.generate(g, out_dir=self.out_dir)
        except ValueError:
            pass
        self.assertEqual(os.listdir(self.out_dir), [])

    # --- branching / unsupported graphs -------------------------------------
    def test_branching_graph_rejected_cleanly(self):
        g = linear_chain()
        g["graph"]["nodes"].append(
            node("n5", "tokenize", {"seq_len": 512, "vocab": 32768}))
        g["graph"]["edges"].append(
            {"id": "e4", "from": "n1", "to": "n5",
             "fromPort": "cleaned", "toPort": "cleaned-dir"})
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("linear chains only", str(cm.exception))

    def test_diamond_graph_rejected(self):
        g = {
            "schema": "flow/0.1",
            "meta": {"name": "diamond"},
            "graph": {
                "nodes": [
                    dataset_node("a"), dataset_node("b"),
                    node("c", "prepare", {"rows": 10, "val_fraction": 0.02,
                                          "min_chars": 1}),
                ],
                "edges": [
                    {"id": "e1", "from": "a", "to": "c",
                     "fromPort": "cleaned", "toPort": "cleaned-dir"},
                    {"id": "e2", "from": "b", "to": "c",
                     "fromPort": "cleaned", "toPort": "cleaned-dir"},
                ],
            },
        }
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("linear chains only", str(cm.exception))

    # --- missing source values -------------------------------------------
    def test_train_without_dataset_chain_lists_missing_input(self):
        g = {
            "schema": "flow/0.1",
            "meta": {"name": "lonely"},
            "graph": {
                "nodes": [node("t1", "train", {"preset": "Small (~12M, smoke)",
                                               "steps": 10,
                                               "lr_preset": "Pretrain 4e-4"})],
                "edges": [],
            },
        }
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("shard-dir", str(cm.exception))
        self.assertIn("t1", str(cm.exception))

    def test_dataset_without_name_lists_missing_input(self):
        g = linear_chain()
        g["graph"]["nodes"][0].pop("label")
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("dataset name", str(cm.exception))

    def test_train_missing_steps_lists_missing_input(self):
        g = linear_chain()
        g["graph"]["nodes"][3]["props"].pop("steps")
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("steps", str(cm.exception))

    def test_unknown_preset_string_rejected(self):
        g = linear_chain()
        g["graph"]["nodes"][3]["props"]["preset"] = "turbo"
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("preset", str(cm.exception))

    def test_unknown_lr_preset_string_rejected(self):
        g = linear_chain()
        g["graph"]["nodes"][3]["props"]["lr_preset"] = "s999"
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("lr_preset", str(cm.exception))

    # --- name handling ----------------------------------------------------
    def test_non_slug_name_rejected(self):
        g = linear_chain()
        g["meta"]["name"] = "../evil"
        with self.assertRaises(ValueError) as cm:
            config_gen.generate(g, out_dir=self.out_dir)
        self.assertIn("slug", str(cm.exception))
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_preset_and_lr_preset_are_strings_in_props(self):
        g = linear_chain()
        g["graph"]["nodes"][3]["props"]["preset"] = 4  # number, not string
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("preset", str(cm.exception))


class ReviewFixTests(unittest.TestCase):
    """Post-review fixes: reserved run names, rows cap, ValueError-only
    failure paths, generated-path safety, webui parity, dataset-name rule.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def _named(self, name):
        g = linear_chain()
        g["meta"]["name"] = name
        return g

    # --- IMPORTANT-1: reserved run names ---------------------------------
    def test_reserved_run_names_rejected(self):
        for name in sorted(config_gen.RESERVED_RUN_NAMES):
            self.assertEqual(name, name.lower())  # slugs are lowercase
            with self.assertRaises(ValueError) as cm:
                config_gen.generate(self._named(name), out_dir=self.out_dir)
            msg = str(cm.exception)
            self.assertIn("reserved shipped-config name", msg)
            self.assertIn("runs/%s" % name, msg)   # collision explanation
            self.assertIn("data/%s" % name, msg)
            self.assertEqual(os.listdir(self.out_dir), [])  # nothing written

    def test_benign_run_name_still_allowed(self):
        path = config_gen.generate(self._named("fresh-run-42"),
                                   out_dir=self.out_dir)
        self.assertTrue(os.path.isfile(path))

    # --- IMPORTANT-2: rows hard cap --------------------------------------
    def test_rows_cap_boundary(self):
        g = linear_chain()
        g["graph"]["nodes"][1]["props"]["rows"] = config_gen.MAX_ROWS
        cfg = config_gen.build_config(g)
        self.assertEqual(cfg["data"]["rows"], config_gen.MAX_ROWS)

    def test_rows_above_cap_rejected(self):
        g = linear_chain()
        g["graph"]["nodes"][1]["props"]["rows"] = config_gen.MAX_ROWS + 1
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("hard cap %d" % config_gen.MAX_ROWS, str(cm.exception))
        self.assertIn("(net time + disk)", str(cm.exception))

    def test_rows_cap_matches_webui_constant(self):
        # Source-parsed (see parity test for why webui.app is not imported).
        self.assertEqual(config_gen.MAX_ROWS,
                         ReviewFixTests.webui_assign("MAX_ROWS"))

    # --- MINOR-3: node/props failures raise ValueError, never KeyError ----
    def test_missing_props_dict_raises_value_error_not_keyerror(self):
        g = linear_chain()
        g["graph"]["nodes"][1].pop("props")   # prepare node, no props key
        with self.assertRaises(ValueError) as cm:
            config_gen.build_config(g)
        self.assertIn("prepare", str(cm.exception))
        self.assertIn("props", str(cm.exception))
        self.assertNotIn("KeyError", str(cm.exception))

    # --- MINOR-5: PRESETS / LR_PRESETS parity with webui ------------------
    @staticmethod
    def webui_assign(name):
        """Parse webui/app.py with ast and evaluate the module-level assign
        to `name`. webui.app is deliberately NOT imported: importing it
        builds the full Gradio demo (heavy deps + module side effects), so
        the robust drift check is source parsing + evaluation of the single
        assignment expression (no exec of statements, no gradio import).
        """
        import ast
        src = (REPO / "webui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(src, filename="webui/app.py")
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name
                    for t in stmt.targets):
                code = compile(ast.Expression(stmt.value),
                               "<webui-%s>" % name, "eval")
                return eval(code, {"__builtins__": {}}, {"dict": dict})
        raise AssertionError("webui/app.py has no top-level %r" % name)

    def test_presets_parity_with_webui_dicts(self):
        self.assertEqual(config_gen.PRESETS, self.webui_assign("PRESETS"))
        self.assertEqual(config_gen.LR_PRESETS,
                         self.webui_assign("LR_PRESETS"))

    def test_reserved_names_parity_with_webui_build_config(self):
        # The webui build_config guard tuple is the source of truth.
        self.assertEqual(
            sorted(config_gen.RESERVED_RUN_NAMES),
            ["custom_example", "pilot", "sft_t1", "smoke", "target"])
        src = (REPO / "webui" / "app.py").read_text(encoding="utf-8")
        self.assertIn('("smoke", "pilot", "target", "sft_t1", "custom_example")',
                      src)  # build_config guard tuple verbatim

    # --- MINOR-6: dataset label as HF-name shape -------------------------
    def test_valid_dataset_names_accepted(self):
        for label in ("tinystories", "TheGamingMahi/TinyCode",
                      "my_data.v2/name-1", "org/name", "0123"):
            g = linear_chain()
            g["graph"]["nodes"][0]["label"] = label
            cfg = config_gen.build_config(g)
            self.assertEqual(cfg["data"]["dataset_candidates"],
                             [{"name": label}])

    def test_invalid_dataset_names_rejected_naming_the_node(self):
        for label in ("", "   ", "has space", "row$1", "org//name",
                      "org/", "/name", "org/name/extra", "r\u00e9sum\u00e9"):
            g = linear_chain()
            g["graph"]["nodes"][0]["label"] = label
            with self.assertRaises(ValueError) as cm:
                config_gen.build_config(g)
            self.assertIn("dataset name", str(cm.exception))
            self.assertIn("n1", str(cm.exception))  # node id named


if __name__ == "__main__":
    unittest.main()
