"""Task 6 TDD tests: flow/server/runner.py (RED first).

Fake scripts live only in tempfile dirs; the runner gets an injected
script path + interpreter so no test ever launches a real GPU job.
"""

import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from flow.server import runner

REPO_ROOT = Path(__file__).resolve().parents[3]

# CPython used to run the fake scripts (same venv the runner defaults to,
# but the test passes it explicitly so defaults may evolve independently).
INTERP = sys.executable

FAKE_SCRIPT = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    cfg = Path(sys.argv[sys.argv.index("--config") + 1])
    out_dir = None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("output_dir:"):
            out_dir = Path(s[len("output_dir:"):].strip())
            break
    assert out_dir is not None, "no output_dir in config"
    mode = os.environ.get("FAKE_MODE", "stop")
    if mode == "exit":
        print("first"); print("second"); sys.exit(3)
    if mode == "spam":
        for i in range(1, 251):
            print("line %d" % i, flush=True)
        sys.exit(0)
    # stop mode: print a few lines, then poll for the STOP flag
    for i in range(3):
        print("working %d" % i, flush=True)
    deadline = time.time() + 8
    while time.time() < deadline:
        if (out_dir / "STOP").is_file():
            print("saw stop flag", flush=True)
            sys.exit(0)
        time.sleep(0.05)
    print("never saw flag", flush=True)
    sys.exit(4)
''')


class _TempBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.script = self.base / "run_custom.py"
        self.script.write_text(FAKE_SCRIPT, encoding="utf-8")
        self.out_dir = self.base / "out"
        self.out_dir.mkdir()
        self.config = self.base / "flow_test.yaml"
        cfg = ("name: fake\ntrain:\n"
                   "  output_dir: %s\n") % str(self.out_dir).replace(os.sep, "/")
        self.config.write_text(cfg, encoding="utf-8")

    def tearDown(self):
        runner.wait_finished(timeout=10)
        self._tmp.cleanup()

    def start(self, mode="stop", **kw):
        kw.setdefault("script_path", str(self.script))
        if mode is not None:
            kw.setdefault("env", dict(os.environ, FAKE_MODE=mode))
        kw.setdefault("interpreter", INTERP)
        return runner.start(self.config, **kw)


class TestRunner(_TempBase):
    def test_missing_config_raises_typed_error(self):
        with self.assertRaises(FileNotFoundError) as cm:
            runner.start(self.base / "nope.yaml",
                         script_path=str(self.script), interpreter=INTERP)
        self.assertIn("nope.yaml", str(cm.exception))

    def test_second_start_rejected_while_running(self):
        self.start()
        try:
            with self.assertRaises(ValueError) as cm:
                self.start()
            self.assertIn("GPU job already running", str(cm.exception))
        finally:
            self.wait_exit()

    def test_completion_captures_exit_code(self):
        self.start(mode="exit")
        st = self.wait_exit()
        self.assertFalse(st["running"])
        self.assertEqual(st["exit_code"], 3)
        self.assertIn("second", st["tail"][-1])
        self.assertTrue(st["started_at"] > 0)

    def test_tail_trimmed_to_maxlen(self):
        self.start(mode="spam")
        st = self.wait_exit()
        self.assertEqual(len(st["tail"]), runner.TAIL_MAXLEN)
        self.assertEqual(st["tail"][-1], "line 250")
        self.assertEqual(st["tail"][0], "line 51")

    def test_stop_flag_written_and_script_exits_on_it(self):
        pid = self.start()
        st_before = runner.status()
        self.assertTrue(st_before["running"])
        self.assertFalse((self.out_dir / "STOP").exists())
        flag = runner.request_stop()
        self.assertEqual(Path(flag), self.out_dir / "STOP")
        self.assertTrue((self.out_dir / "STOP").is_file())
        self.assertTrue((self.out_dir / "STOP").read_text(encoding="utf-8"))
        st = self.wait_exit()
        self.assertEqual(st["exit_code"], 0)
        self.assertIn("saw stop flag", "\n".join(st["tail"]))
        self.assertGreater(st["exit_at"], st_before["started_at"])
        _ = pid

    def test_status_none_safe_and_request_stop_without_job(self):
        st = runner.status()
        self.assertFalse(st["running"])
        self.assertIsInstance(st["tail"], list)
        # tail is a copy: mutating it must not corrupt the runner state
        st["tail"].append("sentinel")
        self.assertNotIn("sentinel", runner.status()["tail"])
        with self.assertRaises(RuntimeError):
            runner.request_stop()  # no live job (webui: "no live job to stop")

    def wait_exit(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = runner.status()
            if not st["running"]:
                return st
            time.sleep(0.05)
        self.fail("runner job did not finish within %ss" % timeout)
        return None


if __name__ == "__main__":
    unittest.main()