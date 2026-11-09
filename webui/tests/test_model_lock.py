"""M4.1: _MODEL race-condition tests for webui/app.py.

The chat service keeps the loaded model in the module global _MODEL while
unload_model / load_model / the train-launch guards read and write it from
Gradio's event threads. These tests pin the locking discipline without a
GPU or a real checkpoint: torch, transformers and the HF loaders are
replaced with fakes, so what gets exercised is the lock, not the ML stack.

Run from the repo root:
    .venv/Scripts/python.exe -W error::SyntaxWarning -m unittest discover -s webui/tests -t .
"""
import contextlib
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

WEBUI_DIR = Path(__file__).resolve().parents[1]
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))

import app as app_module  # noqa: E402  (webui/ put on sys.path above)


# ------------------------------------------------------------------- fakes

class _FakeTensor:
    """Stands in for the HF batch: ids_full[:, -room:].to(device)."""

    def __init__(self, n_tokens):
        self.shape = (1, n_tokens)

    def __getitem__(self, sl):
        return self

    def to(self, device):
        return self


class _FakeBatch:
    def __init__(self, n_tokens):
        self.input_ids = _FakeTensor(n_tokens)


class _FakeTok:
    """Stands in for the HF tokenizer; records prompts for assertions."""

    eos_token_id = 0

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, return_tensors=None):
        self.prompts.append(prompt)
        return _FakeBatch(5)


class _FakeModel:
    """Stands in for the HF causal LM: .to().eval() in load, .generate in chat."""

    def __init__(self):
        self.generate_calls = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1


class _FakeStreamer:
    """Stands in for TextIteratorStreamer: yields a few tokens, then ends."""

    def __init__(self, tok, **kwargs):
        self.tok = tok
        self.ended = False

    def __iter__(self):
        for piece in ("a", "b", "c"):
            yield piece

    def end(self):
        self.ended = True


def _fake_torch():
    """A torch stand-in: no CUDA, no_grad is a null context."""
    return types.SimpleNamespace(
        no_grad=lambda: contextlib.nullcontext(),
        cuda=types.SimpleNamespace(is_available=lambda: False,
                                   empty_cache=lambda: None))


def _valid_entry(tok=None, model=None):
    """A well-formed _MODEL dict with a stubbed tok/model (no torch/HF)."""
    return {"label": "fake/ckpt", "path": Path("/tmp/fake-ckpt"),
            "tok": tok or _FakeTok(), "model": model or _FakeModel(),
            "device": "cpu", "ctx": 1024, "cfg": {}, "note": "test"}


@contextlib.contextmanager
def _fake_ml_stack():
    """Replace torch/transformers with fakes (no GPU, no HF download)."""
    fake_tok = _FakeTok()
    fake_model = _FakeModel()
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=mock.Mock(from_pretrained=mock.Mock(return_value=fake_tok)),
        AutoModelForCausalLM=mock.Mock(from_pretrained=mock.Mock(
            return_value=fake_model)),
        StoppingCriteriaList=mock.Mock(),
        TextIteratorStreamer=_FakeStreamer)
    with mock.patch.dict(sys.modules, {"torch": _fake_torch(),
                                       "transformers": fake_transformers}):
        yield types.SimpleNamespace(tok=fake_tok, model=fake_model)


class ModelLockTest(unittest.TestCase):

    def setUp(self):
        self._old_model = app_module._MODEL
        app_module._MODEL = None

    def tearDown(self):
        app_module._MODEL = self._old_model

    # ----------------------------------------------------------- behaviour

    def test_chat_stream_placeholder_when_no_model(self):
        pieces = list(app_module.chat_stream("hello", [], False, False, 128,
                                             0.8, 0.95, 50))
        self.assertEqual(pieces,
                         ["_(load a checkpoint above, then send a code prefix)_"])

    def test_chat_stream_placeholder_when_empty_message(self):
        app_module._MODEL = _valid_entry()
        pieces = list(app_module.chat_stream("   ", [], False, False, 128,
                                             0.8, 0.95, 50))
        self.assertEqual(pieces,
                         ["_(load a checkpoint above, then send a code prefix)_"])

    def test_chat_stream_survives_unload_mid_generation(self):
        """Brief Step 5: unload while the stream is yielding tokens."""
        with _fake_ml_stack() as fakes:
            app_module._MODEL = _valid_entry(fakes.tok, fakes.model)
            gen = app_module.chat_stream("hello", [], False, False, 128,
                                         0.8, 0.95, 50)
            first = next(gen)  # generation is underway
            # another thread unloads while tokens are still streaming
            t = threading.Thread(target=app_module.unload_model)
            t.start()
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
            pieces = [first] + list(gen)
            deadline = time.time() + 5
            while fakes.model.generate_calls < 1 and time.time() < deadline:
                time.sleep(0.01)
        self.assertEqual(pieces, ["a", "ab", "abc"])
        self.assertEqual(fakes.model.generate_calls, 1)
        self.assertIsNone(app_module._MODEL)

    def test_chat_stream_uses_copied_cfg_for_template(self):
        """cfg is copied under the lock too: the instruct template is read
        from the local copy, not from _MODEL, during generation."""
        with _fake_ml_stack() as fakes:
            entry = _valid_entry(fakes.tok, fakes.model)
            entry["cfg"] = {"data": {"template": "T:{instruction}"}}
            app_module._MODEL = entry
            pieces = list(app_module.chat_stream("do it", [], True, False, 128,
                                                0.8, 0.95, 50))
        self.assertEqual(pieces, ["a", "ab", "abc"])
        self.assertEqual(fakes.tok.prompts, ["T:do it"])

    def test_chat_stream_copies_model_before_use(self):
        """Deterministic regression test for the _MODEL race.

        The fake transformers module sleeps on import. In the unfixed code
        that import sits BETWEEN the _MODEL None-check and the reference
        copy, so an unload landing during the import makes the copy read
        None and crash the stream. The fixed code copies under the lock
        before importing, so the same unload is harmless.
        """
        sentinel_dir = Path(tempfile.mkdtemp())
        sentinel = sentinel_dir / "importing"
        (sentinel_dir / "transformers.py").write_text(
            "import pathlib, time\n"
            "pathlib.Path(" + repr(str(sentinel)) + ").write_text('1')\n"
            "time.sleep(0.4)\n"
            "class StoppingCriteriaList:\n"
            "    def __init__(self, *a, **k): pass\n"
            "class TextIteratorStreamer:\n"
            "    def __init__(self, tok, **k): self.tok = tok\n"
            "    def __iter__(self):\n"
            "        yield 'a'\n"
            "        yield 'b'\n"
            "        yield 'c'\n"
            "    def end(self): pass\n",
            encoding="utf-8")
        old_path = list(sys.path)
        real_transformers = sys.modules.get("transformers")
        sys.modules.pop("transformers", None)
        sys.path.insert(0, str(sentinel_dir))
        try:
            with mock.patch.dict(sys.modules, {"torch": _fake_torch()}):
                app_module._MODEL = _valid_entry()
                gen = app_module.chat_stream("hello", [], False, False, 128,
                                             0.8, 0.95, 50)
                outcome = {}

                def _first_next():
                    try:
                        outcome["first"] = next(gen)
                    except BaseException as e:  # noqa: BLE001
                        outcome["error"] = e

                t = threading.Thread(target=_first_next)
                t.start()
                deadline = time.time() + 10
                while not sentinel.is_file() and time.time() < deadline:
                    time.sleep(0.01)
                self.assertTrue(sentinel.is_file(),
                                "fake transformers import never started")
                # unload while chat_stream is inside its import window
                app_module.unload_model()
                t.join(timeout=10)
                self.assertFalse(t.is_alive(), "first next() never finished")
                self.assertNotIn(
                    "error", outcome,
                    "chat_stream crashed on concurrent unload: "
                    + repr(outcome.get("error")))
                pieces = [outcome["first"]] + list(gen)
        finally:
            sys.path[:] = old_path
            sys.modules.pop("transformers", None)
            if real_transformers is not None:
                sys.modules["transformers"] = real_transformers
        self.assertEqual(pieces, ["a", "ab", "abc"])
        self.assertIsNone(app_module._MODEL)

    # ----------------------------------------------------------- concurrency

    def test_concurrent_load_unload_no_crash(self):
        """Brief Step 5: concurrent load + unload neither crashes nor tears
        _MODEL. A stress test - the deterministic crash-catcher is
        test_chat_stream_copies_model_before_use."""
        errors = []
        stop = threading.Event()

        def load_loop():
            while not stop.is_set():
                try:
                    app_module.load_model("fake/ckpt")
                except Exception as e:  # noqa: BLE001
                    errors.append(("load", repr(e)))

        def unload_loop():
            while not stop.is_set():
                try:
                    app_module.unload_model()
                except Exception as e:  # noqa: BLE001
                    errors.append(("unload", repr(e)))

        def status_loop():
            while not stop.is_set():
                try:
                    app_module._model_status_md()
                except Exception as e:  # noqa: BLE001
                    errors.append(("status", repr(e)))

        with _fake_ml_stack(), \
                mock.patch.object(app_module, "_ckpts",
                                  return_value={"fake/ckpt":
                                                Path("/tmp/fake-ckpt")}), \
                mock.patch.object(app_module.status, "load_phase_cfg",
                                  return_value={"model": {"ctx": 1024,
                                                           "layers": 4},
                                                "data": {}}), \
                mock.patch.object(app_module, "_weights_mb", return_value=100.0), \
                mock.patch.object(app_module.run_custom, "gpu_compute_pids",
                                  return_value=[]), \
                mock.patch.object(app_module, "_free_vram_mib",
                                  return_value=100000):
            threads = [threading.Thread(target=load_loop) for _ in range(2)]
            threads += [threading.Thread(target=unload_loop) for _ in range(2)]
            threads.append(threading.Thread(target=status_loop))
            for t in threads:
                t.start()
            time.sleep(1.0)
            stop.set()
            for t in threads:
                t.join(timeout=10)
            self.assertFalse([t for t in threads if t.is_alive()],
                             "a race thread never finished")
        self.assertEqual(errors, [])
        entry = app_module._MODEL
        if entry is not None:
            for key in ("label", "path", "tok", "model", "device", "ctx",
                        "cfg", "note"):
                self.assertIn(key, entry)
            self.assertEqual(entry["label"], "fake/ckpt")

    # ------------------------------------------------------------ lock shape

    def test_model_lock_exists(self):
        """A plain (non-reentrant) Lock: the helpers keep every critical
        section non-nested, so no RLock is needed."""
        self.assertIsInstance(app_module._MODEL_LOCK, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
