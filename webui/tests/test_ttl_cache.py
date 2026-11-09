"""M4.4: TTL-cache tests for webui/app.py.

The three expensive reads (_full_curve, _ckpts, _free_vram_mib) all go
through one shared TTL cache (_ttl_cached). These tests pin the cache
behaviour without a GPU, tensorboard or a real runs/ tree: the underlying
disk/GPU functions are monkeypatched, and a fake monotonic clock is
advanced by hand for the TTL-expiry tests (no sleeping).

Run from the repo root:
    .venv/Scripts/python.exe -W error::SyntaxWarning -m unittest discover -s webui/tests -t .
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

WEBUI_DIR = Path(__file__).resolve().parents[1]
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))

import app as app_module  # noqa: E402  (webui/ put on sys.path above)


class _FakeClock:
    """A monotonic clock the tests advance by hand."""

    def __init__(self, start=1_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TtlCacheTest(unittest.TestCase):
    """The shared helper itself: hit, TTL expiry, version invalidation."""

    def setUp(self):
        app_module._TTL_CACHE.clear()
        self.clock = _FakeClock()
        patcher = mock.patch.object(app_module._time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(app_module._TTL_CACHE.clear)

    @staticmethod
    def _loader(calls, value):
        def load():
            calls.append(1)
            return list(value)  # fresh object: identity asserts can see reloads
        return load

    def test_second_call_returns_same_object(self):
        calls = []
        loader = self._loader(calls, ["x"])
        a = app_module._ttl_cached("k", 30.0, loader)
        b = app_module._ttl_cached("k", 30.0, loader)
        self.assertIs(a, b)
        self.assertEqual(len(calls), 1)

    def test_ttl_expiry_reloads(self):
        calls = []
        loader = self._loader(calls, ["x"])
        a = app_module._ttl_cached("k", 30.0, loader)
        self.clock.advance(29.9)
        self.assertIs(app_module._ttl_cached("k", 30.0, loader), a)
        self.clock.advance(0.2)  # past the 30 s TTL
        c = app_module._ttl_cached("k", 30.0, loader)
        self.assertIsNot(a, c)
        self.assertEqual(len(calls), 2)

    def test_version_change_invalidates_before_ttl(self):
        """A new `version` (e.g. a new file mtime) forces a reload even
        well inside the TTL."""
        calls = []
        loader = self._loader(calls, ["x"])
        a = app_module._ttl_cached("k", 30.0, loader, version=1.0)
        b = app_module._ttl_cached("k", 30.0, loader, version=2.0)
        self.assertIsNot(a, b)
        self.assertEqual(len(calls), 2)

    def test_distinct_keys_are_independent(self):
        calls = []
        loader = self._loader(calls, ["x"])
        a = app_module._ttl_cached("k1", 30.0, loader)
        b = app_module._ttl_cached("k2", 30.0, loader)
        self.assertIsNot(a, b)
        self.assertEqual(len(calls), 2)


class FullCurveCacheTest(unittest.TestCase):
    """_full_curve: identity, TTL expiry, mtime invalidation."""

    def setUp(self):
        app_module._TTL_CACHE.clear()
        self.clock = _FakeClock()
        patcher = mock.patch.object(app_module._time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(app_module._TTL_CACHE.clear)
        self.logs_dir = Path(tempfile.mkdtemp())

    def _patch_reader(self):
        """Replace the tensorboard read with a counting fake."""
        calls = []

        def fake_read(logs_dir, tag):
            calls.append((logs_dir, tag))
            return [1, 2, 3], [0.5, 0.4, 0.3]

        patcher = mock.patch.object(app_module, "_read_full_curve", fake_read)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_second_call_returns_same_object(self):
        calls = self._patch_reader()
        a = app_module._full_curve(self.logs_dir, "train/loss")
        b = app_module._full_curve(self.logs_dir, "train/loss")
        self.assertIs(a, b)
        self.assertEqual(a, ([1, 2, 3], [0.5, 0.4, 0.3]))
        self.assertEqual(len(calls), 1)

    def test_ttl_expiry_reloads(self):
        calls = self._patch_reader()
        a = app_module._full_curve(self.logs_dir, "train/loss")
        self.clock.advance(app_module._CURVE_TTL_S - 0.1)
        self.assertIs(app_module._full_curve(self.logs_dir, "train/loss"), a)
        self.clock.advance(0.2)
        c = app_module._full_curve(self.logs_dir, "train/loss")
        self.assertIsNot(a, c)
        self.assertEqual(len(calls), 2)

    def test_mtime_change_invalidates(self):
        """Brief Step 1/4c: editing the metrics file (new mtime) must
        invalidate the cached curve."""
        calls = self._patch_reader()
        ev = self.logs_dir / "events.out.tfevents.123"
        ev.write_text("v1", encoding="utf-8")
        a = app_module._full_curve(self.logs_dir, "train/loss")
        ev.write_text("v2", encoding="utf-8")
        os.utime(ev, (20_000_000, 20_000_000))  # force a different mtime
        b = app_module._full_curve(self.logs_dir, "train/loss")
        self.assertIsNot(a, b)
        self.assertEqual(len(calls), 2)

    def test_same_mtime_stays_cached(self):
        """Invalidation is keyed on the mtime, not the content: an untouched
        metrics file keeps serving the cached object."""
        calls = self._patch_reader()
        ev = self.logs_dir / "events.out.tfevents.123"
        ev.write_text("v1", encoding="utf-8")
        a = app_module._full_curve(self.logs_dir, "train/loss")
        b = app_module._full_curve(self.logs_dir, "train/loss")
        self.assertIs(a, b)
        self.assertEqual(len(calls), 1)

    def test_unrelated_file_does_not_invalidate(self):
        """Only the metrics file's mtime is in the key: touching another
        file in logs_dir leaves the cache warm."""
        calls = self._patch_reader()
        ev = self.logs_dir / "events.out.tfevents.123"
        ev.write_text("v1", encoding="utf-8")
        a = app_module._full_curve(self.logs_dir, "train/loss")
        other = self.logs_dir / "chain_out.log"
        other.write_text("newer", encoding="utf-8")
        os.utime(other, (30_000_000, 30_000_000))
        b = app_module._full_curve(self.logs_dir, "train/loss")
        self.assertIs(a, b)
        self.assertEqual(len(calls), 1)

    def test_distinct_tags_cached_separately(self):
        calls = self._patch_reader()
        a = app_module._full_curve(self.logs_dir, "train/loss")
        b = app_module._full_curve(self.logs_dir, "eval/loss")
        self.assertIsNot(a, b)
        self.assertEqual(len(calls), 2)

    def test_missing_dir_returns_empty_without_reading(self):
        calls = self._patch_reader()
        missing = self.logs_dir / "nope"
        self.assertEqual(app_module._full_curve(missing, "train/loss"),
                         ([], []))
        self.assertEqual(calls, [])


class CkptsCacheTest(unittest.TestCase):
    """_ckpts: the real listing wrapped in the 30 s cache."""

    def setUp(self):
        app_module._TTL_CACHE.clear()
        self.clock = _FakeClock()
        patcher = mock.patch.object(app_module._time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(app_module._TTL_CACHE.clear)
        self.tmp = Path(tempfile.mkdtemp())
        run = self.tmp / "run1"
        (run / "final").mkdir(parents=True)
        (run / "final" / "model.safetensors").write_bytes(b"w")
        (run / "checkpoint-500").mkdir()
        (run / "checkpoint-500" / "model.safetensors").write_bytes(b"w")
        (self.tmp / "logs").mkdir()  # the shared logs dir is skipped

    def test_real_listing_is_cached(self):
        """The cache wraps the REAL listing: one walk, one dict object."""
        with mock.patch.object(app_module, "RUNS", self.tmp):
            a = app_module._ckpts()
            b = app_module._ckpts()
        self.assertIs(a, b)
        self.assertEqual(set(a), {"run1/final", "run1/checkpoint-500"})
        self.assertEqual(a["run1/final"], self.tmp / "run1" / "final")
        self.assertEqual(a["run1/checkpoint-500"],
                         self.tmp / "run1" / "checkpoint-500")

    def test_second_call_returns_same_object(self):
        with mock.patch.object(app_module, "RUNS", self.tmp), \
                mock.patch.object(app_module, "_list_ckpts",
                                  return_value={"run1/final": self.tmp / "x"}) as m:
            a = app_module._ckpts()
            b = app_module._ckpts()
        self.assertIs(a, b)
        self.assertEqual(m.call_count, 1)

    def test_ttl_expiry_reloads(self):
        listing = mock.Mock(
            side_effect=lambda: {"run1/final": self.tmp / "x"})  # fresh dict
        with mock.patch.object(app_module, "_list_ckpts", listing):
            a = app_module._ckpts()
            self.clock.advance(app_module._CKPTS_TTL_S - 0.1)
            self.assertIs(app_module._ckpts(), a)
            self.clock.advance(0.2)
            c = app_module._ckpts()
        self.assertIsNot(a, c)
        self.assertEqual(listing.call_count, 2)


class FreeVramCacheTest(unittest.TestCase):
    """_free_vram_mib: the nvidia-smi probe wrapped in the 5 s cache."""

    def setUp(self):
        app_module._TTL_CACHE.clear()
        self.clock = _FakeClock()
        patcher = mock.patch.object(app_module._time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(app_module._TTL_CACHE.clear)

    def test_second_call_returns_same_object(self):
        with mock.patch.object(app_module, "_query_free_vram_mib",
                               return_value=12345) as m:
            a = app_module._free_vram_mib()
            b = app_module._free_vram_mib()
        self.assertIs(a, b)
        self.assertEqual(a, 12345)
        self.assertEqual(m.call_count, 1)

    def test_ttl_expiry_reloads_and_sees_new_value(self):
        probe = mock.Mock(side_effect=[1000, 2000])
        with mock.patch.object(app_module, "_query_free_vram_mib", probe):
            self.assertEqual(app_module._free_vram_mib(), 1000)
            self.clock.advance(app_module._FREE_VRAM_TTL_S - 0.1)
            self.assertEqual(app_module._free_vram_mib(), 1000)  # still cached
            self.clock.advance(0.2)
            self.assertEqual(app_module._free_vram_mib(), 2000)
        self.assertEqual(probe.call_count, 2)

    def test_probe_failure_returns_none_and_is_cached(self):
        """No GPU / no nvidia-smi: the fallback None is cached too, so a
        broken probe is not re-spawned on every call."""
        with mock.patch.object(app_module, "_query_free_vram_mib",
                               return_value=None) as m:
            self.assertIsNone(app_module._free_vram_mib())
            self.assertIsNone(app_module._free_vram_mib())
        self.assertEqual(m.call_count, 1)

    def test_raw_probe_returns_none_when_nvidia_smi_missing(self):
        """The underlying query's fallback path, without a GPU."""
        with mock.patch("subprocess.run", side_effect=OSError("no nvidia-smi")):
            self.assertIsNone(app_module._query_free_vram_mib())


if __name__ == "__main__":
    unittest.main()
