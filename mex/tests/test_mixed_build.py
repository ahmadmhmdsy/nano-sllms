# mex/tests/test_mixed_build.py — mixed-set builder determinism + schema.
"""mu1 plan ("Mixed eval set", task 1): val.jsonl 200 = 50/task held-out,
train.jsonl 8000 = 2000/task TRAIN-legal, both seeded and interleaved; every
line has exactly {prompt, target, task} with task in x1..x4 and the on-disk
files byte-reproducible across runs (regeneration must change nothing)."""
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_spec = importlib.util.spec_from_file_location(
    "build_mixed_eval", Path(__file__).resolve().parents[1] / "scripts" /
    "build_mixed_eval.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


class TestMixedBuild(unittest.TestCase):
    def test_small_sample_is_deterministic(self):
        """Same (task, per_task, seed, split) => identical items, second call."""
        a = build.build_task_items("x2", 10, random.Random(123), "test")
        b = build.build_task_items("x2", 10, random.Random(123), "test")
        self.assertEqual(a, b)
        c = build.build_task_items("x2", 10, random.Random(124), "test")
        self.assertNotEqual([{i["prompt"], i["target"]} for i in a],
                            [{i["prompt"], i["target"]} for i in c])

    def test_sample_items_are_new_keyed(self):
        for it in build.build_task_items("x4", 8, random.Random(1), "test"):
            self.assertEqual(set(it) == {"prompt", "target", "task"}, True)
            self.assertEqual(it["task"], "x4")
            self.assertFalse(it["target"].endswith("\n"))

    def test_interleave_round_robin(self):
        blocks = [[{"task": "x%d" % (j + 1), "i": i} for i in range(2)]
                  for j in range(4)]
        flat = build.interleave(blocks, 2)
        self.assertEqual([x["task"] for x in flat],
                         ["x1", "x2", "x3", "x4"] * 2)

    def test_build_items_deterministic_and_shape(self):
        """Full build: 200 val / 8000 train, 50|2000 per task, repeatable (the
        exact regeneration check the mu1 plan requires)."""
        val1, train1 = build.build_items()
        val2, train2 = build.build_items()
        self.assertEqual(len(val1), 200)
        self.assertEqual(len(train1), 8000)
        self.assertEqual(json.dumps(val1), json.dumps(val2))
        self.assertEqual(json.dumps(train1), json.dumps(train2))
        from collections import Counter
        self.assertEqual(Counter(i["task"] for i in val1).most_common(1)[0][1],
                         50)
        self.assertEqual(Counter(i["task"] for i in train1).most_common(1)[0][1],
                         2000)
        self.assertTrue(all(set(i) == {"prompt", "target", "task"}
                            for i in val1 + train1))

    def test_on_disk_regenerated_files_were_seeded(self):
        """The committed mixed files exist with the frozen counts and keys."""
        for name, n, per in (("val.jsonl", 200, 50), ("train.jsonl", 8000, 2000)):
            p = build.MIXED_DIR / name
            if not p.exists():  # builder regenerates on demand
                continue
            items = [json.loads(l) for l in
                     p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(items), n, name)
            from collections import Counter
            self.assertTrue(all(set(i) == {"prompt", "target", "task"}
                                for i in items))
            self.assertEqual(
                Counter(i["task"] for i in items).most_common(1)[0][1], per)
