import unittest

import torch

from langid.src import features
from langid.src.model import HashedLangID, encode_batch


class TestEncodeBatch(unittest.TestCase):
    def test_offsets_valid(self):
        ids, offsets = encode_batch(["hello world", "bonjour", "123"])
        self.assertEqual(int(offsets[0]), 0)
        self.assertTrue(torch.all(offsets[1:] >= offsets[:-1]).item())
        self.assertGreater(len(ids), 0)

    def test_empty_batch_gets_dummy_bucket(self):
        ids, offsets = encode_batch(["123"])
        self.assertEqual(len(ids), 1)


class TestModelLearns(unittest.TestCase):
    def test_loss_decreases_on_toy_task(self):
        torch.manual_seed(0)
        texts = ["aaa bbb"] * 16 + ["xxx yyy"] * 16
        labels = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
        model = HashedLangID(2, features.NUM_BUCKETS)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        ids, offsets = encode_batch(texts)
        first = last = None
        for step in range(30):
            loss = torch.nn.functional.cross_entropy(model(ids, offsets), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        self.assertLess(last, first)
        with torch.no_grad():
            acc = float((model(ids, offsets).argmax(1) == labels).float().mean())
        self.assertEqual(acc, 1.0)
