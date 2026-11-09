"""Unit tests for committee distillation (MU1 arm D, row E-29).

Numeric cases are chosen so softmax / CE / KL have closed forms checked
independently of the implementation under test.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.kd import TeacherLogitCache, kd_loss


LOG4 = float(np.log(4.0))


class TestKdLoss(unittest.TestCase):
    def _l(self, vals):
        return torch.tensor(vals, dtype=torch.float32).view(1, 5, 4)

    def test_alpha_zero_is_pure_ce(self):
        # student all zeros: CE = ln(4) at every position.
        logits = self._l([0] * 20)
        labels = torch.tensor([[0, 1, 2, 3, 3]])
        val = kd_loss(logits, torch.zeros_like(logits), labels, alpha=0.0)
        # labels[:, 1:] = [1, 2, 3, 3] -> 4 valid positions, batchmean means
        # the mean stays ln(4).
        self.assertAlmostEqual(val.item(), LOG4, places=6)

    def test_hand_computed_uniform_teacher(self):
        # teacher zeros -> softmax(0) uniform; student zeros -> log_softmax
        # = -ln(4), so KL(uniform || student) = 0 exactly, and
        # loss = alpha*0 + (1-alpha)*CE = 0.5*ln(4).
        logits = self._l([0] * 20)
        labels = torch.tensor([[0, 1, 2, 3, 3]])
        val = kd_loss(logits, torch.zeros_like(logits), labels, alpha=0.5)
        self.assertAlmostEqual(val.item(), 0.5 * LOG4, places=6)

    def test_hand_computed_temperature_scale(self):
        # teacher = student -> the KL term is exactly 0 at ANY temperature,
        # so the loss must equal 0.5*CE(shifted labels). Hand-computed CE:
        # 3 positions with all-zero logits -> ln(4) each; the shifted
        # position whose logit row holds k=4*ln(4) at the label column has
        # CE = log(sum_i exp(x_i)) - x_label = ln(3+e^k) - k.
        k = 4.0 * LOG4
        student = self._l([0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
                           0, 0, 0, k,  0, 0, 0, 0])
        labels = torch.tensor([[0, 1, 2, 3, 3]])
        val = kd_loss(student, student.clone(), labels, alpha=0.5,
                      temperature=2.0)
        ce_pos = float(np.log(3.0 + np.exp(k))) - k   # k-row position
        expected = 0.5 * (3.0 * LOG4 + ce_pos) / 4.0   # shifted -> 4 valid
        self.assertAlmostEqual(val.item(), expected, places=5)


class TestTeacherLogitCache(unittest.TestCase):
    SEQ, VOCAB = 4, 5

    def test_lookup_roundtrip_and_miss(self):
        a = np.arange(2 * self.SEQ * self.VOCAB, dtype=np.float32) \
                  .reshape(2, self.SEQ, self.VOCAB)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cache.npy"
            np.save(path, a)
            cache = TeacherLogitCache(path, seq_len=self.SEQ, vocab=self.VOCAB)
            self.assertEqual(cache.n_blocks, 2)
            np.testing.assert_array_equal(np.asarray(cache.block_logits(0)), a[0])
            np.testing.assert_array_equal(np.asarray(cache.block_logits(1)), a[1])
            miss = np.asarray(cache.block_logits(2))  # miss -> zeros (pad)
            self.assertEqual(miss.shape, (self.SEQ, self.VOCAB))
            self.assertEqual(float(miss.sum()), 0.0)
            # Release the memmap BEFORE temp-dir cleanup: on Windows an open
            # np.memmap keeps the file locked and cleanup fails (WinError 32).
            cache.cache._mmap.close()

    def test_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.npy"
            np.save(path, np.zeros((2, self.SEQ + 1, self.VOCAB), dtype=np.float32))
            with self.assertRaises(ValueError):
                TeacherLogitCache(path, seq_len=self.SEQ, vocab=self.VOCAB)


if __name__ == "__main__":
    unittest.main()

