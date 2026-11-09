"""Unit tests for src.inference.line_crop (pure numpy - runs in the base .venv)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr"))

from src.inference.line_crop import (crop_lines, crop_rect,  # noqa: E402
                                     merge_boxes_to_lines, quad_to_rect,
                                     union_rect)


class TestQuadToRect(unittest.TestCase):
    def test_quad(self):
        quad = [[10, 20], [110, 22], [108, 60], [12, 58]]
        self.assertEqual(quad_to_rect(quad), (10.0, 20.0, 110.0, 60.0))

    def test_rect(self):
        self.assertEqual(quad_to_rect([5, 6, 1, 2]), (1.0, 2.0, 5.0, 6.0))

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            quad_to_rect([1, 2, 3])


class TestMergeBoxesToLines(unittest.TestCase):
    def test_two_words_on_one_line_merge(self):
        # gap 15 px against a 30 px line height = 0.5 < gap_ratio 0.6 -> merge
        boxes = [[0, 100, 40, 130], [55, 102, 120, 132]]
        self.assertEqual(merge_boxes_to_lines(boxes), [(0.0, 100.0, 120.0, 132.0)])

    def test_word_gap_at_the_boundary_is_not_merged(self):
        # gap 20 px against a 30 px line height = 0.67 > gap_ratio 0.6
        boxes = [[0, 100, 40, 130], [60, 102, 120, 132]]
        self.assertEqual(len(merge_boxes_to_lines(boxes)), 2)

    def test_two_lines_stay_separate(self):
        boxes = [[0, 10, 100, 40], [0, 200, 100, 230]]
        lines = merge_boxes_to_lines(boxes)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][1], 10.0)
        self.assertEqual(lines[1][1], 200.0)

    def test_large_horizontal_gap_is_not_merged(self):
        boxes = [[0, 10, 40, 40], [240, 10, 280, 40]]
        self.assertEqual(len(merge_boxes_to_lines(boxes)), 2)

    def test_three_word_chain_merges(self):
        boxes = [[0, 10, 40, 40], [50, 10, 90, 40], [100, 10, 140, 40]]
        self.assertEqual(len(merge_boxes_to_lines(boxes)), 1)

    def test_tiny_boxes_dropped(self):
        boxes = [[0, 10, 100, 40], [50, 12, 52, 14]]
        self.assertEqual(merge_boxes_to_lines(boxes), [(0.0, 10.0, 100.0, 40.0)])

    def test_empty(self):
        self.assertEqual(merge_boxes_to_lines([]), [])

    def test_partial_vertical_overlap_groups(self):
        boxes = [[0, 0, 50, 50], [60, 20, 110, 70]]
        self.assertEqual(len(merge_boxes_to_lines(boxes, y_overlap=0.5)), 1)
        self.assertEqual(len(merge_boxes_to_lines(boxes, y_overlap=0.8)), 2)

    def test_sorted_reading_order(self):
        boxes = [[0, 200, 50, 230], [0, 10, 50, 40], [60, 10, 110, 40]]
        lines = merge_boxes_to_lines(boxes)
        self.assertEqual([r[1] for r in lines], [10.0, 200.0])


class TestUnionRect(unittest.TestCase):
    def test_union_of_split_line_boxes(self):
        boxes = [[0, 10, 40, 40], [200, 12, 260, 44]]
        self.assertEqual(union_rect(boxes), [(0.0, 10.0, 260.0, 44.0)])

    def test_union_ignores_noise(self):
        boxes = [[0, 10, 40, 40], [200, 12, 260, 44], [5, 5, 6, 6]]
        self.assertEqual(union_rect(boxes), [(0.0, 10.0, 260.0, 44.0)])

    def test_union_empty(self):
        self.assertEqual(union_rect([]), [])


class TestCrop(unittest.TestCase):
    def test_crop_rect_clips_to_image(self):
        img = np.zeros((50, 60, 3), dtype=np.uint8)
        crop, rect = crop_rect(img, (-10, -10, 20, 20), pad=2)
        self.assertEqual(crop.shape[:2], (22, 22))
        self.assertEqual(rect, (0, 0, 22, 22))

    def test_crop_rect_returns_none_outside(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        crop, _ = crop_rect(img, (100, 100, 120, 120), pad=2)
        self.assertIsNone(crop)

    def test_crop_lines_reading_order_and_pad(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[10:40, 0:100] = 255
        img[200:230, 0:100] = 255
        out = crop_lines(img, [[0, 10, 100, 40], [0, 200, 100, 230]], pad=1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["rect"], (0, 9, 101, 41))
        self.assertTrue((out[0]["crop"] > 0).any())
        self.assertEqual(out[1]["rect"][1], 199)


if __name__ == "__main__":
    unittest.main()
