"""Stage 2 -> Stage 3 hand-off: merge detector boxes into text LINES and crop them.

The pretrained detector (see scripts/detector_gpu.py) emits one box per word /
PAW. The Stage 3 recognizer reads one LINE at a time, so the boxes are first
grouped by vertical overlap and then merged horizontally inside a group. That
single rule covers both cases the pipeline sees:

* a real data/hflines line crop - the 2-3 word boxes share one vertical band and
  collapse to 1 line box;
* a synthetic PAGE - the lines are vertically separated, so they stay separate
  (measured: 8.7 boxes -> 8 lines on the val pages).

Pure numpy (no cv2) on purpose: this module is importable and unit-testable in
the base .venv, while the detector itself needs .venv-paddle-gpu.
"""
from __future__ import annotations

import numpy as np

DEFAULT_Y_OVERLAP = 0.5   # min vertical overlap (fraction of the shorter box)
DEFAULT_GAP_RATIO = 0.6   # max horizontal gap to merge, x line height
DEFAULT_PAD = 2           # px of padding added around each crop
MIN_SIDE = 4              # boxes thinner than this are dropped as noise


def quad_to_rect(box):
    """Axis-aligned (x0, y0, x1, y1) from a 4-point quad or a rect."""
    a = np.asarray(box, dtype=np.float64)
    if a.ndim == 2 and a.shape[0] >= 2:
        return (float(a[:, 0].min()), float(a[:, 1].min()),
                float(a[:, 0].max()), float(a[:, 1].max()))
    if a.size == 4:
        x0, y0, x1, y1 = (float(v) for v in a.reshape(-1))
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    raise ValueError("cannot read a box from shape %s" % (a.shape,))


def _v_overlap(a, b):
    """Vertical overlap of two rects as a fraction of the SHORTER one."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    if inter <= 0:
        return 0.0
    shorter = min(a[3] - a[1], b[3] - b[1])
    return float(inter / shorter) if shorter > 0 else 0.0


def _bbox(group):
    return (min(r[0] for r in group), min(r[1] for r in group),
            max(r[2] for r in group), max(r[3] for r in group))


def _merge_x(group, gap_ratio):
    """Merge horizontally adjacent boxes of one line group (chain-merge)."""
    out = [list(r) for r in sorted(group, key=lambda r: r[0])]
    merged = [out[0]]
    for r in out[1:]:
        prev = merged[-1]
        height = max(prev[3] - prev[1], r[3] - r[1])
        if r[0] - prev[2] <= gap_ratio * height:
            prev[2] = max(prev[2], r[2])
            prev[1] = min(prev[1], r[1])
            prev[3] = max(prev[3], r[3])
        else:
            merged.append(list(r))
    return [tuple(r) for r in merged]


def merge_boxes_to_lines(boxes, y_overlap=DEFAULT_Y_OVERLAP,
                         gap_ratio=DEFAULT_GAP_RATIO, min_side=MIN_SIDE):
    """Group detector boxes into lines and merge each line's boxes.

    Args:
        boxes: iterable of 4-point quads (detector output) or (x0,y0,x1,y1).
        y_overlap: a box joins a group when its vertical overlap with the
            group's bounding rect is at least this fraction of the shorter box.
        gap_ratio: two boxes of one group merge when the horizontal gap is at
            most this fraction of the line height.
        min_side: drop boxes whose width or height is below this.

    Returns:
        List of (x0, y0, x1, y1) line rects, sorted top-to-bottom, left-to-right.
    """
    rects = []
    for b in boxes:
        r = quad_to_rect(b)
        if r[2] - r[0] < min_side or r[3] - r[1] < min_side:
            continue
        rects.append(r)
    if not rects:
        return []

    groups = []
    for r in sorted(rects, key=lambda r: (r[1], r[0])):
        best, best_ov = None, 0.0
        for g in groups:
            ov = _v_overlap(r, _bbox(g))
            if ov > best_ov:
                best, best_ov = g, ov
        if best is not None and best_ov >= y_overlap:
            best.append(r)
        else:
            groups.append([r])

    lines = []
    for g in groups:
        lines.extend(_merge_x(g, gap_ratio))
    lines.sort(key=lambda r: (r[1], r[0]))
    return lines


def union_rect(boxes, min_side=MIN_SIDE):
    """Union of all boxes as one rect - for inputs that are ALREADY a line crop.

    Grouping is the right rule for a page (lines are vertically separated), but
    a data/hflines-style crop contains exactly one line, so any split is a
    detector artefact and the union is the correct region.

    Returns:
        [(x0, y0, x1, y1)] or [] when no usable box is present.
    """
    rects = []
    for b in boxes:
        r = quad_to_rect(b)
        if r[2] - r[0] >= min_side and r[3] - r[1] >= min_side:
            rects.append(r)
    if not rects:
        return []
    return [(min(r[0] for r in rects), min(r[1] for r in rects),
             max(r[2] for r in rects), max(r[3] for r in rects))]


def crop_rect(image, rect, pad=DEFAULT_PAD):
    """Crop an axis-aligned rect with padding, clipped to the image.

    Returns:
        (crop, clipped_rect); crop is None when the rect falls outside.
    """
    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(rect[0])) - pad)
    y0 = max(0, int(np.floor(rect[1])) - pad)
    x1 = min(w, int(np.ceil(rect[2])) + pad)
    y1 = min(h, int(np.ceil(rect[3])) + pad)
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def crop_lines(image, boxes, y_overlap=DEFAULT_Y_OVERLAP,
               gap_ratio=DEFAULT_GAP_RATIO, pad=DEFAULT_PAD,
               min_side=MIN_SIDE):
    """Detect-then-crop convenience: merge boxes to lines, then crop each one.

    Returns:
        List of dicts {rect, crop} in reading order (crops are never None).
    """
    out = []
    for line in merge_boxes_to_lines(boxes, y_overlap=y_overlap,
                                     gap_ratio=gap_ratio, min_side=min_side):
        crop, rect = crop_rect(image, line, pad=pad)
        if crop is not None:
            out.append({"rect": rect, "crop": crop})
    return out
