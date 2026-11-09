"""DBNet-style detection targets and box metrics for PAW detection (Stage 2).

Stage 2 of the Arabic OCR ladder (docs/plans/2026-09-29-arabic-ocr-nano-model.md).

A PAW (part of an Arabic word) is a connected component of the rendered ink, so
the synthetic engine already knows its exact box. The detection targets follow
DBNet (arXiv 1911.10194): a probability map over the SHRUNK box, an auxiliary
threshold map holding the normalised distance to the box boundary, and an
"ignore" band between the shrunk and the dilated box so the model is not
punished on the ambiguous rim.

The offset distance is DBNet's polygon formula

    d = area * (1 - shrink_ratio**2) / perimeter

which is exact for the axis-aligned boxes the renderer emits (pyclipper's round
joins only round the corners), so neither pyclipper nor cv2 is needed.

Boxes whose minimum side is below min_side are dropped from the target and
counted. Measured on the 200-page gate: the median PAW is 15x24 px and 72% of
PAWs have a minimum side >= 8 px. A 3 px haraka cannot be localised at any
useful output stride, and pretending otherwise would make the gate unreachable
rather than honest.
"""
from __future__ import annotations

import numpy as np


def paw_boxes_from_record(lines):
    """All PAW boxes of one page record, in page pixel coordinates."""
    out = []
    for ln in lines:
        for x0, y0, x1, y1 in ln["paws"]:
            out.append([float(x0), float(y0), float(x1), float(y1)])
    return out


def filter_boxes(boxes, min_side):
    """Split boxes into (kept, dropped) by minimum side length."""
    kept, dropped = [], []
    for b in boxes:
        if min(b[2] - b[0], b[3] - b[1]) >= min_side:
            kept.append(b)
        else:
            dropped.append(b)
    return kept, dropped


def shrink_distance(box, shrink_ratio):
    """DBNet's offset distance for one box; exact for axis-aligned rectangles."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    perim = 2.0 * (w + h)
    if perim <= 0:
        return 0.0
    return (w * h) * (1.0 - shrink_ratio * shrink_ratio) / perim


def build_targets(boxes, height, width, shrink_ratio=0.4):
    """Build (prob, ignore, thresh, thresh_mask) target maps.

    prob         float32 (H, W) in {0, 1}   - 1 inside the shrunk box
    ignore       bool    (H, W)             - dilated box minus shrunk box
    thresh       float32 (H, W) in [0, 1]   - 1 at the box edge, 0 at distance d
    thresh_mask  bool    (H, W)             - the dilated box
    """
    prob = np.zeros((height, width), dtype=np.float32)
    thresh = np.zeros((height, width), dtype=np.float32)
    tmask = np.zeros((height, width), dtype=bool)

    for box in boxes:
        x0, y0, x1, y1 = box
        d = shrink_distance(box, shrink_ratio)
        if d <= 0:
            continue
        dx0 = max(0, int(np.floor(x0 - d)))
        dy0 = max(0, int(np.floor(y0 - d)))
        dx1 = min(width, int(np.ceil(x1 + d)))
        dy1 = min(height, int(np.ceil(y1 + d)))
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        ys = np.arange(dy0, dy1, dtype=np.float32)[:, None]
        xs = np.arange(dx0, dx1, dtype=np.float32)[None, :]
        dist = np.minimum(np.minimum(xs - x0, x1 - xs), np.minimum(ys - y0, y1 - ys))
        dist = np.clip(dist, 0.0, d)
        val = 1.0 - dist / d
        thresh[dy0:dy1, dx0:dx1] = np.maximum(thresh[dy0:dy1, dx0:dx1], val)
        tmask[dy0:dy1, dx0:dx1] = True

        sx0 = int(np.ceil(x0 + d))
        sy0 = int(np.ceil(y0 + d))
        sx1 = int(np.floor(x1 - d))
        sy1 = int(np.floor(y1 - d))
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        prob[sy0:sy1, sx0:sx1] = 1.0

    ignore = tmask & (prob < 0.5)
    return prob, ignore, thresh, tmask


def downsample_targets(prob, ignore, thresh, tmask, factor=2):
    """Area-downsample target maps to the network's output resolution.

    prob/thresh are averaged (they are continuous values), ignore/tmask are
    max-pooled so that an ambiguous or supervised pixel is never silently
    promoted to a clean negative by averaging.
    """
    H, W = prob.shape
    h, w = H // factor, W // factor

    def area(a):
        return a.reshape(h, factor, w, factor).mean(axis=(1, 3))

    def mx(a):
        return a.reshape(h, factor, w, factor).max(axis=(1, 3))

    return (area(prob).astype(np.float32), mx(ignore).astype(bool),
            area(thresh).astype(np.float32), mx(tmask).astype(bool))


def boxes_iou(a, b):
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix1 - ix0, 0, None)
    ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ub = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = ua + ub - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


def match_boxes(pred, gt, iou_thr=0.5):
    """Greedy one-to-one matching at IoU >= iou_thr. Returns (tp, fp, fn)."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    iou = boxes_iou(pred, gt)
    order = np.argsort(-iou, axis=None)
    used_p, used_g, tp = set(), set(), 0
    for flat in order:
        i, j = divmod(int(flat), iou.shape[1])
        if iou[i, j] < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    """Precision / recall / Hmean from counts."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    h = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, h


def unclip_boxes(boxes, ratio=1.5):
    """Expand xyxy boxes outward by DBNet's unclip offset.

    The probability target is the box SHRUNK by shrink_ratio, so a decoded
    component is smaller than the PAW it came from (IoU ~0.34 against its own
    ground truth - far below any sane match threshold). DBNet's unclip expands a
    box by `area * ratio / perimeter`; 1.5 is DBNet's own default and is the
    right scale here too: for shrink_ratio 0.4 the exact inverse ratio only
    varies with aspect ratio between ~1.45 (square) and ~1.55 (long), which
    still lands within ~1-2% of the true box.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    perim = np.maximum(2.0 * (w + h), 1e-6)
    d = (w * h * ratio / perim).astype(np.float32)[:, None]
    return np.concatenate([boxes[:, :2] - d, boxes[:, 2:] + d], axis=1)


def unclip_boxes_exact(boxes, shrink_ratio=0.4):
    """Invert build_targets' rectangular erosion exactly, per box.

    A shrunk box is the original eroded by d on every side, and d itself depends
    on the ORIGINAL size: d = A(1 - r^2)/P. Substituting w = w' + 2d, h = h' + 2d
    and clearing denominators leaves a quadratic in d,

        (8 - 4c) d^2 + 2(1 - c)(w' + h') d - c w' h' = 0,   c = 1 - r^2

    whose positive root recovers d exactly. A single fixed ratio cannot: the
    effective ratio runs from ~1.45 for a square PAW to ~2.65 for a 7x40 one,
    so DBNet's fixed 1.5 leaves an elongated PAW at IoU 0.65 while this recovers
    it at IoU 1.00.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    c = 1.0 - shrink_ratio * shrink_ratio
    a = 8.0 - 4.0 * c
    b = 2.0 * (1.0 - c) * (w + h)
    disc = np.maximum(b * b + 4.0 * a * c * w * h, 0.0)
    d = ((-b + np.sqrt(disc)) / (2.0 * a)).astype(np.float32)[:, None]
    return np.concatenate([boxes[:, :2] - d, boxes[:, 2:] + d], axis=1)


def craft_blob_scale(thresh=0.5):
    """Width of a Gaussian's level set at `thresh`, in units of sigma.

    exp(-r^2 / (2 sigma^2)) == thresh  =>  r = sigma * sqrt(2 ln(1/thresh)),
    so the blob's full width is 2 * sqrt(2 ln(1/thresh)) * sigma. The size decode
    is only exact at the threshold the mask was built with: 2.3548 at 0.5, but
    3.1035 at 0.3 -- using the 0.5 constant on a 0.3 mask inflates every box ~32%.
    """
    return 2.0 * float(np.sqrt(2.0 * np.log(1.0 / float(thresh))))


CRAFT_DECODE_THRESH = 0.5
CRAFT_BLOB_SCALE = craft_blob_scale(CRAFT_DECODE_THRESH)  # 2.3548


def build_craft_prob(boxes, height, width, k=0.2, factor=2):
    """CRAFT-style region score: the max of one Gaussian per PAW.

    sigma is proportional to the PAW's OWN size, which is what makes the decode
    recover the box with a single constant (see decode_craft_boxes). The map is
    rasterised directly at the output resolution, so downsampling cannot blur the
    peaks away and shrink the blob below its true extent.

    Measured on 40 val pages with perfect maps, this raises the decoding ceiling
    at stride 2 from 78.6% (DBNet shrunk boxes, whose components merge because
    50.9% of PAWs share bounding-box area) to 92.5%.
    """
    h, w = height // factor, width // factor
    m = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2.0 / factor
        cy = (y0 + y1) / 2.0 / factor
        sx = max(k * (x1 - x0) / factor, 0.7)
        sy = max(k * (y1 - y0) / factor, 0.7)
        rx, ry = int(np.ceil(3.0 * sx)), int(np.ceil(3.0 * sy))
        xs = np.arange(int(cx) - rx, int(cx) + rx + 1)
        ys = np.arange(int(cy) - ry, int(cy) + ry + 1)
        xs = xs[(xs >= 0) & (xs < w)]
        ys = ys[(ys >= 0) & (ys < h)]
        if xs.size == 0 or ys.size == 0:
            continue
        gx = np.exp(-((xs - cx) ** 2) / (2.0 * sx * sx))
        gy = np.exp(-((ys - cy) ** 2) / (2.0 * sy * sy))
        blk = np.outer(gy, gx)
        sub = m[np.ix_(ys, xs)]
        m[np.ix_(ys, xs)] = np.maximum(sub, blk)
    return m


def decode_craft_boxes(mask, k=0.2, stride=2, min_side=6, thresh=CRAFT_DECODE_THRESH,
                       kx=None, ky=None):
    """Components of a CRAFT region score -> xyxy boxes in INPUT coordinates.

    A Gaussian of sigma = k * size has an extent of craft_blob_scale(thresh) * sigma
    at level `thresh`, so dividing the component extent by that constant recovers the
    PAW size. `thresh` MUST match the threshold `mask` was built with. min_side is
    applied after the rescale, matching the filter the ground-truth boxes went through.
    """
    kx = k if kx is None else kx
    ky = k if ky is None else ky
    comp = decode_boxes(mask, min_side=1, stride=stride)
    if len(comp) == 0:
        return comp
    cx = (comp[:, 0] + comp[:, 2]) / 2.0
    cy = (comp[:, 1] + comp[:, 3]) / 2.0
    bs = craft_blob_scale(thresh)
    bw = (comp[:, 2] - comp[:, 0]) / (bs * kx)
    bh = (comp[:, 3] - comp[:, 1]) / (bs * ky)
    out = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1)
    kept, _ = filter_boxes(out, min_side)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 4)

def build_centernet_targets(boxes, height, width, factor=2, sigma_k=0.25, sigma_min=1.0):
    """CenterNet-style targets: a centre heatmap plus direct w/h regression.

    A PAW is a small, densely packed object whose box today is read off a level
    set of a Gaussian. That couples the box to the blob's SHAPE, and the measured
    failure (HANDOFF 2026-10-04 (k)) is exactly that: decoded singletons came out
    25.5 x 25.5 (square) against tall GT PAWs (20.4 x 22.1), and 36.8% of PAWs
    merged into a neighbour at the decode threshold. PAW centres are 99.97%
    unique, so one peak per PAW removes both failure modes by construction --
    there is no level set to get the shape wrong and no blob to merge.

    Rasterised at the OUTPUT resolution, like build_craft_prob, so no
    downsampling step is needed.

    Returns (heat, size, off, mask):
      heat  (h, w)    float32  Gaussian at each PAW centre, peak 1
      size  (2, h, w) float32  PAW w/h in INPUT pixels at the centre pixel
      off   (2, h, w) float32  sub-pixel centre offset from that pixel
      mask  (h, w)    bool     the centre pixels -- the only supervised ones
    """
    h, w = height // factor, width // factor
    heat = np.zeros((h, w), dtype=np.float32)
    size = np.zeros((2, h, w), dtype=np.float32)
    off = np.zeros((2, h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        cx = (x0 + x1) / 2.0 / factor
        cy = (y0 + y1) / 2.0 / factor
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        s = max(sigma_k * min(bw, bh) / factor, sigma_min)
        r = int(np.ceil(3.0 * s))
        xs = np.arange(ix - r, ix + r + 1)
        ys = np.arange(iy - r, iy + r + 1)
        xs = xs[(xs >= 0) & (xs < w)]
        ys = ys[(ys >= 0) & (ys < h)]
        if xs.size == 0 or ys.size == 0:
            continue
        gx = np.exp(-((xs - cx) ** 2) / (2.0 * s * s))
        gy = np.exp(-((ys - cy) ** 2) / (2.0 * s * s))
        heat[np.ix_(ys, xs)] = np.maximum(heat[np.ix_(ys, xs)], np.outer(gy, gx))
        size[0, iy, ix] = bw
        size[1, iy, ix] = bh
        off[0, iy, ix] = cx - ix
        off[1, iy, ix] = cy - iy
        mask[iy, ix] = True
    return heat, size, off, mask


def decode_centernet_boxes(heat, size, off, stride=2, min_side=6, topk=200,
                           peak_thresh=0.3):
    """Peaks of a centre heatmap -> xyxy boxes in INPUT coordinates.

    Each PAW is one peak, so there is no level set and no merging: the box is
    read directly from the regressed w/h at the peak pixel, recentred by the
    regressed sub-pixel offset.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(heat, -30.0, 30.0)))
    ph, pw = p.shape
    mx = p.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros_like(p)
            y0, y1 = max(0, dy), min(ph, ph + dy)
            x0, x1 = max(0, dx), min(pw, pw + dx)
            sh[y0:y1, x0:x1] = p[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            mx = np.maximum(mx, sh)
    ys, xs = np.nonzero((p == mx) & (p >= peak_thresh))
    if len(ys) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    order = np.argsort(-p[ys, xs])[:topk]
    ys, xs = ys[order], xs[order]
    cx = (xs + off[0, ys, xs]) * stride
    cy = (ys + off[1, ys, xs]) * stride
    bw = size[0, ys, xs]
    bh = size[1, ys, xs]
    out = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1)
    kept, _ = filter_boxes(out, min_side)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 4)



def _label_components(mask):
    """8-connected components of a boolean map (scipy-free two-pass union-find).

    Returns (labels, n) with labels in [1, n]. Diagonal neighbours are merged,
    which is what Arabic ink needs: a stroke that only touches its neighbour on
    a corner is one PAW, not two.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = [0]

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    nxt = 1
    for y in range(H):
        xs_fg = np.flatnonzero(mask[y])
        if xs_fg.size == 0:
            continue
        lab = labels[y]
        prev = labels[y - 1] if y else None
        for x in xs_fg:
            cands = []
            if x and lab[x - 1]:
                cands.append(lab[x - 1])
            if prev is not None:
                if prev[x]:
                    cands.append(prev[x])
                if x and prev[x - 1]:
                    cands.append(prev[x - 1])
                if x + 1 < W and prev[x + 1]:
                    cands.append(prev[x + 1])
            if not cands:
                lab[x] = nxt
                parent.append(nxt)
                nxt += 1
            else:
                root = find(cands[0])
                for c in cands[1:]:
                    rc = find(c)
                    if rc != root:
                        parent[max(root, rc)] = min(root, rc)
                        root = min(root, rc)
                lab[x] = root
    for y in range(H):
        xs_fg = np.flatnonzero(labels[y])
        for x in xs_fg:
            l = labels[y, x]
            if l:
                labels[y, x] = find(l)
    return labels, nxt - 1


def decode_boxes(mask, min_side=6, stride=2):
    """Connected components of a binary map -> xyxy boxes in INPUT coordinates.

    min_side is expressed in input pixels, so the same threshold applies to
    the target boxes and to the decoded predictions.
    """
    labels, n = _label_components(mask)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    H, W = labels.shape
    miny = np.full(n + 1, H, dtype=np.int64)
    maxy = np.full(n + 1, -1, dtype=np.int64)
    minx = np.full(n + 1, W, dtype=np.int64)
    maxx = np.full(n + 1, -1, dtype=np.int64)
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    np.minimum.at(miny, labs, ys)
    np.maximum.at(maxy, labs, ys)
    np.minimum.at(minx, labs, xs)
    np.maximum.at(maxx, labs, xs)
    out = []
    for lab in range(1, n + 1):
        y0, y1 = int(miny[lab]), int(maxy[lab]) + 1
        x0, x1 = int(minx[lab]), int(maxx[lab]) + 1
        if min(x1 - x0, y1 - y0) * stride >= min_side:
            out.append([x0 * stride, y0 * stride, x1 * stride, y1 * stride])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)
