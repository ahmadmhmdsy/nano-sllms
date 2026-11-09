"""Stage 2 -> Stage 3 hand-off: detect text LINES with the pretrained detector, crop them.

The detector (scripts/detector_gpu.py) emits one box per word/PAW; the Stage 3
recognizer reads one LINE at a time, so boxes are grouped by vertical overlap
and merged horizontally (src/inference/line_crop.py) before cropping.

Runs under .venv-paddle-gpu (needs paddle + cv2 + PIL):

    # gate check on synthetic val pages, with the y_overlap sweep
    & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\stage2_detect_lines.py ^
        --dir runs\stage2_synth\val\pages --limit 30 ^
        --gt runs\stage2_synth\val\pages.jsonl --sweep 0.3,0.5,0.7

    # real line crops: does a whole line collapse to ONE box?
    & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\stage2_detect_lines.py ^
        --dir data\hflines\val --limit 100 --ink

    # materialise crops for the recognizer
    & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\stage2_detect_lines.py ^
        --dir runs\stage2_synth\val\pages --limit 30 --out runs\stage2_lines\val --manifest runs\stage2_lines\val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr"))
sys.path.insert(0, str(ROOT / "arabic_ocr" / "scripts"))

from detector_gpu import detect, load_model  # noqa: E402
from src.inference.line_crop import (DEFAULT_GAP_RATIO, DEFAULT_PAD,  # noqa: E402
                                     DEFAULT_Y_OVERLAP, crop_rect,
                                     merge_boxes_to_lines, quad_to_rect,
                                     union_rect)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_EXT)


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gt[Path(d["image"]).name] = d
    return gt


def iou_rect(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def ink_ratio(image, rects):
    """Fraction of the image's ink pixels inside the merged rects, and ink count."""
    gray = image.astype(np.float32).mean(axis=2)
    ink = gray < 128
    total = int(ink.sum())
    if total == 0:
        return float("nan"), 0
    mask = np.zeros(ink.shape, dtype=bool)
    for r in rects:
        x0 = max(0, int(r[0])); y0 = max(0, int(r[1]))
        x1 = min(ink.shape[1], int(np.ceil(r[2]))); y1 = min(ink.shape[0], int(np.ceil(r[3])))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float((ink & mask).sum() / total), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of page/line images")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    ap.add_argument("--gt", default=None, help="pages.jsonl with GT line boxes")
    ap.add_argument("--sweep", default=None, help="comma-separated y_overlap values to compare")
    ap.add_argument("--y-overlap", type=float, default=DEFAULT_Y_OVERLAP)
    ap.add_argument("--gap-ratio", type=float, default=DEFAULT_GAP_RATIO)
    ap.add_argument("--pad", type=int, default=DEFAULT_PAD)
    ap.add_argument("--no-merge", action="store_true", help="report raw detector boxes only")
    ap.add_argument("--union", action="store_true",
                    help="treat each image as ONE line crop: union of all boxes")
    ap.add_argument("--ink", action="store_true", help="report ink coverage of the merged rects")
    ap.add_argument("--out", default=None, help="write line crops here")
    ap.add_argument("--manifest", default=None, help="write a JSONL manifest of the crops")
    ap.add_argument("--viz", default=None, help="write a box overlay PNG per image here")
    args = ap.parse_args()

    images = list_images(args.dir)
    if args.limit:
        images = images[:args.limit]
    if not images:
        print("no images found in %s" % args.dir)
        return 1

    gt = load_gt(args.gt) if args.gt else {}
    sweep = [float(v) for v in args.sweep.split(",")] if args.sweep else [args.y_overlap]
    if args.y_overlap not in sweep:
        sweep.insert(0, args.y_overlap)

    model = load_model()
    out_dir = Path(args.out) if args.out else None
    viz_dir = Path(args.viz) if args.viz else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    fallbacks = 0
    rows = []            # (name, n_boxes, {yo: lines}, gt_boxes, ink_ratio)
    t0 = time.time()
    for p in images:
        img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]   # BGR
        boxes = detect(model, img)
        per_yo = {}
        for yo in sweep:
            if args.no_merge:
                per_yo[yo] = [quad_to_rect(b) for b in boxes]
            elif args.union:
                per_yo[yo] = union_rect(boxes)
            else:
                per_yo[yo] = merge_boxes_to_lines(boxes, y_overlap=yo,
                                                  gap_ratio=args.gap_ratio)
        if args.union:
            # A line-crop input with NO detection still has to reach the
            # recognizer: fall back to the whole image.
            h, w = img.shape[:2]
            for yo in per_yo:
                if not per_yo[yo]:
                    per_yo[yo] = [(0.0, 0.0, float(w), float(h))]
                    fallbacks += 1
        gt_boxes = [ln["box"] for ln in gt.get(p.name, {}).get("lines", [])]
        ink, n_ink = ink_ratio(img, per_yo[args.y_overlap]) if args.ink else (float("nan"), 0)
        rows.append((p.name, len(boxes), per_yo, gt_boxes, ink, n_ink))

        if out_dir:
            gt_lines = gt.get(p.name, {}).get("lines", [])
            for i, line in enumerate(per_yo[args.y_overlap]):
                crop, rect = crop_rect(img, line, pad=args.pad)
                if crop is None:
                    continue
                name = "%s_line%02d.png" % (p.stem, i)
                Image.fromarray(crop[:, :, ::-1]).save(out_dir / name)
                # Label by GEOMETRY, never by index: one missed or extra line
                # would silently shift every transcript after it.
                text, best = None, 0.0
                for g in gt_lines:
                    ov = iou_rect((g["box"][0], g["box"][1], g["box"][2], g["box"][3]), rect)
                    if ov > best:
                        best, text = ov, g.get("text")
                manifest.append({"image": p.name, "crop": str(out_dir / name),
                                 "line": i, "rect": list(rect),
                                 "text": text if best >= 0.5 else None,
                                 "gt_iou": round(best, 3)})
        if viz_dir:
            import cv2
            vis = img.copy()
            for line in per_yo[args.y_overlap]:
                cv2.rectangle(vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])),
                              (0, 0, 255), 2)
            cv2.imwrite(str(viz_dir / (p.stem + "_lines.png")), vis)
    dt = time.time() - t0

    n = len(rows)
    print("images=%d boxes=%d (%.2f/image) %.1fs (%.3fs/image)"
          % (n, sum(r[1] for r in rows), sum(r[1] for r in rows) / n, dt, dt / n))
    print("")
    print("%-9s %9s %9s %9s %9s %9s" % ("y_overlap", "lines/img", "rec@0.3", "rec@0.5", "rec@0.7", "meanIoU"))
    for yo in sweep:
        counts = [len(r[2][yo]) for r in rows]
        ious = []
        for r in rows:
            for gb in r[3]:
                g = (gb[0], gb[1], gb[2], gb[3])
                ious.append(max((iou_rect(g, d) for d in r[2][yo]), default=0.0))
        line = "%-9.2f %9.2f" % (yo, float(np.mean(counts)))
        if ious:
            a = np.array(ious)
            line += " %9.3f %9.3f %9.3f %9.3f" % ((a > 0.3).mean(), (a > 0.5).mean(),
                                                  (a > 0.7).mean(), a.mean())
        print(line)

    counts = np.array([len(r[2][args.y_overlap]) for r in rows])
    print("")
    print("lines/image @y_overlap=%.2f: min=%d max=%d median=%.0f | 0:%d 1:%d 2-3:%d >3:%d"
          % (args.y_overlap, counts.min(), counts.max(), np.median(counts),
             int((counts == 0).sum()), int((counts == 1).sum()),
             int(((counts >= 2) & (counts <= 3)).sum()), int((counts > 3).sum())))
    if args.union:
        print("union fallback to the full image: %d image(s)" % fallbacks)
    if args.ink:
        inks = np.array([r[4] for r in rows if not np.isnan(r[4])])
        if inks.size:
            print("ink coverage of merged rects: mean=%.3f median=%.3f min=%.3f (n=%d)"
                  % (inks.mean(), np.median(inks), inks.min(), inks.size))
            one = np.array([r[4] for r in rows
                            if len(r[2][args.y_overlap]) == 1 and not np.isnan(r[4])])
            many = np.array([r[4] for r in rows
                             if len(r[2][args.y_overlap]) != 1 and not np.isnan(r[4])])
            if one.size:
                print("  coverage | 1 line  : mean=%.3f (n=%d)" % (one.mean(), one.size))
            if many.size:
                print("  coverage | !=1 line : mean=%.3f (n=%d)" % (many.mean(), many.size))
    if manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        matched = sum(1 for m in manifest if m.get("text"))
        print("wrote %d crops -> %s (manifest %s; %d labelled by GT at IoU>=0.5)"
              % (len(manifest), out_dir, args.manifest, matched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
