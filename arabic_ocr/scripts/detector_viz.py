"""Visualize detector boxes on images for quality validation.

Usage:
  & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\detector_viz.py --image <path> --out <path>
  & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\detector_viz.py --dir <path> --limit 5 --outdir <path>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

# Reuse the detector functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_gpu import load_model, detect, preprocess, postprocess

import cv2


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw boxes on a copy of the image."""
    vis = image.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--out", type=str, help="Output path for visualization")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--limit", type=int, default=5, help="Max images to process")
    parser.add_argument("--outdir", type=str, help="Output directory for visualizations")
    args = parser.parse_args()
    
    model = load_model()
    
    if args.image:
        image = np.array(Image.open(args.image).convert("RGB"))
        image = image[:, :, ::-1]  # RGB -> BGR
        t0 = time.time()
        boxes = detect(model, image)
        t1 = time.time()
        vis = draw_boxes(image, boxes)
        out_path = args.out or "detector_viz.png"
        cv2.imwrite(out_path, vis)
        print(f"Image: {args.image}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        print(f"Detected {len(boxes)} boxes in {t1-t0:.3f}s")
        print(f"Saved visualization to {out_path}")
        
        # Print box statistics
        if boxes:
            areas = []
            aspect_ratios = []
            for box in boxes:
                pts = np.array(box)
                w = np.linalg.norm(pts[1] - pts[0])
                h = np.linalg.norm(pts[3] - pts[0])
                areas.append(w * h)
                aspect_ratios.append(w / h if h > 0 else 0)
            print(f"Box areas: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}")
            print(f"Aspect ratios: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}")
        return
    
    if args.dir:
        image_dir = Path(args.dir)
        outdir = Path(args.outdir) if args.outdir else Path("detector_viz")
        outdir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
        images = images[:args.limit]
        print(f"Processing {len(images)} images from {image_dir}")
        
        for img_path in images:
            image = np.array(Image.open(img_path).convert("RGB"))
            image = image[:, :, ::-1]
            t0 = time.time()
            boxes = detect(model, image)
            t1 = time.time()
            vis = draw_boxes(image, boxes)
            out_path = outdir / f"{img_path.stem}_det.png"
            cv2.imwrite(str(out_path), vis)
            print(f"  {img_path.name}: {len(boxes)} boxes in {t1-t0:.3f}s -> {out_path.name}")
        
        print(f"\nVisualizations saved to {outdir}")
        return
    
    print("No action specified. Use --image or --dir")


if __name__ == "__main__":
    main()
