"""Stage 2 hybrid detector - PP-OCRv5_mobile_det on GPU.

Loads the pretrained PaddleOCR detector with paddle.jit.load() (no paddleocr
package needed), preprocesses images (BGR, resize_long=960, ImageNet normalize,
CHW), runs inference on GPU, and post-processes with DBPostProcess to get text
line/word boxes.

Usage:
  & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\detector_gpu.py --probe
  & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\detector_gpu.py --image <path>
  & .\.venv-paddle-gpu\Scripts\python.exe arabic_ocr\scripts\detector_gpu.py --dir <path> --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import paddle
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "arabic_ocr/models/ppocrv5_mobile_det"

# DBPostProcess config. thresh / max_candidates / unclip_ratio are the model's
# own inference.yml values; box_thresh is deliberately LOWERED (0.6 -> 0.3),
# see the measurement note below.
DB_THRESH = 0.3
DB_BOX_THRESH = 0.3
DB_MAX_CANDIDATES = 1000
DB_UNCLIP_RATIO = 1.5

# box_thresh: the model's stock 0.6 is incompatible with the 7px dilation. The
# dilation merges strokes into blobs whose tight minAreaRect encloses the
# inter-stroke whitespace, so the box score (p50 over all contours = 0.55) falls
# under 0.6 and the box is dropped. Measured on 30 synthetic val pages / 240 GT
# lines (2026-10-05): box_thresh 0.6 -> 3.2 boxes/page, recall@IoU0.5 0.404;
# 0.3 -> 8.5 boxes/page (8 GT lines/page), recall 0.912; 0.0 -> 10.0 boxes/page,
# recall 0.933. 0.3 is adopted: the gate (>= 0.90) still passes with 15% fewer
# boxes to feed the recognizer. Score mode (raw bitmap vs prob map) is immaterial.
#
# Local additions, measured on the 30-page synthetic val split (2026-10-05).
# DB's probability map is sharp-edged, so a single ink stroke often splits into
# several contours; a 7px dilation re-joins them. The detector fires on the ink
# itself, while a line box additionally carries ascender/descender/diacritic
# padding, so a 1.8x vertical expansion about the box centre recovers the line
# extent (recall@IoU0.5 0.40 -> 0.93 on the synthetic val pages).
DB_DILATE = 7
DB_V_EXPAND = 1.8

# ImageNet normalization (from inference.yml)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    """Load the detector model on GPU with paddle.jit.load()."""
    prefix = str(MODEL_DIR / "inference")
    print(f"Loading detector from {prefix}...")
    m = paddle.jit.load(prefix)
    m.eval()
    paddle.set_device("gpu")
    print(f"Model loaded. Device: {paddle.get_device()}")
    return m


def preprocess(image, resize_long=960):
    """Preprocess image for the detector.
    
    Args:
        image: HWC BGR image (numpy array).
        resize_long: Resize the longer side to this value.
    
    Returns:
        (tensor, orig_size, scale_factor) where tensor is CHW float32.
    """
    import cv2
    
    h, w = image.shape[:2]
    orig_size = (h, w)
    
    # Resize: longer side to resize_long, keep aspect ratio
    scale = resize_long / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to multiple of 32
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Normalize BGR directly (PaddleOCR applies mean/std to BGR channels)
    normalized = (padded.astype(np.float32) / 255.0 - MEAN) / STD
    chw = normalized.transpose(2, 0, 1)  # CHW
    
    return chw, orig_size, scale


def postprocess(prob_map, orig_size, scale,
                thresh=DB_THRESH, box_thresh=DB_BOX_THRESH,
                max_candidates=DB_MAX_CANDIDATES,
                unclip_ratio=DB_UNCLIP_RATIO,
                dilate_kernel=DB_DILATE, v_expand=DB_V_EXPAND):
    """DBPostProcess: threshold, find contours, unclip, filter.
    
    Args:
        prob_map: 1-channel probability map (HxW).
        orig_size: (h, w) of the original image.
        scale: scale factor used in preprocessing.
        dilate_kernel: square dilation kernel applied before contour extraction
            so fragmented strokes merge into one box (0 disables).
        v_expand: vertical expansion factor about each box centre; >1 recovers
            the line extent around the ink stroke (1.0 disables).
    
    Returns:
        List of boxes, each as 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    """
    import cv2
    
    h, w = orig_size
    
    # Threshold. The bitmap stays UN-dilated: DB scores each box on the raw
    # bitmap (PaddleOCR's box_score_fast), and dilation only helps contour
    # extraction.
    bitmap = (prob_map > thresh).astype(np.uint8)
    
    # Optional dilation (see dilate_kernel) - for contour extraction only.
    binary = bitmap
    if dilate_kernel > 0:
        binary = cv2.dilate(binary, np.ones((dilate_kernel, dilate_kernel), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours[:max_candidates]:
        area = cv2.contourArea(contour)
        if area < 10:
            continue
        
        # Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box_np = cv2.boxPoints(rect).astype(np.float32)
        
        # Filter by box_thresh: score the TIGHT rect against the raw bitmap.
        # Scoring the dilated + unclipped box instead dilutes the mean with
        # background pixels and rejects nearly every small PAW box (measured
        # 2026-10-05: 1 box/page and recall@0.5 0.12, vs 8 boxes/page and 0.93).
        mask = np.zeros(bitmap.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [box_np.astype(np.int32)], 1)
        if mask.sum() > 0 and (bitmap * mask).sum() / mask.sum() < box_thresh:
            continue
        
        # Unclip
        center = box_np.mean(axis=0)
        peri = cv2.arcLength(contour, True)
        if peri > 0:
            distance = area * (unclip_ratio - 1.0) / peri
            for i in range(4):
                diff = box_np[i] - center
                norm = np.linalg.norm(diff)
                if norm > 0:
                    box_np[i] = box_np[i] + diff / norm * distance
        
        # Scale back to original image coordinates
        box_np = box_np / scale
        
        # Optional vertical expansion about the box centre (see v_expand).
        if v_expand != 1.0:
            cy = box_np[:, 1].mean()
            box_np[:, 1] = cy + (box_np[:, 1] - cy) * v_expand
        
        # Clip to image bounds
        box_np[:, 0] = np.clip(box_np[:, 0], 0, w)
        box_np[:, 1] = np.clip(box_np[:, 1], 0, h)
        
        boxes.append(box_np.tolist())
    
    return boxes


def detect(model, image, **postprocess_kwargs):
    """Run detection on an image.
    
    Args:
        model: loaded paddle model.
        image: HWC BGR image.
        **postprocess_kwargs: overrides forwarded to postprocess().
    
    Returns:
        List of boxes in original image coordinates.
    """
    chw, orig_size, scale = preprocess(image)
    
    # Run inference
    input_tensor = paddle.to_tensor(chw[np.newaxis, :, :, :], dtype="float32")
    with paddle.no_grad():
        output = model(input_tensor)
    
    # Get probability map
    prob_map = output.numpy()[0, 0]  # HxW
    
    # Post-process
    boxes = postprocess(prob_map, orig_size, scale, **postprocess_kwargs)
    
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Probe GPU and model loading")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--limit", type=int, default=10, help="Max images to process")
    args = parser.parse_args()
    
    if args.probe:
        print("=== GPU Probe ===")
        print(f"paddle version: {paddle.__version__}")
        print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
        print(f"CUDA device count: {paddle.device.cuda.device_count()}")
        paddle.set_device("gpu")
        print(f"Device: {paddle.get_device()}")
        
        print("\n=== Model Loading ===")
        model = load_model()
        
        print("\n=== Test Inference ===")
        # Create a dummy image
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        boxes = detect(model, dummy)
        t1 = time.time()
        print(f"Dummy image (640x480): {len(boxes)} boxes in {t1-t0:.3f}s")
        
        print("\nProbe PASS")
        return
    
    if args.image:
        model = load_model()
        image = np.array(Image.open(args.image).convert("RGB"))
        # RGB -> BGR for the model
        image = image[:, :, ::-1]
        t0 = time.time()
        boxes = detect(model, image)
        t1 = time.time()
        print(f"Image: {args.image}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        print(f"Detected {len(boxes)} boxes in {t1-t0:.3f}s")
        for i, box in enumerate(boxes[:5]):
            print(f"  Box {i}: {box}")
        return
    
    if args.dir:
        model = load_model()
        image_dir = Path(args.dir)
        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
        images = images[:args.limit]
        print(f"Processing {len(images)} images from {image_dir}")
        
        total_boxes = 0
        total_time = 0
        for img_path in images:
            image = np.array(Image.open(img_path).convert("RGB"))
            image = image[:, :, ::-1]
            t0 = time.time()
            boxes = detect(model, image)
            t1 = time.time()
            total_boxes += len(boxes)
            total_time += t1 - t0
            print(f"  {img_path.name}: {len(boxes)} boxes in {t1-t0:.3f}s")
        
        print(f"\nTotal: {total_boxes} boxes in {total_time:.3f}s "
              f"({total_time/len(images):.3f}s/image)")
        return
    
    print("No action specified. Use --probe, --image, or --dir")


if __name__ == "__main__":
    main()
