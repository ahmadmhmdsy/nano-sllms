"""Re-evaluate a Stage 2 PAW-detector checkpoint with the CORRECTED metric.

The first Stage 2 run selected checkpoints with an eval that compared SHRUNK
decoded components against FULL ground-truth boxes (IoU ~0.34, far below the 0.5
match threshold) and applied min_side to the shrunk boxes. Its reported Hmean was
therefore not a detection-quality number. This script re-scores a checkpoint with
the unclipped metric and can sweep the binarisation threshold and unclip ratio.

    python arabic_ocr/scripts/stage2_reeval.py --ckpt runs/stage2_paw/best.pt --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr"))

from src.data.synthetic_pages import SyntheticPageDataset, collate_pages
from src.models.paw_detector import PAWDetector
from src.training.detection import evaluate_detector

KEYS = ("images", "tp", "fp", "fn", "precision", "recall", "hmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "arabic_ocr" / "configs" / "stage2_paw.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, m, ev = cfg["data"], cfg["model"], cfg["eval"]
    factor = int(d.get("output_stride", 2))
    root = ROOT / (d["val_dir"] if args.split == "val" else d["train_dir"])
    ds = SyntheticPageDataset(root, height=d["image_height"], width=d["image_width"],
                              min_side=d["min_side"], shrink_ratio=d["shrink_ratio"],
                              limit=args.limit, factor=factor)
    print("split=%s pages=%d kept_paws=%d dropped=%d" %
          (args.split, len(ds), ds.n_paw_kept, ds.n_paw_dropped))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_pages)

    model = PAWDetector(widths=tuple(m["widths"]), blocks=tuple(m["blocks"]), fpn=m["fpn"])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("ckpt=%s epoch=%s stored_hmean=%s" %
          (args.ckpt, state.get("epoch") if isinstance(state, dict) else "?",
           state.get("hmean") if isinstance(state, dict) else "?"))

    results = {}
    modes = [("exact", {}), ("r1.5", {"unclip_ratio": 1.5}), ("r2.1", {"unclip_ratio": 2.1})]
    if args.sweep:
        for bt in (0.3, 0.4, 0.5):
            for name, kw in modes:
                r = evaluate_detector(model, loader, device, min_side=ev["min_side"],
                                      iou_thr=ev["iou_thr"], stride=factor,
                                      shrink_ratio=d["shrink_ratio"],
                                      bin_thresh=bt, **kw)
                results["bt%.1f_%s" % (bt, name)] = {k: r[k] for k in KEYS}
                print("bt=%.1f unclip=%-5s P=%.2f R=%.2f H=%.2f" %
                      (bt, name, r["precision"], r["recall"], r["hmean"]), flush=True)
    else:
        r = evaluate_detector(model, loader, device, min_side=ev["min_side"],
                              iou_thr=ev["iou_thr"], stride=factor,
                              shrink_ratio=d["shrink_ratio"])
        results["default"] = {k: r[k] for k in KEYS}
        print("P=%.2f R=%.2f H=%.2f  (tp %d fp %d fn %d, %d images)" %
              (r["precision"], r["recall"], r["hmean"], r["tp"], r["fp"], r["fn"],
               r["images"]))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
