"""Torch dataset over synthetic pages rendered by src/data/synthetic.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .paw_targets import (build_centernet_targets, build_craft_prob, build_targets,
                             downsample_targets, filter_boxes, paw_boxes_from_record)


class SyntheticPageDataset(Dataset):
    """Pages + detection targets. Pages are resized to (height, width); boxes follow.

    target_mode="dbnet" builds DBNet shrunk-box targets; "craft" replaces the
    probability map with a CRAFT-style Gaussian region score (the threshold map is
    kept either way). See build_craft_prob for why the region score decodes better.
    """

    def __init__(self, root, height=720, width=960, min_side=6, shrink_ratio=0.4, limit=0,
                 factor=2, target_mode="dbnet", craft_k=0.2, sigma_k=0.25):
        self.root = Path(root)
        with (self.root / "pages.jsonl").open(encoding="utf-8") as f:
            self.recs = [json.loads(l) for l in f if l.strip()]
        if limit:
            self.recs = self.recs[:limit]
        self.H, self.W = int(height), int(width)
        self.min_side = float(min_side)
        self.shrink_ratio = shrink_ratio
        self.factor = int(factor)
        self.target_mode = str(target_mode)
        self.craft_k = float(craft_k)
        self.sigma_k = float(sigma_k)
        kept = dropped = 0
        for rec in self.recs:
            sx = self.W / rec["width"]
            sy = self.H / rec["height"]
            boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                     for b in paw_boxes_from_record(rec["lines"])]
            k, d = filter_boxes(boxes, self.min_side)
            kept += len(k)
            dropped += len(d)
        self.n_paw_total = kept + dropped
        self.n_paw_kept = kept
        self.n_paw_dropped = dropped

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        sx = self.W / rec["width"]
        sy = self.H / rec["height"]
        with Image.open(self.root / rec["image"]) as im:
            img = im.convert("L").resize((self.W, self.H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                 for b in paw_boxes_from_record(rec["lines"])]
        boxes, _ = filter_boxes(boxes, self.min_side)
        dprob, ignore, thresh, tmask = build_targets(boxes, self.H, self.W, self.shrink_ratio)
        if self.target_mode == "craft":
            # region score is rasterised at output resolution; only the auxiliary
            # threshold map still needs downsampling. CRAFT has no ambiguous rim.
            prob = build_craft_prob(boxes, self.H, self.W, self.craft_k, self.factor)
            # ignore must stay at full resolution: only it and thresh/tmask are downsampled
            ignore = np.zeros((self.H, self.W), dtype=bool)
            if self.factor > 1:
                _, _, thresh, tmask = downsample_targets(
                    dprob, ignore, thresh, tmask, self.factor)
        else:
            prob = dprob
            if self.factor > 1:
                prob, ignore, thresh, tmask = downsample_targets(
                    prob, ignore, thresh, tmask, self.factor)
        if self.target_mode == "centernet":
            # centre heatmap + direct w/h regression, rasterised at output
            # resolution like the CRAFT region score (no downsampling step).
            heat, size, off, cmask = build_centernet_targets(
                boxes, self.H, self.W, self.factor, self.sigma_k)
            return {
                "image": x,
                "heat": torch.from_numpy(heat),
                "size": torch.from_numpy(size),
                "off": torch.from_numpy(off),
                "mask": torch.from_numpy(cmask),
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "index": i,
            }
        return {
            "image": x,
            "prob": torch.from_numpy(prob),
            "ignore": torch.from_numpy(ignore),
            "thresh": torch.from_numpy(thresh),
            "tmask": torch.from_numpy(tmask),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "index": i,
        }


def collate_pages(batch):
    """Stack the tensor targets; the key set depends on target_mode."""
    out = {"image": torch.stack([b["image"] for b in batch]),
           "boxes": [b["boxes"] for b in batch],
           "index": [b["index"] for b in batch]}
    for k in ("prob", "ignore", "thresh", "tmask", "heat", "size", "off", "mask"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out
