"""DBNet loss and PAW-detection evaluation (Stage 2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.paw_targets import (CRAFT_DECODE_THRESH, decode_boxes, decode_centernet_boxes,
                                 decode_craft_boxes, filter_boxes,
                                match_boxes, prf, unclip_boxes, unclip_boxes_exact)


class DBNetLoss(nn.Module):
    """Probability + dice + threshold + differentiable-binarization loss."""

    def __init__(self, k=50.0, thresh_weight=1.0, dice_weight=1.0, binary_weight=1.0):
        super().__init__()
        self.k = k
        self.thresh_weight = thresh_weight
        self.dice_weight = dice_weight
        self.binary_weight = binary_weight

    def forward(self, pred, target):
        prob_logit = pred["prob"]
        thresh_logit = pred["thresh"]
        # heads emit (N, 1, H, W); the target maps are (N, H, W)
        if prob_logit.dim() == 4 and prob_logit.shape[1] == 1:
            prob_logit = prob_logit[:, 0]
        if thresh_logit.dim() == 4 and thresh_logit.shape[1] == 1:
            thresh_logit = thresh_logit[:, 0]
        gt_prob = target["prob"]
        ignore = target["ignore"]
        gt_thresh = target["thresh"]
        tmask = target["tmask"]

        keep = ~ignore
        if int(keep.sum()) == 0:
            zero = prob_logit.sum() * 0.0
            return zero, {"prob": 0.0, "dice": 0.0, "thresh": 0.0, "binary": 0.0}

        p = prob_logit[keep]
        t = gt_prob[keep]
        npos = t.sum()
        nneg = t.numel() - npos
        if npos > 0:
            pos_weight = (nneg / npos).clamp(min=1.0, max=100.0)
        else:
            pos_weight = torch.ones((), device=t.device)
        l_prob = F.binary_cross_entropy_with_logits(p, t, pos_weight=pos_weight)

        prob = torch.sigmoid(prob_logit) * keep.float()
        tgt = gt_prob * keep.float()
        inter = (prob * tgt).sum()
        union = prob.sum() + tgt.sum()
        l_dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        th = torch.sigmoid(thresh_logit)
        tm = tmask.float()
        l_thresh = (torch.abs(th - gt_thresh) * tm).sum() / (tm.sum() + 1e-6)

        # binarization in logit space: sigmoid(k*(prob - thresh)) with k=50
        binlogit = self.k * (prob_logit - thresh_logit)
        binmap = torch.sigmoid(binlogit)
        l_bin = F.binary_cross_entropy_with_logits(binlogit[keep], t)

        loss = (l_prob + self.dice_weight * l_dice
                + self.thresh_weight * l_thresh + self.binary_weight * l_bin)
        return loss, {"prob": float(l_prob.detach()), "dice": float(l_dice.detach()),
                      "thresh": float(l_thresh.detach()), "binary": float(l_bin.detach())}


class CRAFTLoss(nn.Module):
    """Weighted MSE on a CRAFT region score + the auxiliary threshold map.

    A plain (unweighted) MSE on a map that is ~99% background collapses to all
    zeros; the positive weight fixes the imbalance. MSE is used rather than a
    focal loss on purpose: the decode recovers each PAW's SIZE from the blob's
    extent, so the loss must preserve the Gaussian's shape. A loss that drives
    every pixel above 0.5 to exactly 1 would flatten the peak and inflate every
    decoded box.
    """

    def __init__(self, pos_weight=10.0, thresh_weight=1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.thresh_weight = float(thresh_weight)

    def forward(self, pred, target):
        logit = pred["prob"]
        thresh_logit = pred["thresh"]
        if logit.dim() == 4 and logit.shape[1] == 1:
            logit = logit[:, 0]
        if thresh_logit.dim() == 4 and thresh_logit.shape[1] == 1:
            thresh_logit = thresh_logit[:, 0]
        t = target["prob"]
        p = torch.sigmoid(logit)
        w = 1.0 + self.pos_weight * (t > 0.5).float()
        l_prob = ((p - t) ** 2 * w).sum() / w.sum().clamp(min=1.0)

        th = torch.sigmoid(thresh_logit)
        tm = target["tmask"].float()
        l_thresh = (torch.abs(th - target["thresh"]) * tm).sum() / (tm.sum() + 1e-6)

        loss = l_prob + self.thresh_weight * l_thresh
        return loss, {"prob": float(l_prob.detach()), "dice": 0.0,
                      "thresh": float(l_thresh.detach()), "binary": 0.0}


class CenterNetLoss(nn.Module):
    """Weighted MSE on the centre heatmap + L1 on w/h and the offset.

    The heatmap is the only thing that says WHERE a PAW is; w/h and the offset
    are supervised at the centre pixels alone, which is exactly where they are
    read at decode time.

    The heat term is a weighted MSE, NOT a penalty-reduced focal. A focal loss's
    raw negative sum over a 360x480 map with only ~300 positive centre pixels is
    ~2000x the size term (measured: heat loss 63,284 at init vs size 20), so at
    lr 1e-3 it drives every logit negative and the model emits no detections at
    all (measured: all heat logits < 0 by step 30, zero boxes at epoch 6). The
    decode reads each PAW's size from the regression, so the heatmap only has to
    localise -- the same reason CRAFTLoss uses weighted MSE.
    """

    def __init__(self, pos_weight=10.0, size_weight=1.0, off_weight=1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.size_weight, self.off_weight = float(size_weight), float(off_weight)

    def forward(self, pred, target):
        logit = pred["heat"]
        if logit.dim() == 4 and logit.shape[1] == 1:
            logit = logit[:, 0]
        t = target["heat"]
        p = torch.sigmoid(logit)
        w = 1.0 + self.pos_weight * (t > 0.5).float()
        l_heat = ((p - t) ** 2 * w).sum() / w.sum().clamp(min=1.0)

        # (B, 1, h, w) so it broadcasts against the (B, 2, h, w) size/off maps.
        m = target["mask"].unsqueeze(1).float()
        nf = m.sum().clamp(min=1.0)
        l_size = (torch.abs(pred["size"] - target["size"]) * m).sum() / (2.0 * nf)
        l_off = (torch.abs(pred["off"] - target["off"]) * m).sum() / (2.0 * nf)

        loss = l_heat + self.size_weight * l_size + self.off_weight * l_off
        return loss, {"heat": float(l_heat.detach()), "size": float(l_size.detach()),
                      "off": float(l_off.detach())}


@torch.no_grad()
def evaluate_detector(model, loader, device, min_side=6, iou_thr=0.5, stride=2,
                      amp=True, max_batches=0, bin_thresh=0.3, shrink_ratio=0.4,
                      unclip_ratio=None, use_learned_thresh=False,
                      target_mode="dbnet", craft_k=0.2, craft_kx=None, craft_ky=None,
                      cn_peak_thresh=0.3, cn_topk=200):
    """PAW Hmean over a loader. Boxes are matched greedily at IoU >= iou_thr.

    DBNet path (target_mode="dbnet"): the probability head is trained on the
    SHRUNK box, so a decoded component is smaller than the PAW it came from and
    must be unclipped back to the original frame before it can be compared with
    the ground-truth boxes. min_side is applied AFTER unclipping so the same
    filter the targets used applies to the predictions as well.

    CRAFT path (target_mode="craft"): the box size is read off the half-max level
    set of the region score, so this branch binarizes at CRAFT_DECODE_THRESH (0.5)
    instead of bin_thresh and passes that same threshold to decode_craft_boxes.
    Decoding a CRAFT mask at the wrong threshold is not a small error: on perfect
    target maps tau=0.3 with the 0.5 constant scores 74.5%, tau=0.5 scores 92.3%.
    """
    model.eval()
    tp = fp = fn = 0
    n_images = 0
    per_image = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(images)
        if target_mode == "centernet":
            prob = torch.sigmoid(out["heat"]).float().cpu().numpy()
        else:
            prob = torch.sigmoid(out["prob"]).float().cpu().numpy()
            thresh = torch.sigmoid(out["thresh"]).float().cpu().numpy()
        for b in range(prob.shape[0]):
            if target_mode == "centernet":
                # One peak per PAW: no binarization, no level set, no merging.
                # The box is the regressed w/h at the peak, recentred by the
                # regressed sub-pixel offset.
                pred = decode_centernet_boxes(
                    prob[b, 0], out["size"][b].float().cpu().numpy(),
                    out["off"][b].float().cpu().numpy(), stride, min_side,
                    topk=cn_topk, peak_thresh=cn_peak_thresh)
            elif target_mode == "craft":
                # The CRAFT size decode reads the box size off the half-max level set
                # of the region score, so it must binarize at CRAFT_DECODE_THRESH and
                # use the blob constant for THAT threshold (craft_blob_scale). Using
                # the 0.5 constant on a 0.3 mask inflates every box ~32%.
                mask = prob[b, 0] > CRAFT_DECODE_THRESH
                pred = decode_craft_boxes(mask, craft_k, stride, min_side,
                                          thresh=CRAFT_DECODE_THRESH,
                                          kx=craft_kx, ky=craft_ky)
            else:
                if use_learned_thresh:
                    mask = prob[b, 0] > thresh[b, 0]
                else:
                    mask = prob[b, 0] > bin_thresh
                pred = decode_boxes(mask, min_side=1, stride=stride)
                if unclip_ratio is None:
                    pred = unclip_boxes_exact(pred, shrink_ratio)
                else:
                    pred = unclip_boxes(pred, unclip_ratio)
                pred, _ = filter_boxes(pred, min_side)
                pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
            gt = batch["boxes"][b].numpy()
            t, f_p, f_n = match_boxes(pred, gt, iou_thr)
            tp += t
            fp += f_p
            fn += f_n
            n_images += 1
            per_image.append({"tp": t, "fp": f_p, "fn": f_n,
                              "n_pred": int(len(pred)), "n_gt": int(len(gt))})
    p, r, h = prf(tp, fp, fn)
    return {"images": n_images, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(100 * p, 4), "recall": round(100 * r, 4),
            "hmean": round(100 * h, 4), "per_image": per_image}
