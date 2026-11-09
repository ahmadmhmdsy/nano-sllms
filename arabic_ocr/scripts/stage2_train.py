"""Stage 2: train the PAW detector on synthetic pages.

Gate (plan V2, Stage 2): PAW Hmean >= 90% on held-out synthetic pages.

    python arabic_ocr/scripts/stage2_train.py --config arabic_ocr/configs/stage2_paw.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr"))

from src.data.synthetic_pages import SyntheticPageDataset, collate_pages
from src.models.paw_detector import PAWDetector
from src.training.detection import (CRAFTLoss, CenterNetLoss, DBNetLoss,
                                      evaluate_detector)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "arabic_ocr" / "configs" / "stage2_paw.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    t = cfg["train"]
    e = cfg["eval"]

    epochs = args.epochs or t["epochs"]
    bs = args.batch_size or d["batch_size"]
    lr = args.lr or t["lr"]
    out = Path(args.out or t["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.limit_train = args.limit_train or 16
        args.limit_val = args.limit_val or 8
        epochs = 1

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp = bool(t.get("amp", True)) and device.type == "cuda"

    factor = int(d.get("output_stride", 2))
    target_mode = str(d.get("target_mode", "dbnet"))
    craft_k = float(d.get("craft_k", 0.2))
    craft_kx = d.get("craft_kx", None)
    craft_ky = d.get("craft_ky", None)
    sigma_k = float(d.get("sigma_k", 0.25))
    extra = ""
    if target_mode == "craft":
        extra = "  craft_k=%.2f" % craft_k
    elif target_mode == "centernet":
        extra = "  sigma_k=%.2f" % sigma_k
    print("target=%s  output_stride=%d%s" % (target_mode, factor, extra))
    train_ds = SyntheticPageDataset(d["train_dir"], d["image_height"], d["image_width"],
                                    d["min_side"], d["shrink_ratio"], limit=args.limit_train,
                                    factor=factor, target_mode=target_mode, craft_k=craft_k,
                                    sigma_k=sigma_k)
    val_ds = SyntheticPageDataset(d["val_dir"], d["image_height"], d["image_width"],
                                  d["min_side"], d["shrink_ratio"], limit=args.limit_val,
                                  factor=factor, target_mode=target_mode, craft_k=craft_k,
                                  sigma_k=sigma_k)
    print("train pages=%d  paws=%d (kept %d, dropped %d = %.1f%%)"
          % (len(train_ds), train_ds.n_paw_total, train_ds.n_paw_kept,
             train_ds.n_paw_dropped,
             100.0 * train_ds.n_paw_dropped / max(1, train_ds.n_paw_total)))
    print("val   pages=%d  paws=%d (kept %d, dropped %d = %.1f%%)"
          % (len(val_ds), val_ds.n_paw_total, val_ds.n_paw_kept,
             val_ds.n_paw_dropped,
             100.0 * val_ds.n_paw_dropped / max(1, val_ds.n_paw_total)))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=d.get("num_workers", 4), pin_memory=True,
                              collate_fn=collate_pages, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=d.get("num_workers", 4), pin_memory=True,
                            collate_fn=collate_pages)

    model = PAWDetector(in_ch=1, widths=tuple(cfg["model"]["widths"]),
                        blocks=tuple(cfg["model"]["blocks"]),
                        fpn=cfg["model"]["fpn"],
                        out_stride=int(cfg["model"].get("out_stride", 2)),
                        head_type=str(cfg["model"].get("head", "craft"))).to(device)
    print("model params=%d (%.2fM)" % (model.count_params(), model.count_params() / 1e6))

    if target_mode == "craft":
        # In craft mode evaluate_detector never reads the thresh head, so training it
        # only spends backbone capacity and gradient on a map nothing consumes. Its
        # term measured 21-61x the prob term (HANDOFF 2026-10-04 (j)).
        criterion = CRAFTLoss(pos_weight=float(t.get("pos_weight", 10.0)),
                              thresh_weight=float(t.get("thresh_weight", 1.0)))
        print("craft loss: pos_weight=%.1f  thresh_weight=%.2f"
              % (float(t.get("pos_weight", 10.0)), float(t.get("thresh_weight", 1.0))))
    elif target_mode == "centernet":
        criterion = CenterNetLoss(pos_weight=float(t.get("pos_weight", 10.0)),
                                  size_weight=float(t.get("size_weight", 1.0)),
                                  off_weight=float(t.get("off_weight", 1.0)))
        print("centernet loss: size_weight=%.2f  off_weight=%.2f"
              % (criterion.size_weight, criterion.off_weight))
    else:
        criterion = DBNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=t.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    accum = int(t.get("accum", 4))
    max_norm = float(t.get("max_grad_norm", 5.0))

    best = -1.0
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running = {}
        running_total = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            # Derive the target keys from the batch, not a hardcoded list:
            # adding a target design (CenterNet) must not require editing the trainer.
            target = {k: v.to(device, non_blocking=True)
                      for k, v in batch.items() if k not in ("boxes", "index")}
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(images)
                loss, parts = criterion(pred, target)
            # Accumulate the ACTUAL weighted objective, not one component: logging
            # only "prob" hid a thresh term 21-61x larger for a whole 30-epoch run.
            running_total += float(loss.detach())
            loss = loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n += 1
            if (step + 1) % int(t.get("log_interval", 20)) == 0:
                print("  epoch %d step %d/%d  loss=%.4f total=%.4f"
                      % (ep, step + 1, len(train_loader),
                         running.get("prob", running.get("heat", 0.0)) / max(1, n),
                         running_total / max(1, n)))
        scheduler.step()

        # eval_interval is honoured here: at stride 1 a full-val decode costs
        # ~4 min, so evaluating every epoch would dominate the run.
        do_eval = (ep % int(t.get("eval_interval", 1)) == 0) or ep == epochs - 1
        if do_eval:
            metrics = evaluate_detector(model, val_loader, device, min_side=e["min_side"],
                                        iou_thr=e["iou_thr"], stride=factor, amp=amp,
                                        target_mode=target_mode, craft_k=craft_k,
                                        craft_kx=craft_kx, craft_ky=craft_ky,
                                        cn_peak_thresh=float(e.get("peak_thresh", 0.3)),
                                        cn_topk=int(e.get("topk", 200)))
        else:
            metrics = {"precision": 0.0, "recall": 0.0, "hmean": 0.0,
                       "tp": 0, "fp": 0, "fn": 0}
        row = {"epoch": ep, "seconds": round(time.time() - t0, 1),
               "lr": round(optimizer.param_groups[0]["lr"], 8),
               "loss": round(running.get("prob", running.get("heat", 0.0)) / max(1, n), 5),
               "loss_total": round(running_total / max(1, n), 5),
               "precision": metrics["precision"], "recall": metrics["recall"],
               "hmean": metrics["hmean"], "tp": metrics["tp"],
               "fp": metrics["fp"], "fn": metrics["fn"]}
        history.append(row)
        print("epoch %d  %.1fs  loss=%.4f total=%.4f  P=%.2f R=%.2f H=%.2f  (tp=%d fp=%d fn=%d)"
              % (ep, row["seconds"], row["loss"], row["loss_total"], row["precision"],
                 row["recall"], row["hmean"], row["tp"], row["fp"], row["fn"]))
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if metrics["hmean"] > best:
            best = metrics["hmean"]
            torch.save({"model": model.state_dict(), "epoch": ep, "hmean": best,
                        "config": cfg}, out / "best.pt")
            print("  -> new best H=%.2f  saved %s" % (best, out / "best.pt"))
        gate = float(t.get("gate", 90.0))
        if best >= gate:
            print("gate %.1f met at epoch %d (H=%.2f) - stopping early" % (gate, ep, best))
            break

    report = {
        "stage": "stage2_paw_detector",
        "gate": "PAW Hmean >= 90% on held-out synthetic pages",
        "target_mode": target_mode,
        "craft_k": craft_k if target_mode == "craft" else None,
        "config": cfg,
        "params": model.count_params(),
        "train_pages": len(train_ds),
        "val_pages": len(val_ds),
        "paw_kept": train_ds.n_paw_kept,
        "paw_dropped": train_ds.n_paw_dropped,
        "best_hmean": best,
        "final": history[-1] if history else None,
        "history": history,
        "verdict": "PASS" if best >= 90.0 else "FAIL",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("")
    print("GATE: %s  (best Hmean=%.2f%%)" % (report["verdict"], best))
    print("report: %s" % (out / "report.json"))


if __name__ == "__main__":
    main()
