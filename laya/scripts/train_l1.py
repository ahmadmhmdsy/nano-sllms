r"""Laya-line L1: MiniLM-L12-H384 + Laya decision head, soft-CE only.

Pre-registered: research/EXPERIMENTS.md E-62. Notebook-parity recipe:
4 epochs, micro 8 x accum 8 (eff 64), lr 2.5e-5 enc / 1e-4 head, wd 0.01,
cosine -> 1e-6, fp16 autocast, clip 1.0, seed 42. Auto-resumes from
runs/laya/l1/last.pt at epoch boundaries with zero flags (crash-safe).

Usage: & .\.venv\Scripts\python.exe laya/scripts/train_l1.py --config configs/laya_l1.yaml [--probe]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import (LayaDecisionModel, collate, option_log_probs,  # noqa: E402
                       soft_ce_loss)


def forward_batch(model, batch, device):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["marker_pos"], batch["marker_batch"], batch["qtype"])
    logp, blocks = option_log_probs(logits.float(), batch["n_options"])
    return batch, logp, blocks


def evaluate(model, items, pad_id, device, micro_batch=16):
    model.eval()
    correct, soft_sum, brier_sum, n = 0, 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), micro_batch):
            chunk = items[i:i + micro_batch]
            batch, logp, _ = forward_batch(model, collate(chunk, pad_id), device)
            ofs = 0
            for b, cnt in enumerate(batch["n_options"]):
                p = logp[ofs:ofs + cnt].exp()
                gold_idx = int(chunk[b]["gold_idx"])
                pred = int(p.argmax().item())
                correct += int(pred == gold_idx)
                soft_sum += float(p[gold_idx].item())
                onehot = torch.zeros_like(p)
                onehot[gold_idx] = 1.0
                brier_sum += float(((p - onehot) ** 2).mean().item())
                n += 1
                ofs += cnt
    model.train()
    return {"acc": correct / max(1, n), "soft_acc": soft_sum / max(1, n),
            "brier": brier_sum / max(1, n), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l1.yaml")
    ap.add_argument("--probe", action="store_true",
                    help="one fwd+bwd micro-step, print peak VRAM, exit")
    args = ap.parse_args()
    import yaml
    from torch.utils.tensorboard import SummaryWriter
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))
    device = "cuda"
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    train_items = torch.load(cfg["data_train"], weights_only=False)
    test_items = torch.load(cfg["data_test"], weights_only=False)
    print("[l1] train %d items, test %d items" % (len(train_items), len(test_items)),
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id

    model = LayaDecisionModel(cfg["encoder"], head_layers=int(cfg["head_layers"]),
                              dropout=float(cfg["dropout"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": float(cfg["lr_encoder"])},
        {"params": head_params, "lr": float(cfg["lr_head"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")

    micro = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])

    def make_batches():
        order = sorted(range(len(train_items)),
                       key=lambda i: len(train_items[i]["input_ids"]))
        chunks = [order[i:i + micro] for i in range(0, len(order), micro)]
        random.shuffle(chunks)
        return chunks

    micro_per_epoch = math.ceil(len(train_items) / micro)
    total_updates = math.ceil(micro_per_epoch / accum) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates,
                                                       eta_min=1e-6)

    start_epoch, step, update = 0, 0, 0
    eval_curve = []
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["sched"])
        start_epoch = int(ck.get("epoch", 0))
        step, update = int(ck["step"]), int(ck["update"])
        eval_curve = ck.get("eval_curve", [])
        print("[l1] resumed %s: epoch %d, update %d (partial epoch redone)"
              % (last_path, start_epoch, update), flush=True)
    else:
        print("[l1] fresh start", flush=True)

    model.train()
    t0 = time.time()
    peak_mib = 0.0
    loss_acc, loss_n = 0.0, 0

    if args.probe:
        chunks = make_batches()
        batch, logp, _ = forward_batch(model, collate([train_items[i] for i in chunks[0]],
                                                       pad_id), device)
        loss = soft_ce_loss(logp, None, batch["targets"], batch["n_options"])
        scaler.scale(loss).backward()
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print("[probe] fwd+bwd OK; peak allocated %.0f MiB; params %.1fM; "
              "micro_batch=%d" % (peak, n_params / 1e6, micro), flush=True)
        writer.close()
        return

    for epoch in range(start_epoch, epochs):
        chunks = make_batches()
        for chunk in chunks:
            batch, logp, _ = forward_batch(model, collate([train_items[i] for i in chunk],
                                                           pad_id), device)
            loss = soft_ce_loss(logp, None, batch["targets"], batch["n_options"])
            scaler.scale(loss / accum).backward()
            loss_acc += float(loss.item())
            loss_n += 1
            step += 1
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                update += 1
                if update % 20 == 0:
                    mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_mib = max(peak_mib, mib)
                    writer.add_scalar("train/loss", loss_acc / max(1, loss_n), update)
                    writer.add_scalar("train/lr_head", sched.get_last_lr()[1], update)
                    print("[l1] update %d/%d loss %.4f peak %.0f MiB %.0fs" %
                          (update, total_updates, loss_acc / max(1, loss_n), mib,
                           time.time() - t0), flush=True)
                    loss_acc, loss_n = 0.0, 0
                if update % int(cfg["save_every_steps"]) == 0:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                                "step": step, "update": update, "epoch": epoch,
                                "eval_curve": eval_curve}, last_path)
        m = evaluate(model, test_items, pad_id, device)
        eval_curve.append({"epoch": epoch + 1, **m})
        writer.add_scalar("eval/acc", m["acc"], update)
        writer.add_scalar("eval/brier", m["brier"], update)
        print("[l1] epoch %d eval acc %.4f soft %.4f brier %.4f" %
              (epoch + 1, m["acc"], m["soft_acc"], m["brier"]), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                    "step": step, "update": update, "epoch": epoch + 1,
                    "eval_curve": eval_curve}, last_path)

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    summary = {
        "updates": update, "epochs": epochs,
        "wall_s": round(time.time() - t0, 1),
        "peak_vram_mib": round(peak_mib, 1),
        "params_m": round(n_params / 1e6, 2),
        "eval_curve": eval_curve,
        "config": {k: cfg[k] for k in ("encoder", "epochs", "micro_batch",
                                        "grad_accum", "lr_encoder", "lr_head", "seed")},
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("[l1] DONE updates=%d wall=%.0fs peak=%.0fMiB params=%.1fM" %
          (update, time.time() - t0, peak_mib, n_params / 1e6), flush=True)
    writer.close()


if __name__ == "__main__":
    main()
