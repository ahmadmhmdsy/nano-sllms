r"""Laya-line L2 (E-63): stage A mixture pretrain -> stage B typed fine-tune.

Pre-registered: research/EXPERIMENTS.md E-63. Same notebook-parity recipe as
E-62 per stage (eff 64, lr 2.5e-5 / 1e-4, cosine, fp16, clip 1.0, seed 42).
Auto-resumes per stage at epoch boundaries with zero flags.

Usage: & .\.venv\Scripts\python.exe laya/scripts/train_l2.py --config configs/laya_l2.yaml
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_head import LayaDecisionModel, collate, soft_ce_loss  # noqa: E402
import train_l1 as t1  # noqa: E402  (forward_batch / evaluate helpers)


def per_source_acc(model, items, pad_id, device):
    by_src = {}
    for it in items:
        by_src.setdefault(it["workflow"], []).append(it)
    agg = {s: round(t1.evaluate(model, its, pad_id, device)["acc"], 4)
           for s, its in sorted(by_src.items())}
    macro = round(sum(agg.values()) / max(1, len(agg)), 4)
    return {"per_source": agg, "macro": macro, "overall_n": len(items)}


def run_stage(model, cfg, items, eval_items, pad_id, device, tag, epochs, writer,
              eval_fn):
    out_dir = cfg["out_dir"]
    micro = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": float(cfg["lr_encoder"])},
        {"params": head_params, "lr": float(cfg["lr_head"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")

    def make_batches():
        order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
        chunks = [order[i:i + micro] for i in range(0, len(order), micro)]
        random.shuffle(chunks)
        return chunks

    micro_per_epoch = math.ceil(len(items) / micro)
    total_updates = math.ceil(micro_per_epoch / accum) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates,
                                                       eta_min=1e-6)

    start_epoch, step, update = 0, 0, 0
    curve = []
    last_path = os.path.join(out_dir, tag + "_last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["sched"])
        start_epoch, step, update = int(ck["epoch"]), int(ck["step"]), int(ck["update"])
        curve = ck.get("eval_curve", [])
        print("[l2:%s] resumed epoch %d update %d" % (tag, start_epoch, update), flush=True)

    model.train()
    t0 = time.time()
    peak_mib = 0.0
    loss_acc, loss_n = 0.0, 0
    for epoch in range(start_epoch, epochs):
        for chunk in make_batches():
            batch, logp, _ = t1.forward_batch(model, collate([items[i] for i in chunk],
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
                if update % 50 == 0:
                    mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_mib = max(peak_mib, mib)
                    writer.add_scalar("train/%s_loss" % tag, loss_acc / max(1, loss_n), update)
                    print("[l2:%s] update %d/%d loss %.4f peak %.0f MiB %.0fs" %
                          (tag, update, total_updates, loss_acc / max(1, loss_n), mib,
                           time.time() - t0), flush=True)
                    loss_acc, loss_n = 0.0, 0
                if update % int(cfg["save_every_steps"]) == 0:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                                "step": step, "update": update, "epoch": epoch,
                                "eval_curve": curve}, last_path)
        m = eval_fn(model, eval_items, pad_id, device)
        curve.append({"epoch": epoch + 1, **m})
        print("[l2:%s] epoch %d eval %s" % (tag, epoch + 1, json.dumps(m)[:240]), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                    "step": step, "update": update, "epoch": epoch + 1,
                    "eval_curve": curve}, last_path)
    return {"updates": update, "wall_s": round(time.time() - t0, 1),
            "peak_vram_mib": round(peak_mib, 1), "eval_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l2.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="1-batch validation run: tiny slices, 1 epoch/stage, out_dir+_smoke")
    args = ap.parse_args()
    import yaml
    from torch.utils.tensorboard import SummaryWriter
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))
    device = "cuda"
    out_dir = cfg["out_dir"] + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        cfg["out_dir"] = out_dir  # run_stage reads cfg["out_dir"] - keep smoke off real checkpoints
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pad_id = tok.pad_token_id

    def eval_A(model, items, pad, dev):
        r = per_source_acc(model, items, pad, dev)
        return {"macro": r["macro"], "per_source": r["per_source"]}

    def eval_B(model, items, pad, dev):
        return t1.evaluate(model, items, pad, dev)

    mixture_train = torch.load(cfg["mixture_train"], weights_only=False)
    mixture_held = torch.load(cfg["mixture_heldout"], weights_only=False)
    typed_train = torch.load(cfg["typed_train"], weights_only=False)
    typed_test = torch.load(cfg["typed_test"], weights_only=False)
    print("[l2] mixture %d/%d, typed %d/%d" %
          (len(mixture_train), len(mixture_held), len(typed_train), len(typed_test)),
          flush=True)
    if args.smoke:
        mixture_train, mixture_held = mixture_train[:64], mixture_held[:32]
        typed_train, typed_test = typed_train[:64], typed_test[:32]
        print("[l2] SMOKE: tiny slices, 1 epoch/stage", flush=True)

    model = LayaDecisionModel(cfg["encoder"], head_layers=int(cfg["head_layers"]),
                              dropout=float(cfg["dropout"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print("[l2] STAGE A: mixture pretrain (%d epochs)" % int(cfg["epochs_stageA"]), flush=True)
    sa = run_stage(model, cfg, mixture_train, mixture_held, pad_id, device, "A",
                   1 if args.smoke else int(cfg["epochs_stageA"]), writer, eval_A)

    print("[l2] STAGE B: typed-decisions fine-tune (%d epochs)" % int(cfg["epochs_stageB"]),
          flush=True)
    sb = run_stage(model, cfg, typed_train, typed_test, pad_id, device, "B",
                   1 if args.smoke else int(cfg["epochs_stageB"]), writer, eval_B)

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    summary = {"params_m": round(n_params / 1e6, 2), "stageA": sa, "stageB": sb,
               "config": {k: cfg[k] for k in ("encoder", "epochs_stageA",
                                              "epochs_stageB", "micro_batch",
                                              "grad_accum", "seed")}}
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("[l2] DONE A(upd=%d %.0fs) B(upd=%d %.0fs)" %
          (sa["updates"], sa["wall_s"], sb["updates"], sb["wall_s"]), flush=True)
    writer.close()


if __name__ == "__main__":
    main()
