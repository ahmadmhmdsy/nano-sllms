"""M1 VRAM probe: target config on synthetic tokens (no data, no network).

Same settings as train.py (fp16 + grad checkpointing + AdamW + accum), so the
peak VRAM measured here is the honest fit test for the 250M config.
Decision gate (PLAN.md §M1): peak <= 6.5 GB -> T approved; 6.5-7.8 -> 8-bit
Adam re-probe; OOM -> stay at pilot scale.
Run: .venv/Scripts/python scripts/vram_probe.py --config configs/target.yaml
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_optimizer(model, tcfg):
    """Honor the config's optim so per-arm peaks are honest (Milestone B)."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = tcfg.get("optim", "adamw_torch")
    if name == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=1e-4, betas=(0.9, 0.95),
                                   weight_decay=0.1)
    if name != "adamw_torch":
        raise SystemExit(f"vram_probe supports adamw_torch/adamw_bnb_8bit, got {name}")
    import torch
    return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/target.yaml")
    ap.add_argument("--steps", type=int, default=20, help="optimizer steps")
    ap.add_argument("--seq", type=int, default=None, help="override model ctx")
    ap.add_argument("--lora", action="store_true",
                    help="apply the config peft block (frozen base + LoRA adapter)")
    args = ap.parse_args()

    import torch
    import yaml

    from src.model import build_model, maybe_wrap_peft

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    seq = int(args.seq or cfg["model"]["ctx"])
    batch, accum = int(t["batch"]), int(t["accum"])
    vocab = int(cfg["tokenizer"]["vocab_size"])

    if not torch.cuda.is_available():
        raise SystemExit("vram_probe needs CUDA")
    model = build_model(cfg, vocab_size=vocab)
    if args.lora:
        if not cfg.get("peft"):
            raise SystemExit("vram_probe --lora needs a peft block in the config")
        model = maybe_wrap_peft(model, cfg)
    model = model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = _make_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda")
    print(f"[probe] {cfg['name']}: {n_params / 1e6:.1f}M params seq={seq} "
          f"batch={batch} accum={accum}", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(0)

    def micro_step():
        ids = torch.randint(0, vocab, (batch, seq), device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, labels=ids)
        scaler.scale(out.loss / accum).backward()
        return float(out.loss)

    # Warmup: allocator + cuBLAS handles, excluded from the measured window.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            micro_step()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    micro, loss_sum = 0, 0.0
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            loss_sum += micro_step()
            micro += 1
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    result = {
        "config": cfg["name"],
        "params_m": round(n_params / 1e6, 2),
        "trainable_params_m": round(n_train / 1e6, 2),
        "seq": seq,
        "optimizer_steps": args.steps,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "tokens_per_s": round(micro * batch * seq / dt),
        "mean_loss": round(loss_sum / micro, 3),
    }
    print("[probe] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
