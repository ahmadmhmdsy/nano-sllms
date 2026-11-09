"""Grow a trained CNN-Transformer-CTC checkpoint into a wider/deeper one (plan B).

Why this exists: E-103 measured the published 4.25M backbone to be UNDERFIT on its own
training labels (muharaf_train emission CER 12.02 vs muharaf_test 16.95, a ~5 pp gap), so
capacity - not label noise - is the binding constraint. Plan (B) therefore grows the
TRAINED model instead of re-initialising one, which is what option (D) did and what E-104
showed cannot work from a random init.

The surgery is warm-start-preserving:
  * a parameter whose shape already matches is copied verbatim;
  * when d_model or dim_feedforward grows, the overlapping block is copied into the
    top-left corner - except self_attn.in_proj_{weight,bias}, which is packed as three
    concatenated (d_model, d_model) Q/K/V matrices and is therefore split into thirds
    before copying (a naive block copy would write Q's rows into K's slot);
  * NEW encoder layers get their residual branches zero-initialised (out_proj and
    linear2), so at init each new layer is an exact identity and the grown model
    computes EXACTLY the same function as its source. --verify proves it.

DEPTH-ONLY growth is exactly function-preserving (verified: max|delta| = 0). WIDENING is
not, and cannot be made so with this architecture: LayerNorm normalises across all d_model
dimensions, so appending zero-padded dimensions changes every normalisation statistic even
when the new weights are zero. --verify reports the resulting max|delta| instead of
pretending the function survived.

Usage:
  python grow_ctc_model.py --src <ckpt> --out <ckpt> --arch-layers 8 --verify
  python grow_ctc_model.py --src <ckpt> --out <ckpt> --arch-d-model 384 --arch-layers 8 --arch-ffn 1536
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
CKPT_PATH = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"

sys.path.insert(0, str(MODEL_DIR))


def build(cfg, charset, device="cpu"):
    from submission_code.torch_models import CNNTransformerCTC

    return CNNTransformerCTC(
        num_classes=charset.vocab_size,
        d_model=cfg["d_model"],
        num_encoder_layers=cfg["transformer_layers"],
        num_heads=cfg["transformer_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg.get("dropout", 0.1),
        height_pooling=cfg.get("height_pooling", "attention"),
    ).to(device)


def _block_copy(dst, src, name, d_src, d_dst, ffn_src, ffn_dst):
    """Copy src's overlapping block into dst. False = a shape this tool cannot grow."""
    if name.endswith("self_attn.in_proj_weight"):
        for i in range(3):                      # Q | K | V packed on dim 0
            # the destination third is d_dst wide but only d_src of it overlaps
            dst[i * d_dst:i * d_dst + d_src, :d_src] = src[i * d_src:(i + 1) * d_src, :d_src]
        return True
    if name.endswith("self_attn.in_proj_bias"):
        for i in range(3):
            dst[i * d_dst:i * d_dst + d_src] = src[i * d_src:(i + 1) * d_src]
        return True
    if name.endswith("self_attn.out_proj.weight"):
        dst[:d_src, :d_src] = src[:d_src, :d_src]
        return True
    if name.endswith("self_attn.out_proj.bias"):
        dst[:d_src] = src[:d_src]
        return True
    if name.endswith("linear1.weight"):
        dst[:ffn_src, :d_src] = src[:ffn_src, :d_src]
        return True
    if name.endswith("linear1.bias"):
        dst[:ffn_src] = src[:ffn_src]
        return True
    if name.endswith("linear2.weight"):
        dst[:d_src, :ffn_src] = src[:d_src, :ffn_src]
        return True
    if name.endswith("linear2.bias"):
        dst[:d_src] = src[:d_src]
        return True
    if name.endswith(("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                      "norm.weight", "norm.bias")):
        dst[:d_src] = src[:d_src]
        return True
    if name.endswith("classifier.weight"):
        dst[:, :d_src] = src[:, :d_src]
        return True
    if name.endswith("projection.weight"):
        dst[:d_src, :] = src[:d_src, :]
        return True
    if name.endswith("projection.bias"):
        dst[:d_src] = src[:d_src]
        return True
    return False


def transfer(dst_sd, src_sd, d_src, d_dst, ffn_src, ffn_dst):
    exact, block, fresh = [], [], []
    for name, dst in dst_sd.items():
        src = src_sd.get(name)
        if src is None:                      # a brand-new encoder layer
            fresh.append(name)
            continue
        if tuple(src.shape) == tuple(dst.shape):
            dst.copy_(src)
            exact.append(name)
            continue
        if _block_copy(dst, src, name, d_src, d_dst, ffn_src, ffn_dst):
            block.append(name)
        else:
            fresh.append(name)
    return exact, block, fresh


def zero_residual(sd, layer_indices):
    """Make each listed encoder layer an exact identity: x + 0, x + 0."""
    for i in layer_indices:
        p = "encoder.layers.%d." % i
        for suffix in ("self_attn.out_proj.weight", "self_attn.out_proj.bias",
                       "linear2.weight", "linear2.bias"):
            sd[p + suffix].zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(CKPT_PATH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch-d-model", type=int, default=0, help="0 = keep source")
    ap.add_argument("--arch-layers", type=int, default=0)
    ap.add_argument("--arch-heads", type=int, default=0)
    ap.add_argument("--arch-ffn", type=int, default=0)
    ap.add_argument("--no-zero-init-new-layers", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="forward both models on a synthetic batch and report max|delta|")
    args = ap.parse_args()

    from submission_code.ocr_helpers import Charset

    blob = torch.load(args.src, map_location="cpu", weights_only=False)
    src_cfg = dict(blob.get("model_config", {}))
    charset = Charset(blob["charset"])
    src_sd = blob["model_state"]

    d_src, ffn_src = src_cfg["d_model"], src_cfg["dim_feedforward"]
    l_src, h_src = src_cfg["transformer_layers"], src_cfg["transformer_heads"]
    d_dst = args.arch_d_model or d_src
    ffn_dst = args.arch_ffn or ffn_src
    l_dst = args.arch_layers or l_src
    h_dst = args.arch_heads or h_src

    if d_dst % h_dst:
        raise SystemExit("d_model %d is not divisible by heads %d" % (d_dst, h_dst))
    if l_dst < l_src or d_dst < d_src or ffn_dst < ffn_src:
        raise SystemExit("this tool only GROWS (src %d/%d/%d -> dst %d/%d/%d)"
                         % (d_src, l_src, ffn_src, d_dst, l_dst, ffn_dst))

    cfg = dict(src_cfg)
    cfg.update(d_model=d_dst, transformer_layers=l_dst, transformer_heads=h_dst,
               dim_feedforward=ffn_dst)
    model = build(cfg, charset)
    dst_sd = model.state_dict()
    n_dst = sum(p.numel() for p in model.parameters())

    exact, block, fresh = transfer(dst_sd, src_sd, d_src, d_dst, ffn_src, ffn_dst)
    new_layers = list(range(l_src, l_dst))
    if new_layers and not args.no_zero_init_new_layers:
        zero_residual(dst_sd, new_layers)
    model.load_state_dict(dst_sd)

    n_src = sum(v.numel() for k, v in src_sd.items()
                if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    print("src %s: d=%d L=%d H=%d F=%d  params=%d" % (Path(args.src).name, d_src, l_src,
                                                      h_src, ffn_src, n_src))
    print("dst:      d=%d L=%d H=%d F=%d  params=%d" % (d_dst, l_dst, h_dst, ffn_dst,
                                                        n_dst))
    print("transfer: exact=%d block=%d fresh=%d   new encoder layers=%s (zero-init=%s)"
          % (len(exact), len(block), len(fresh), new_layers,
             not args.no_zero_init_new_layers and bool(new_layers)))
    if block:
        print("  block-copied: " + ", ".join(sorted(block)[:8]) +
              (" ..." if len(block) > 8 else ""))
    if fresh:
        print("  fresh:        " + ", ".join(sorted(fresh)[:8]) +
              (" ..." if len(fresh) > 8 else ""))

    if args.verify:
        src_model = build(src_cfg, charset)
        src_model.load_state_dict(src_sd)
        src_model.eval()
        model.eval()
        torch.manual_seed(0)
        h = src_cfg.get("image_height", 96)
        x = torch.rand(2, 1, h, 320) * 0.5 + 0.25
        w = torch.tensor([320, 240])
        with torch.no_grad():
            a = src_model(x, input_widths=w)
            b = model(x, input_widths=w)
        same = tuple(a.shape) == tuple(b.shape)
        print("verify: src%s dst%s  max|delta| %s"
              % (tuple(a.shape), tuple(b.shape),
                 ("%.3e" % float((a - b).abs().max())) if same else "SHAPE MISMATCH"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "charset": blob["charset"],
                "model_config": cfg, "step": 0, "epoch": 0, "stage": "grow",
                "target_height": src_cfg.get("image_height", 96),
                "grown_from": str(args.src),
                "growth_report": {
                    "d_model": [d_src, d_dst], "layers": [l_src, l_dst],
                    "heads": [h_src, h_dst], "dim_feedforward": [ffn_src, ffn_dst],
                    "params": [n_src, n_dst], "exact": len(exact), "block": len(block),
                    "fresh": len(fresh), "zero_init_layers": new_layers}},
               out)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
