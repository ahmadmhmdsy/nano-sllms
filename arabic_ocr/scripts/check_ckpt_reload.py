"""Reload a run checkpoint on CPU and verify it rebuilds through load_ctc_backbone.

Why this exists: an 8-hour from-scratch run is worthless if its checkpoint cannot be
reloaded, and load_ctc_backbone rebuilds the architecture purely from the stored
model_config - so a wrong model_config surfaces only at evaluation time, hours later.
This is the pre-flight gate for that. It is CPU-only, so it never touches the GPU that
a training run is using.

Usage:
  python arabic_ocr/scripts/check_ckpt_reload.py --ckpt <run>/ckpt_last.pt --expect-params 15336743
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr" / "models" / "muharaf_ocr"))

import torch  # noqa: E402
from submission_code.torch_models import load_ctc_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-params", type=int, default=0)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    problems = []

    if "model_config" not in blob:
        problems.append("checkpoint has no model_config: load_ctc_backbone cannot rebuild it")
    if "charset" not in blob:
        problems.append("checkpoint has no charset")

    model, charset, cfg = load_ctc_backbone(args.ckpt, torch.device("cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    if args.expect_params and n_params != args.expect_params:
        problems.append("param mismatch: got %d want %d" % (n_params, args.expect_params))

    h = int(blob.get("target_height", 96))
    w = args.width
    logits = None
    try:
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, h, w), input_widths=torch.tensor([w]))
    except Exception as exc:  # noqa: BLE001 - report, never mask
        problems.append("forward pass failed at height %d: %r" % (h, exc))

    if logits is not None:
        if int(logits.shape[-1]) != charset.vocab_size:
            problems.append("logits classes %d != charset.vocab_size %d"
                            % (int(logits.shape[-1]), charset.vocab_size))
        want_t = int(model.output_lengths(torch.tensor([w]))[0])
        if int(logits.shape[0]) != want_t:
            problems.append("logits T %d != output_lengths %d" % (int(logits.shape[0]), want_t))

    out = {
        "ckpt": str(args.ckpt),
        "step": blob.get("step"),
        "stage": blob.get("stage"),
        "target_height": h,
        "params": n_params,
        "model_config": {k: cfg.get(k) for k in ("d_model", "transformer_layers",
                                                 "transformer_heads", "dim_feedforward",
                                                 "height_pooling")},
        "charset_vocab": charset.vocab_size,
        "forward_logits_shape": list(logits.shape) if logits is not None else None,
        "has_optimizer_state": "optimizer_state" in blob,
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
