"""Stage 3 diagnostic: is the decoder reading the image, or is it a character LM?

Three numbers on the same val subset:
  1. free-running CER    - model.generate on the real image
  2. teacher-forced CER  - argmax of the logits given the GT prefix
  3. zero-image CER      - model.generate on an all-zero image, plus the
     positional char agreement between the real-image and zero-image output.

If (3) is close to (1) the decoder ignores the encoder (a character LM).
The gap between (1) and (2) is exposure bias.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "arabic_ocr"))

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, decode_greedy, decode_token_ids, decode_ctc


def char_agreement(a, b):
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    same = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
    return same / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/hflines/val")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = config["data"]
    ds = ArabicOCRDataset(
        args.data,
        image_height=data_cfg["image_height"],
        image_width=data_cfg["image_width"],
        augment=False,
        vocab_path=data_cfg.get("vocab_path"),
        max_len=data_cfg.get("max_len", 256),
        min_text_len=data_cfg.get("min_text_len", 0),
        max_text_len=data_cfg.get("max_text_len"),
        keep_aspect=data_cfg.get("keep_aspect", True),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=data_cfg.get("num_workers", 4),
                        collate_fn=collate_fn, pin_memory=True)

    model_cfg = dict(config["model"])
    model_cfg["vocab_size"] = len(ds.char_to_idx)
    model_cfg["image_height"] = data_cfg["image_height"]
    model_cfg["image_width"] = data_cfg["image_width"]
    model_cfg["max_seq_len"] = ds.max_len
    model = create_model(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    idx_to_char = {i: c for c, i in ds.char_to_idx.items()}

    free_preds, tf_preds, zero_preds, targets = [], [], [], []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"].to(device)
            text_lens = batch["text_lens"].to(device)
            raw = batch["raw_texts"]

            out = model(images, text=texts, text_lens=text_lens)
            tf_preds.extend(decode_greedy(out["logits"], idx_to_char))

            tok = model.generate(images, max_len=ds.max_len)
            free_preds.extend(decode_token_ids(tok, idx_to_char))

            tok0 = model.generate(torch.zeros_like(images), max_len=ds.max_len)
            zero_preds.extend(decode_token_ids(tok0, idx_to_char))

            targets.extend(raw)
            n_seen += len(raw)
            if n_seen >= args.limit:
                break

    n = len(targets)
    lens = [len(p) for p in free_preds]
    tlens = [len(t) for t in targets]
    result = {
        "checkpoint": str(args.checkpoint),
        "data": args.data,
        "n": n,
        "cer_free": compute_cer(free_preds, targets),
        "cer_teacher_forced": compute_cer(tf_preds, targets),
        "cer_zero_image": compute_cer(zero_preds, targets),
        "char_agreement_free_vs_zero": sum(char_agreement(a, b) for a, b in zip(free_preds, zero_preds)) / n,
        "mean_pred_len": sum(lens) / n,
        "mean_target_len": sum(tlens) / n,
        "frac_pred_at_maxlen": sum(1 for l in lens if l >= ds.max_len - 1) / n,
    }
    if getattr(model, "ctc_head", None) is not None:
        lp = model.ctc_log_probs(model.encode(images))
        result["ctc_cer_last_batch"] = compute_cer(decode_ctc(lp, idx_to_char), raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for i in range(min(6, n)):
        print("  T: " + targets[i])
        print("  F: " + free_preds[i])
        print("  Z: " + zero_preds[i])
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
