"""Free-running Stage 3 gate eval: CER/WER on real handwriting val lines.

This is the honest gate metric. Teacher-forced CER (what the training loop
logs when eval_generate is false) is a language-model score and is NOT used
here. The model generates autoregressively (model.generate) and the prediction
is compared to the transcript with no mirroring and no rescoring.
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
from src.training.eval import compute_cer, compute_wer, decode_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/hflines/val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
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
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn, pin_memory=True,
    )

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

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            tokens = model.generate(images, max_len=ds.max_len)
            preds.extend(decode_token_ids(tokens, idx_to_char))
            targets.extend(batch["raw_texts"])

    cer = compute_cer(preds, targets)
    wer = compute_wer(preds, targets)
    result = {
        "checkpoint": str(args.checkpoint),
        "data": args.data,
        "n": len(targets),
        "cer": cer,
        "wer": wer,
        "examples": [
            {"target": t, "pred": p} for t, p in list(zip(targets, preds))[:10]
        ],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "examples"},
                     ensure_ascii=False, indent=2))
    for ex in result["examples"]:
        print("  T: " + ex["target"])
        print("  P: " + ex["pred"])
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
