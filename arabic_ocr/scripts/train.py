"""Training script for Arabic OCR."""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.trainer import OCRTrainer


def build_dataset(root, data_cfg, augment=False):
    """Build a dataset honouring every filtering/length key in the config."""
    return ArabicOCRDataset(
        root,
        image_height=data_cfg["image_height"],
        image_width=data_cfg["image_width"],
        augment=augment,
        vocab_path=data_cfg.get("vocab_path"),
        max_len=data_cfg.get("max_len", 256),
        min_text_len=data_cfg.get("min_text_len", 0),
        max_text_len=data_cfg.get("max_text_len"),
        max_image_height=data_cfg.get("max_image_height"),
        keep_aspect=data_cfg.get("keep_aspect", True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="arabic_ocr/configs/smoke.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    print("Config: " + str(config["name"]))
    print("Model type: " + str(config["model"]["type"]))

    train_dataset = build_dataset(data_cfg["train_dir"], data_cfg, augment=data_cfg.get("augment", False))
    val_dataset = build_dataset(data_cfg["val_dir"], data_cfg, augment=False)

    print("Train samples: %d" % len(train_dataset))
    print("Val samples: %d" % len(val_dataset))
    print("Vocab size: %d" % len(train_dataset.char_to_idx))
    print("Max seq len: %d" % train_dataset.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Vocab and sequence length must follow the data, never the config default.
    model_config = dict(config["model"])
    model_config["vocab_size"] = len(train_dataset.char_to_idx)
    model_config["image_height"] = data_cfg["image_height"]
    model_config["image_width"] = data_cfg["image_width"]
    model_config["max_seq_len"] = train_dataset.max_len
    model = create_model(model_config)

    pretrained_path = config.get("train", {}).get("pretrained")
    if pretrained_path and Path(pretrained_path).exists():
        print("Loading pre-trained weights from " + pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("Model params: %.1fM" % (model.count_params() / 1e6))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = dict(config["train"])
    train_cfg.setdefault("model_type", config["model"]["type"])
    trainer = OCRTrainer(model, train_cfg, device=device, char_to_idx=train_dataset.char_to_idx)

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
