"""Fine-tune TrOCR on Arabic OCR data."""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import Dataset as HFDataset

# Config
MODEL_NAME = "RayR1/trocr-base-arabic-handwritten"
TRAIN_DIR = "data/hflines/train"
VAL_DIR = "data/hflines/val"
OUTPUT_DIR = "runs/trocr_finetune"
MAX_TRAIN_SAMPLES = 500  # Small data smoke test
MAX_VAL_SAMPLES = 100
MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 5e-5


class ArabicOCRDataset(Dataset):
    """Dataset for Arabic OCR using TrOCR processor."""

    def __init__(self, root, processor, max_samples=None, max_length=256):
        self.root = Path(root)
        self.processor = processor
        self.max_length = max_length

        # Load labels (labels.json is UTF-8 -- verified at byte level 2026-10-03)
        labels_path = self.root / "labels.json"
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        if max_samples and len(self.labels) > max_samples:
            self.labels = self.labels[:max_samples]

        print(f"Loaded {len(self.labels)} samples from {root}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        image_path = self.root / item["image"]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]

        # Tokenize text
        text = item["text"]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Replace padding token id with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def main():
    print(f"Loading model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set model config for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    train_dataset = ArabicOCRDataset(TRAIN_DIR, processor, MAX_TRAIN_SAMPLES, MAX_LENGTH)
    val_dataset = ArabicOCRDataset(VAL_DIR, processor, MAX_VAL_SAMPLES, MAX_LENGTH)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        processing_class=processor,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    # Quick eval
    print("\nRunning evaluation...")
    results = trainer.evaluate()
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
