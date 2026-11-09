"""Fine-tune TrOCR on Arabic OCR data - v2 (aspect-ratio padding fix).

Fixes over v1:
  1. Images are padded to square BEFORE the processor resizes to 384x384.
     v1 fed raw wide line images (e.g. 1959x128) which the processor squashed
     into a square, destroying the text. The pretrained model could not read
     them (it emitted "."); with square padding it emits real Arabic.
  2. eos_token_id is pinned to the tokenizer's SEP id (3) so generation stops.
  3. save_only_model=True -> no 4 GB optimizer.pt (v1 crashed on a full disk).
  4. A free-running (generate) eval runs at the end and reports CER/WER.
"""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from datasets import Dataset as HFDataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Config
MODEL_NAME = "RayR1/trocr-base-arabic-handwritten"
TRAIN_DIR = "data/hflines/train"
VAL_DIR = "data/hflines/val"
OUTPUT_DIR = "runs/trocr_finetune_v2"
MAX_TRAIN_SAMPLES = 500
MAX_VAL_SAMPLES = 100
MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 5e-5


def pad_to_square(img):
    """Pad to square with white background, preserving aspect ratio."""
    w, h = img.size
    if w > h:
        new_h = w
        top = (new_h - h) // 2
        p = Image.new("RGB", (w, new_h), (255, 255, 255))
        p.paste(img, (0, top))
    else:
        new_w = h
        left = (new_w - w) // 2
        p = Image.new("RGB", (new_w, h), (255, 255, 255))
        p.paste(img, (left, 0))
    return p


class ArabicOCRDataset(torch.utils.data.Dataset):
    def __init__(self, root, processor, max_samples=None, max_length=256):
        self.root = Path(root)
        self.processor = processor
        self.max_length = max_length
        with open(self.root / "labels.json", "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        if max_samples and len(self.labels) > max_samples:
            self.labels = self.labels[:max_samples]
        print(f"Loaded {len(self.labels)} samples from {root}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        image = Image.open(self.root / item["image"]).convert("RGB")
        image = pad_to_square(image)  # <-- the v1 fix
        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def free_running_eval(model, processor, val_dir, n=100):
    import editdistance
    with open(Path(val_dir) / "labels.json", encoding="utf-8") as f:
        data = json.load(f)[:n]
    preds, targets = [], []
    model.eval()
    start = time.time()
    for item in data:
        img = pad_to_square(Image.open(Path(val_dir) / item["image"]).convert("RGB"))
        pv = processor(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, max_length=MAX_LENGTH, num_beams=1)
        preds.append(processor.batch_decode(out, skip_special_tokens=True)[0])
        targets.append(item["text"])
    elapsed = time.time() - start
    cers = [editdistance.eval(p, t) / max(len(t), 1) for p, t in zip(preds, targets)]
    wers = [editdistance.eval(p.split(), t.split()) / max(len(t.split()), 1)
            for p, t in zip(preds, targets)]
    empty = sum(1 for p in preds if not p.strip())
    return {
        "cer": sum(cers) / len(cers) * 100,
        "wer": sum(wers) / len(wers) * 100,
        "empty_rate": empty / len(preds) * 100,
        "speed": len(preds) / elapsed,
        "samples": [(t, p) for t, p in zip(targets, preds)],
    }


def main():
    print(f"Loading model: {MODEL_NAME}", flush=True)
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Pin generation ids explicitly (v1 left eos unset -> generation misbehaved)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}, params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    train_dataset = ArabicOCRDataset(TRAIN_DIR, processor, MAX_TRAIN_SAMPLES, MAX_LENGTH)
    val_dataset = ArabicOCRDataset(VAL_DIR, processor, MAX_VAL_SAMPLES, MAX_LENGTH)

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
        save_total_limit=1,
        save_only_model=True,   # v1 fix: skip the 4 GB optimizer.pt
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        # eval during training is loss-only; the honest free-running eval runs once at the end
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        processing_class=processor,
    )

    print("Starting training...", flush=True)
    trainer.train()

    # Save with an explicit, correct generation config
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = MAX_LENGTH
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}", flush=True)

    print("Running free-running eval...", flush=True)
    res = free_running_eval(model, processor, VAL_DIR, MAX_VAL_SAMPLES)
    print(f"FREE-RUNNING CER={res['cer']:.1f}% WER={res['wer']:.1f}% "
          f"empty={res['empty_rate']:.0f}% speed={res['speed']:.1f} img/s", flush=True)
    with open(Path(OUTPUT_DIR) / "free_running_eval.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
