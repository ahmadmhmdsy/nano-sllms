"""Benchmark TrOCR with padded images."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def pad_to_square(img):
    """Pad image to square aspect ratio with white background."""
    w, h = img.size
    if w > h:
        new_h = w
        top = (new_h - h) // 2
        bottom = new_h - h - top
        padded = Image.new('RGB', (w, new_h), (255, 255, 255))
        padded.paste(img, (0, top))
    else:
        new_w = h
        left = (new_w - w) // 2
        right = new_w - w - left
        padded = Image.new('RGB', (new_w, h), (255, 255, 255))
        padded.paste(img, (left, 0))
    return padded

def cer(pred, target):
    """Character error rate."""
    import editdistance
    return editdistance.eval(pred, target) / max(len(target), 1)

def wer(pred, target):
    """Word error rate."""
    import editdistance
    pred_words = pred.split()
    target_words = target.split()
    return editdistance.eval(pred_words, target_words) / max(len(target_words), 1)

# Config
MODEL_NAME = "RayR1/trocr-base-arabic-handwritten"
VAL_DIR = "data/hflines/val"
NUM_SAMPLES = 100

# Load model
print(f"Loading {MODEL_NAME}...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model = model.cuda()
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load test data
with open(f"{VAL_DIR}/labels.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = test_data[:NUM_SAMPLES]
print(f"Testing on {len(test_data)} samples...")

# Run inference
predictions = []
targets = []
start = time.time()

for i, item in enumerate(test_data):
    image_path = Path(VAL_DIR) / item["image"]
    image = Image.open(image_path).convert("RGB")
    
    # Pad to square
    padded = pad_to_square(image)
    
    # Process and generate
    pixel_values = processor(padded, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        output = model.generate(pixel_values, max_length=256, num_beams=1)
    
    pred = processor.batch_decode(output, skip_special_tokens=True)[0]
    predictions.append(pred)
    targets.append(item["text"])
    
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{len(test_data)} - {elapsed:.1f}s - {(i+1)/elapsed:.1f} img/s")

elapsed = time.time() - start
print(f"\nInference: {len(test_data)} images in {elapsed:.1f}s ({len(test_data)/elapsed:.1f} img/s)")

# Compute metrics
try:
    import editdistance
except ImportError:
    print("editdistance not installed, using simple accuracy")
    def cer(pred, target):
        correct = sum(1 for p, t in zip(pred, target) if p == t)
        return 1 - correct / max(len(target), 1)
    def wer(pred, target):
        pred_words = pred.split()
        target_words = target.split()
        correct = sum(1 for p, t in zip(pred_words, target_words) if p == t)
        return 1 - correct / max(len(target_words), 1)

cers = [cer(p, t) for p, t in zip(predictions, targets)]
wers = [wer(p, t) for p, t in zip(predictions, targets)]

print(f"\n=== Free-Running Results (Original Model + Padding) ===")
print(f"CER: {sum(cers)/len(cers)*100:.1f}%")
print(f"WER: {sum(wers)/len(wers)*100:.1f}%")
print(f"Speed: {len(test_data)/elapsed:.1f} img/s")

print(f"\nSample predictions:")
for i in range(min(5, len(predictions))):
    print(f"  Target:  {targets[i][:70]}")
    print(f"  Predict: {predictions[i][:70]}")
    print()
