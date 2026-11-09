"""Fix generation config and re-run benchmark."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Config
MODEL_PATH = "runs/trocr_finetune/checkpoint-189"
VAL_DIR = "data/hflines/val"
NUM_SAMPLES = 100

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

def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    
    # Fix generation config
    # The tokenizer has cls_token_id=2, pad_token_id=0, but no eos_token
    # For TrOCR, we need to set eos_token_id to stop generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.cls_token_id  # Use CLS as EOS
    model.config.bos_token_id = processor.tokenizer.cls_token_id
    
    # Also set generation config
    if hasattr(model, 'generation_config'):
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.cls_token_id
        model.generation_config.bos_token_id = processor.tokenizer.cls_token_id
    
    model = model.cuda()
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
    print(f"eos_token_id: {model.config.eos_token_id}")

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
        pixel_values = processor(image, return_tensors="pt").pixel_values.cuda()

        with torch.no_grad():
            output = model.generate(
                pixel_values,
                max_length=256,
                num_beams=1,
                early_stopping=True,
            )

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

    print(f"\n=== Free-Running Results ===")
    print(f"CER: {sum(cers)/len(cers)*100:.1f}%")
    print(f"WER: {sum(wers)/len(wers)*100:.1f}%")
    print(f"Speed: {len(test_data)/elapsed:.1f} img/s")

    print(f"\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"  Target:  {targets[i][:70]}")
        print(f"  Predict: {predictions[i][:70]}")
        print()

if __name__ == "__main__":
    main()
