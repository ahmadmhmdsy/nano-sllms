"""Focused free-running test: does padding fix TrOCR on Arabic lines?"""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import editdistance

VAL_DIR = "data/hflines/val"
N = 20


def pad_to_square(img):
    w, h = img.size
    if w > h:
        p = Image.new("RGB", (w, w), (255, 255, 255))
        p.paste(img, (0, (w - h) // 2))
    else:
        p = Image.new("RGB", (h, h), (255, 255, 255))
        p.paste(img, ((h - w) // 2, 0))
    return p


def evaluate(model, processor, data, use_padding):
    preds, targets = [], []
    t0 = time.time()
    for item in data:
        img = Image.open(Path(VAL_DIR) / item["image"]).convert("RGB")
        if use_padding:
            img = pad_to_square(img)
        pv = processor(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, max_length=128, num_beams=1)
        preds.append(processor.batch_decode(out, skip_special_tokens=True)[0])
        targets.append(item["text"])
    el = time.time() - t0
    cers = [editdistance.eval(p, t) / max(len(t), 1) for p, t in zip(preds, targets)]
    wers = [editdistance.eval(p.split(), t.split()) / max(len(t.split()), 1)
            for p, t in zip(preds, targets)]
    return {
        "cer": sum(cers) / len(cers) * 100,
        "wer": sum(wers) / len(wers) * 100,
        "empty": sum(1 for p in preds if not p.strip()),
        "speed": len(preds) / el,
        "pairs": list(zip(targets, preds)),
    }


def main():
    with open(Path(VAL_DIR) / "labels.json", encoding="utf-8") as f:
        data = json.load(f)[:N]

    processor = TrOCRProcessor.from_pretrained("RayR1/trocr-base-arabic-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "RayR1/trocr-base-arabic-handwritten").cuda().eval()

    out = {}
    for tag, pad in [("padded", True), ("raw", False)]:
        r = evaluate(model, processor, data, pad)
        out[tag] = r
        print(f"{tag}: CER={r['cer']:.1f}% WER={r['wer']:.1f}% "
              f"empty={r['empty']}/{N} {r['speed']:.1f} img/s", flush=True)

    with open("runs/trocr_finetune/focused_padding_test.txt", "w", encoding="utf-8") as f:
        for tag in ("padded", "raw"):
            f.write(f"=== pretrained / {tag} ===\n")
            f.write(f"CER={out[tag]['cer']:.1f}% WER={out[tag]['wer']:.1f}% "
                    f"empty={out[tag]['empty']}/{N}\n")
            for t, p in out[tag]["pairs"]:
                f.write(f"  T: {t}\n  P: {p}\n")
            f.write("\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
