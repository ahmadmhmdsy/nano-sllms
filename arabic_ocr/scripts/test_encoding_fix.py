"""Test the encoding fix: read labels as UTF-8 (correct) vs CP1256 (mojibake).

Writes results to a UTF-8 file so the Windows console codepage cannot mangle them.
"""
import json
import time
from pathlib import Path

import editdistance
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

VAL_DIR = Path("data/hflines/val")
N = 20
OUT = Path("runs/trocr_finetune/encoding_fix_test.txt")


def pad_to_square(img):
    w, h = img.size
    if w > h:
        p = Image.new("RGB", (w, w), (255, 255, 255))
        p.paste(img, (0, (w - h) // 2))
    else:
        p = Image.new("RGB", (h, h), (255, 255, 255))
        p.paste(img, ((h - w) // 2, 0))
    return p


def evaluate(model, processor, data, targets, use_padding, max_len=128):
    preds = []
    t0 = time.time()
    for item in data:
        img = Image.open(VAL_DIR / item["image"]).convert("RGB")
        if use_padding:
            img = pad_to_square(img)
        pv = processor(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, max_length=max_len, num_beams=1)
        preds.append(processor.batch_decode(out, skip_special_tokens=True)[0])
    el = time.time() - t0
    cers = [editdistance.eval(p, t) / max(len(t), 1) for p, t in zip(preds, targets)]
    wers = [editdistance.eval(p.split(), t.split()) / max(len(t.split()), 1)
            for p, t in zip(preds, targets)]
    return {
        "cer": sum(cers) / len(cers) * 100,
        "wer": sum(wers) / len(wers) * 100,
        "speed": len(preds) / el,
        "preds": preds,
    }


def main():
    # The truth: labels.json is UTF-8 (bytes d9 81 = U+0641 'ف')
    raw = (VAL_DIR / "labels.json").read_bytes()
    data_utf8 = json.loads(raw.decode("utf-8"))[:N]
    data_cp = json.loads(raw.decode("cp1256"))[:N]
    t_utf8 = [d["text"] for d in data_utf8]
    t_cp = [d["text"] for d in data_cp]

    processor = TrOCRProcessor.from_pretrained("RayR1/trocr-base-arabic-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "RayR1/trocr-base-arabic-handwritten").cuda().eval()

    lines = []
    lines.append("ENCODING FIX TEST - pretrained RayR1/trocr-base-arabic-handwritten")
    lines.append(f"bytes of first text: {raw[:0]}")
    lines.append("")
    lines.append("CORRECT target (utf-8): " + t_utf8[0])
    lines.append("MOJIBAKE target (cp1256): " + t_cp[0])
    lines.append("")

    for pad in (True, False):
        preds = evaluate(model, processor, data_utf8, t_utf8, pad)["preds"]
        for enc_name, targets in (("utf-8", t_utf8), ("cp1256", t_cp)):
            cers = [editdistance.eval(p, t) / max(len(t), 1) for p, t in zip(preds, targets)]
            wers = [editdistance.eval(p.split(), t.split()) / max(len(t.split()), 1)
                    for p, t in zip(preds, targets)]
            lines.append(
                f"padding={pad!s:5s} targets={enc_name:6s} "
                f"CER={sum(cers)/len(cers)*100:6.1f}% WER={sum(wers)/len(wers)*100:6.1f}%"
            )
        lines.append("")
        lines.append(f"--- sample predictions (padding={pad}) ---")
        for i in range(6):
            lines.append(f"  IMG {data_utf8[i]['image']}")
            lines.append(f"  TRUE: {t_utf8[i]}")
            lines.append(f"  PRED: {preds[i]}")
            lines.append("")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
