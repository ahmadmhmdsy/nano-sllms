"""Controlled test: can the pretrained model read CLEAN rendered Arabic text?

If it reads rendered text but not the dataset, the data/preprocessing is at fault.
If it fails on rendered text too, the model itself is unsuitable.
"""
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

OUT = Path("runs/trocr_finetune/rendered_test.txt")
FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\trado.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]


def render(text, font_path, size=48, pad=20):
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(tmp)
    box = d.textbbox((0, 0), text, font=font)
    w = box[2] - box[0] + 2 * pad
    h = box[3] - box[1] + 2 * pad
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, font=font, fill="black")
    return img


def pad_square(img):
    w, h = img.size
    if w > h:
        p = Image.new("RGB", (w, w), (255, 255, 255)); p.paste(img, (0, (w - h) // 2))
    else:
        p = Image.new("RGB", (h, h), (255, 255, 255)); p.paste(img, ((h - w) // 2, 0))
    return p


def main():
    # Find a working Arabic-capable font
    font_path = None
    for f in FONTS:
        if Path(f).exists():
            try:
                ImageFont.truetype(f, 40)
                font_path = f
                break
            except Exception:
                continue

    lines = ["CONTROLLED RENDERED-TEXT TEST", f"font={font_path}", ""]
    if font_path is None:
        lines.append("NO FONT FOUND")
        OUT.write_text("\n".join(lines), encoding="utf-8")
        print("no font")
        return

    processor = TrOCRProcessor.from_pretrained("RayR1/trocr-base-arabic-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "RayR1/trocr-base-arabic-handwritten").cuda().eval()

    texts = [
        "في الحج",
        "هل تعلم فائدة الكلمات",
        "انني سقيم",
        "مار جرجس بالقبيات",
    ]

    for txt in texts:
        for use_pad in (False, True):
            img = render(txt, font_path)
            src = pad_square(img) if use_pad else img
            pv = processor(src, return_tensors="pt").pixel_values.cuda()
            with torch.no_grad():
                out = model.generate(pv, max_length=64, num_beams=1)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0]
            lines.append(f"TRUE: {txt}")
            lines.append(f"  size={img.size} padding={use_pad} -> PRED: {pred}")
            lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
