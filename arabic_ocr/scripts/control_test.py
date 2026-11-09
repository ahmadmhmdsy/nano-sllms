"""CONTROL: validate the pipeline with the English TrOCR model.

If microsoft/trocr-base-handwritten reads rendered English correctly, the
pipeline is sound and the Arabic model is the broken component.
"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

OUT = Path("runs/trocr_finetune/control_test.txt")


def render(text, font_path, size=48, pad=20):
    font = ImageFont.truetype(font_path, size)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    b = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad), "white")
    ImageDraw.Draw(img).text((pad - b[0], pad - b[1]), text, font=font, fill="black")
    return img


def run(model_name, texts, font_path, lines, explicit_eos=None):
    lines.append(f"### {model_name}")
    try:
        proc = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda().eval()
    except Exception as e:
        lines.append(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:200]}")
        lines.append("")
        return
    lines.append(f"  config eos={model.config.eos_token_id} start={model.config.decoder_start_token_id}")
    lines.append(f"  gen    eos={model.generation_config.eos_token_id} start={model.generation_config.decoder_start_token_id}")
    kw = dict(max_length=64, num_beams=1)
    if explicit_eos is not None:
        kw["eos_token_id"] = explicit_eos
        kw["decoder_start_token_id"] = model.config.decoder_start_token_id
    for t in texts:
        img = render(t, font_path)
        pv = proc(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, **kw)
        pred = proc.batch_decode(out, skip_special_tokens=True)[0]
        lines.append(f"  TRUE: {t!r}")
        lines.append(f"  PRED: {pred!r}")
    lines.append("")


def main():
    lines = ["CONTROL TEST - pipeline validation", ""]
    en = r"C:\Windows\Fonts\arial.ttf"
    ar = r"C:\Windows\Fonts\arial.ttf"

    run("microsoft/trocr-base-handwritten",
        ["Hello World", "The quick brown fox", "Invoice 12345"], en, lines)
    run("microsoft/trocr-base-printed",
        ["Hello World", "The quick brown fox"], en, lines)
    run("RayR1/trocr-base-arabic-handwritten",
        ["في الحج", "انني سقيم"], ar, lines, explicit_eos=3)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
