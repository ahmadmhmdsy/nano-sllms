"""Clear corrupted TrOCR tokenizer cache, re-download, and validate pipeline."""
import shutil
from pathlib import Path

from huggingface_hub import constants

OUT = Path("runs/trocr_finetune/control_test2.txt")
lines = []

cache = Path(constants.HF_HUB_CACHE)
lines.append(f"HF cache: {cache}")

for name in ["models--microsoft--trocr-base-handwritten",
             "models--microsoft--trocr-base-printed"]:
    d = cache / name
    if not d.exists():
        lines.append(f"{name}: not cached")
        continue
    zeros = [p for p in d.rglob("*") if p.is_file() and p.stat().st_size == 0]
    lines.append(f"{name}: {len(zeros)} zero-byte files")
    for z in zeros[:6]:
        lines.append(f"    {z.relative_to(d)}")
    if zeros:
        shutil.rmtree(d, ignore_errors=True)
        lines.append(f"  -> removed {d} (will re-download)")

# Now try loading + running
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def render(text, size=48, pad=20):
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    b = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad), "white")
    ImageDraw.Draw(img).text((pad - b[0], pad - b[1]), text, font=font, fill="black")
    return img


for name in ["microsoft/trocr-base-handwritten", "microsoft/trocr-base-printed"]:
    lines.append("")
    lines.append(f"### {name}")
    try:
        proc = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name).cuda().eval()
    except Exception as e:
        lines.append(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:300]}")
        continue
    lines.append(f"  loaded OK; vocab={len(proc.tokenizer)}")
    for t in ["Hello World", "The quick brown fox", "Invoice 12345"]:
        img = render(t)
        pv = proc(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, max_length=64, num_beams=1)
        lines.append(f"  TRUE: {t!r}")
        lines.append(f"  PRED: {proc.batch_decode(out, skip_special_tokens=True)[0]!r}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("WROTE", OUT)
