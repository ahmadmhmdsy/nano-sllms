from PIL import Image, ImageDraw, ImageFont
import torch, traceback
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

OUT = __import__("pathlib").Path("runs/trocr_finetune/control_test3.txt")
lines = []

def render(text, size=48, pad=20):
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    b = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (b[2]-b[0]+2*pad, b[3]-b[1]+2*pad), "white")
    ImageDraw.Draw(img).text((pad-b[0], pad-b[1]), text, font=font, fill="black")
    return img

strategies = [
    ("use_fast=True", dict(use_fast=True)),
    ("use_fast=False", dict(use_fast=False)),
]
for name in ["microsoft/trocr-base-handwritten"]:
    for label, kw in strategies:
        lines.append(f"### {name} [{label}]")
        try:
            proc = TrOCRProcessor.from_pretrained(name, **kw)
            model = VisionEncoderDecoderModel.from_pretrained(name).cuda().eval()
            lines.append("  loaded OK")
            for t in ["Hello World", "The quick brown fox"]:
                img = render(t)
                pv = proc(img, return_tensors="pt").pixel_values.cuda()
                with torch.no_grad():
                    out = model.generate(pv, max_length=64, num_beams=1)
                lines.append(f"  TRUE {t!r} -> PRED {proc.batch_decode(out, skip_special_tokens=True)[0]!r}")
        except Exception as e:
            lines.append(f"  FAILED {type(e).__name__}: {str(e)[:160]}")
        lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("WROTE", OUT)
