from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

p = TrOCRProcessor.from_pretrained("RayR1/trocr-base-arabic-handwritten")
m = VisionEncoderDecoderModel.from_pretrained("RayR1/trocr-base-arabic-handwritten").cuda().eval()
tok = p.tokenizer
print("cfg eos", m.config.eos_token_id, "start", m.config.decoder_start_token_id)
print("gen cfg eos", m.generation_config.eos_token_id, "start", m.generation_config.decoder_start_token_id)

f = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
img = Image.new("RGB", (400, 100), "white")
ImageDraw.Draw(img).text((20, 20), "في الحج", font=f, fill="black")
pv = p(img, return_tensors="pt").pixel_values.cuda()
with torch.no_grad():
    out = m.generate(pv, max_length=32, num_beams=1)
print("ids:", out[0].tolist())
print("decoded:", repr(p.batch_decode(out, skip_special_tokens=True)[0]))
print("decoded raw:", repr(p.batch_decode(out, skip_special_tokens=False)[0]))
# What does the tokenizer think the true text is?
print("true ids:", tok("في الحج").input_ids)
