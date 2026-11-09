"""Test TrOCR with padded images."""
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

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

# Load model
print("Loading model...")
processor = TrOCRProcessor.from_pretrained("RayR1/trocr-base-arabic-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("RayR1/trocr-base-arabic-handwritten")
model = model.cuda()
model.eval()

# Test with padded images
for i in range(5):
    img = Image.open(f"data/hflines/val/{i:06d}.png").convert("RGB")
    print(f"\nImage {i:06d}.png: {img.size}")
    
    # Pad to square
    padded = pad_to_square(img)
    print(f"Padded: {padded.size}")
    
    # Process and generate
    pixel_values = processor(padded, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        output = model.generate(pixel_values, max_length=256, num_beams=1)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"Prediction: {decoded[:70]}")
