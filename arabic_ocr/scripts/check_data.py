"""Check downloaded data."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image
from pathlib import Path

# Check misraj data
labels = json.load(open('data/extra/labels_misraj.json', 'r', encoding='utf-8'))
print(f'Total images: {len(labels)}')
print('First 3 labels:')
for i, item in enumerate(labels[:3]):
    text_preview = item['text'][:50].replace('\n', ' ')
    print(f'  {i}: image={item["image"]}, text={text_preview}...')

# Check first image
img_path = Path('data/extra') / labels[0]['image']
print(f'First image exists: {img_path.exists()}')
if img_path.exists():
    img = Image.open(img_path)
    print(f'Image size: {img.size}')
    print(f'Image mode: {img.mode}')

# Count total images in data/extra
img_files = list(Path('data/extra').glob('*.png'))
print(f'Total PNG files in data/extra: {len(img_files)}')
