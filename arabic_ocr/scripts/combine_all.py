"""Combine all available data into a single dataset."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import shutil

# Create output directory
output_dir = Path('data/all_combined')
output_dir.mkdir(parents=True, exist_ok=True)

# Load all label files
all_labels = []

# 1. Combined dataset (real + large)
print('Loading data/combined/train/labels.json...')
combined_labels = json.load(open('data/combined/train/labels.json', 'r', encoding='utf-8'))
print(f'  Found {len(combined_labels)} images')
for item in combined_labels:
    # Copy image to new location
    src = Path('data/combined/train') / item['image']
    dst = output_dir / item['image']
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    all_labels.append(item)

# 2. Misraj dataset
print('Loading data/extra/labels_misraj.json...')
misraj_labels = json.load(open('data/extra/labels_misraj.json', 'r', encoding='utf-8'))
print(f'  Found {len(misraj_labels)} images')
for item in misraj_labels:
    # Copy image to new location
    src = Path('data/extra') / item['image']
    dst = output_dir / item['image']
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    all_labels.append(item)

# Save combined labels
print(f'Total images: {len(all_labels)}')
with open(output_dir / 'labels.json', 'w', encoding='utf-8') as f:
    json.dump(all_labels, f, ensure_ascii=False, indent=2)

print(f'Saved to {output_dir}')
