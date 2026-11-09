"""Combine real and large datasets."""
import json
import shutil
from pathlib import Path

# Create combined directories
combined_train = Path('data/combined/train')
combined_val = Path('data/combined/val')

# Clear existing files
if combined_train.exists():
    shutil.rmtree(combined_train)
if combined_val.exists():
    shutil.rmtree(combined_val)

combined_train.mkdir(parents=True, exist_ok=True)
combined_val.mkdir(parents=True, exist_ok=True)

# Copy real data
print('Copying real data...')
real_train = Path('data/real/train')
real_val = Path('data/real/val')

# Copy train
for f in real_train.glob('*.png'):
    shutil.copy(f, combined_train / f.name)
with open(real_train / 'labels.json', 'r', encoding='utf-8') as f:
    real_train_labels = json.load(f)

# Copy val
for f in real_val.glob('*.png'):
    shutil.copy(f, combined_val / f.name)
with open(real_val / 'labels.json', 'r', encoding='utf-8') as f:
    real_val_labels = json.load(f)

# Copy large data
print('Copying large data...')
large_train = Path('data/large/train')
large_val = Path('data/large/val')

# Copy train with offset
offset = len(real_train_labels)
large_train_labels_new = []
for f in large_train.glob('*.png'):
    new_name = f'{offset:05d}.png'
    shutil.copy(f, combined_train / new_name)
    large_train_labels_new.append({'image': new_name, 'text': ''})  # placeholder
    offset += 1

# Load large train labels and update
with open(large_train / 'labels.json', 'r', encoding='utf-8') as f:
    large_train_labels = json.load(f)

# Update labels with new filenames
for i, label in enumerate(large_train_labels):
    new_idx = len(real_train_labels) + i
    label['image'] = f'{new_idx:05d}.png'

# Copy val with offset
offset = len(real_val_labels)
large_val_labels_new = []
for f in large_val.glob('*.png'):
    new_name = f'{offset:05d}.png'
    shutil.copy(f, combined_val / new_name)
    large_val_labels_new.append({'image': new_name, 'text': ''})  # placeholder
    offset += 1

# Load large val labels and update
with open(large_val / 'labels.json', 'r', encoding='utf-8') as f:
    large_val_labels = json.load(f)

# Update labels with new filenames
for i, label in enumerate(large_val_labels):
    new_idx = len(real_val_labels) + i
    label['image'] = f'{new_idx:05d}.png'

# Combine labels
combined_train_labels = real_train_labels + large_train_labels
combined_val_labels = real_val_labels + large_val_labels

# Save combined labels
with open(combined_train / 'labels.json', 'w', encoding='utf-8') as f:
    json.dump(combined_train_labels, f, ensure_ascii=False, indent=2)
with open(combined_val / 'labels.json', 'w', encoding='utf-8') as f:
    json.dump(combined_val_labels, f, ensure_ascii=False, indent=2)

print(f'Combined train: {len(combined_train_labels)} images')
print(f'Combined val: {len(combined_val_labels)} images')
