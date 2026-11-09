"""Build unified vocabulary from all datasets and retrain."""
import json
import shutil
from pathlib import Path
from collections import Counter

# Load all datasets
datasets = {
    'train': 'data/all_data/train/labels.json',
    'val': 'data/all_data/val/labels.json',
    'isippt': 'data/benchmark/isippt/labels.json',
    'khatt': 'data/benchmark/khatt/labels.json',
}

all_chars = Counter()
for name, path in datasets.items():
    labels = json.loads(Path(path).read_text(encoding='utf-8'))
    for item in labels:
        all_chars.update(item['text'])

# Build unified vocabulary
char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Add all characters
for char, count in sorted(all_chars.items()):
    if char not in char_to_idx:
        idx = len(char_to_idx)
        char_to_idx[char] = idx
        idx_to_char[idx] = char

print(f"Unified vocabulary size: {len(char_to_idx)}")

# Save vocabulary
vocab = {
    'char_to_idx': char_to_idx,
    'idx_to_char': {str(k): v for k, v in idx_to_char.items()},
}
Path('data/unified_vocab.json').write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding='utf-8')
print("Saved unified vocabulary to data/unified_vocab.json")

# Print character statistics
print(f"\nCharacter frequency (top 20):")
for char, count in all_chars.most_common(20):
    print(f"  {repr(char)}: {count}")
