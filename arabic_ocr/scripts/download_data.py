"""Download additional Arabic OCR datasets from Hugging Face."""
import os
from pathlib import Path

# HF_TOKEN comes from the gitignored project-root .env - NEVER hard-code it here.
# hf_hub/datasets do NOT auto-load .env, so load it explicitly (same pattern as
# scripts/prepare_data.py). The previously hard-coded token must be ROTATED.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
except ImportError:
    pass

from datasets import load_dataset
import json
from pathlib import Path
from PIL import Image
import shutil
import requests
from io import BytesIO

# Create output directory
output_dir = Path('data/extra')
output_dir.mkdir(parents=True, exist_ok=True)


def save_image(img, img_path):
    """Save image from various formats."""
    if img is None:
        return False
    if isinstance(img, Image.Image):
        img.save(str(img_path))
        return True
    elif isinstance(img, dict):
        # Handle dict format {'path': ..., 'bytes': ...}
        if 'bytes' in img and img['bytes']:
            with open(img_path, 'wb') as f:
                f.write(img['bytes'])
            return True
        elif 'path' in img and img['path']:
            # Copy from path
            if os.path.exists(img['path']):
                shutil.copy(img['path'], img_path)
                return True
    elif isinstance(img, str):
        # Handle URL
        if img.startswith('http://') or img.startswith('https://'):
            try:
                response = requests.get(img, timeout=30)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception as e:
                print(f'    Error downloading {img}: {e}')
                return False
        # Handle local path
        elif os.path.exists(img):
            shutil.copy(img, img_path)
            return True
    return False


def download_dataset(dataset_name, text_field, output_prefix, max_samples=3000):
    """Download a dataset and save images + labels."""
    print(f'=== Downloading {dataset_name} ===')
    try:
        ds = load_dataset(dataset_name, split='train', streaming=True)
        labels = []
        count = 0
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            try:
                img = sample['image']
                text = sample[text_field]
                if img is None or text is None:
                    continue
                # Save image
                img_filename = f'{output_prefix}_{i:05d}.png'
                img_path = output_dir / img_filename
                if save_image(img, img_path):
                    labels.append({'image': img_filename, 'text': text})
                    count += 1
                    if count % 500 == 0:
                        print(f'  Saved {count} images...')
            except Exception as e:
                print(f'  Error on sample {i}: {e}')
                continue
        # Save labels
        label_file = output_dir / f'labels_{output_prefix}.json'
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f'  Total saved: {count} images')
        return count
    except Exception as e:
        print(f'  Error: {e}')
        return 0


if __name__ == '__main__':
    # Download NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset
    download_dataset(
        'NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset',
        'text',
        'qari_v03',
        max_samples=3000
    )
    
    # Download Misraj/Misraj-DocOCR
    download_dataset(
        'Misraj/Misraj-DocOCR',
        'markdown',
        'misraj',
        max_samples=3000
    )
    
    print('\n=== Done ===')
