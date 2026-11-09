"""Arabic OCR dataset."""
import json
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class ArabicOCRDataset(Dataset):
    """Dataset for Arabic OCR.

    Expects directory structure:
        root/
            labels.json  # [{"image": "xxx.png", "text": "..."}, ...]
            xxx.png

    Optional filtering (max_image_height, min_text_len, max_text_len) lets a
    mixed page/line corpus be narrowed to the line-level domain that the
    ISIPPT/KHATT benchmarks actually measure.
    """

    def __init__(self, root, image_height=48, image_width=256, augment=False,
                 vocab_path=None, char_to_idx=None, max_len=256,
                 min_text_len=0, max_text_len=None, max_image_height=None,
                 keep_aspect=True):
        self.root = Path(root)
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment
        self.max_len = max_len
        self.keep_aspect = keep_aspect

        labels_path = self.root / "labels.json"
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        self.labels = self._filter(labels, min_text_len, max_text_len, max_image_height)

        if char_to_idx is not None:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        elif vocab_path and Path(vocab_path).exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.char_to_idx = vocab["char_to_idx"]
            self.idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
        else:
            self.char_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
            self._build_vocab()

    def _filter(self, labels, min_text_len, max_text_len, max_image_height):
        """Drop entries outside the requested line-level envelope."""
        if not min_text_len and max_text_len is None and max_image_height is None:
            return labels

        kept = []
        for item in labels:
            text = item["text"]
            if len(text) < min_text_len:
                continue
            if max_text_len is not None and len(text) > max_text_len:
                continue
            if max_image_height is not None:
                try:
                    with Image.open(self.root / item["image"]) as im:
                        if im.size[1] > max_image_height:
                            continue
                except Exception:
                    continue
            kept.append(item)
        return kept

    def _build_vocab(self):
        """Build character vocabulary: ASCII + Arabic block + any data chars."""
        idx = len(self.char_to_idx)
        for code in range(32, 127):
            ch = chr(code)
            if ch not in self.char_to_idx:
                self.char_to_idx[ch] = idx
                self.idx_to_char[idx] = ch
                idx += 1
        for code in range(0x0600, 0x0700):
            ch = chr(code)
            if ch not in self.char_to_idx:
                self.char_to_idx[ch] = idx
                self.idx_to_char[idx] = ch
                idx += 1
        for item in self.labels:
            for char in item["text"]:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
                    idx += 1

    def __len__(self):
        return len(self.labels)

    def encode_text(self, text, max_len=None):
        """Encode text to a fixed-length index sequence.

        The sequence is [SOS] + chars + [EOS], truncated so that EOS is always
        retained, then right-padded with PAD up to max_len.
        """
        max_len = max_len or self.max_len
        sos = self.char_to_idx["<SOS>"]
        eos = self.char_to_idx["<EOS>"]
        unk = self.char_to_idx["<UNK>"]
        pad = self.char_to_idx["<PAD>"]

        chars = [self.char_to_idx.get(c, unk) for c in text][:max_len - 2]
        indices = [sos] + chars + [eos]
        indices.extend([pad] * (max_len - len(indices)))
        return indices

    def decode_indices(self, indices):
        """Decode indices to text."""
        chars = []
        for idx in indices:
            if idx == self.char_to_idx["<PAD>"]:
                break
            if idx in (self.char_to_idx["<SOS>"], self.char_to_idx["<EOS>"]):
                continue
            chars.append(self.idx_to_char.get(idx, "<UNK>"))
        return "".join(chars)

    def _resize(self, img):
        """Fit the image inside the target box preserving aspect ratio.

        Line images have extreme aspect ratios (KHATT ~15:1, ISIPPT ~27:1); a
        plain resize squashes the glyphs and destroys legibility.
        """
        w, h = img.size
        if self.keep_aspect:
            scale = min(self.image_width / w, self.image_height / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            canvas = Image.new("RGB", (self.image_width, self.image_height), (255, 255, 255))
            canvas.paste(img, (0, 0))
            return canvas
        return img.resize((self.image_width, self.image_height), Image.BILINEAR)

    def _augment(self, img):
        """Apply random augmentation to image."""
        import random
        import numpy as np

        if random.random() < 0.5:
            img = img.rotate(random.uniform(-3, 3), fillcolor=(255, 255, 255), resample=Image.BILINEAR)

        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            w, h = img.size
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            img = self._resize(img)

        if random.random() < 0.3:
            dx, dy = random.randint(-8, 8), random.randint(-4, 4)
            img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(255, 255, 255))

        if random.random() < 0.6:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.6:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.0))
        if random.random() < 0.25:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0)))
        if random.random() < 0.3:
            arr = np.array(img)
            noise = np.random.normal(0, 8, arr.shape)
            img = Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))

        return img

    def __getitem__(self, idx):
        item = self.labels[idx]

        img = Image.open(str(self.root / item["image"])).convert("RGB")
        if self.augment:
            img = self._augment(img)
        img = self._resize(img)

        import numpy as np
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

        text_indices = self.encode_text(item["text"])
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        return {
            "image": img_tensor,
            "text": text_tensor,
            "text_len": min(len(item["text"]) + 2, self.max_len),
            "raw_text": item["text"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "texts": torch.stack([item["text"] for item in batch]),
        "text_lens": torch.tensor([item["text_len"] for item in batch]),
        "raw_texts": [item["raw_text"] for item in batch],
    }
