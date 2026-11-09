"""Build Stage 3 data: extended vocab + materialized synthetic line crops.

Stage 3 recognizer pretrains on synthetic lines, then fine-tunes on the 25,921
real hflines. The synthetic pages store per-line text with HTML-ish markup
(<p>, <u>, <i>) that must be stripped; the line boxes are ground truth, so no
detector is needed to build the training crops.

Outputs:
  data/stage3_vocab.json                  - focused vocab + frequent data chars
  runs/stage3_synth_lines/{train,val}/     - line crops + labels.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "arabic_ocr"))

FOCUSED = REPO / "data" / "focused_vocab.json"
SYNTH = REPO / "runs" / "stage2_synth"
OUT = REPO / "runs" / "stage3_synth_lines"
VOCAB_OUT = REPO / "data" / "stage3_vocab.json"
PAD = 4
MIN_CHAR_COUNT = 10


def strip_markup(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    return text.strip()


def build_vocab():
    focused = json.loads(FOCUSED.read_text(encoding="utf-8"))["char_to_idx"]
    counts = Counter()
    for split in ("train", "val"):
        labels = json.loads(
            (REPO / "data" / "hflines" / split / "labels.json").read_text(encoding="utf-8")
        )
        for item in labels:
            counts.update(item["text"])
    char_to_idx = dict(focused)
    added = []
    for ch, n in counts.most_common():
        if ch in char_to_idx:
            continue
        if n < MIN_CHAR_COUNT:
            break
        char_to_idx[ch] = len(char_to_idx)
        added.append((ch, n))
    idx_to_char = {str(i): c for c, i in char_to_idx.items()}
    VOCAB_OUT.write_text(
        json.dumps({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return char_to_idx, added


def materialize(split):
    pages = [
        json.loads(l)
        for l in (SYNTH / split / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out_dir = OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for page in pages:
        img = Image.open(SYNTH / split / page["image"]).convert("RGB")
        W, H = img.size
        for li, line in enumerate(page["lines"]):
            text = strip_markup(line["text"])
            if not text:
                continue
            x0, y0, x1, y1 = line["box"]
            x0 = max(0, int(x0) - PAD)
            y0 = max(0, int(y0) - PAD)
            x1 = min(W, int(x1) + PAD)
            y1 = min(H, int(y1) + PAD)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = img.crop((x0, y0, x1, y1))
            stem = page["image"].split("/")[-1].replace(".png", "")
            name = f"{stem}_{li:02d}.png"
            crop.save(out_dir / name)
            labels.append({"image": name, "text": text})
    (out_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(labels)


def main():
    char_to_idx, added = build_vocab()
    print("stage3 vocab: %d classes (+%d added)" % (len(char_to_idx), len(added)))
    for ch, n in added:
        print("  + U+%04X x%d" % (ord(ch), n))
    for split in ("train", "val"):
        n = materialize(split)
        print("materialized %s: %d line crops -> %s" % (split, n, OUT / split))


if __name__ == "__main__":
    main()
