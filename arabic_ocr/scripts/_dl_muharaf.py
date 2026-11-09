"""Stage 0.5c (b) - fetch the second in-domain OCR candidate (TASKS 134).

sdkv2/muharaf-arabic-ocr: character-level CNN-Transformer-CTC trained on the
Muharaf corpus (card: test CER 0.1331 with the full rerank pipeline, ~0.17
backbone-only greedy). Downloaded locally; nothing is pip-installed.
Weights live under arabic_ocr/models/ which is gitignored.
"""
import os
from huggingface_hub import snapshot_download

DEST = os.path.join("arabic_ocr", "models", "muharaf_ocr")
os.makedirs(DEST, exist_ok=True)
p = snapshot_download(
    repo_id="sdkv2/muharaf-arabic-ocr",
    local_dir=DEST,
    # notebooks + demo images are not needed for a line-level eval
    ignore_patterns=["notebooks/*", "data/*", "*.ipynb", "*.png", "*.jpg", ".gitattributes"],
)
print("OK", p)
total = 0
for root, dirs, files in os.walk(DEST):
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        total += sz
        print("%12d  %s" % (sz, os.path.relpath(fp, DEST)))
print("TOTAL_BYTES", total)