
"""Generate mishkala predictions on WikiNews 2024 bare input."""
import sys, os, re, json, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "models" / "e19" / "mishkala"))

# Strip diacritics
BAND = "[ًٌٍَُِّْٰٴۖ-ۭ]"
strip_re = re.compile(BAND)

def strip_marks(t):
    return strip_re.sub("", t)

# Load mishkala model
import torch
from model import load_mishkala, tashkeel

model, tokenizer, device = load_mishkala()
print(f"Model loaded on {device}", flush=True)

# Read WikiNews 2024 multi-ref file and generate bare input
wn2024_path = REPO / "data" / "diac" / "raw" / "wikinews" / "wikinews2024_multi_ref.diac"
lines = []
for line in wn2024_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    # Strip marks to get bare input
    bare = strip_marks(line)
    lines.append(bare)

print(f"Processing {len(lines)} lines...", flush=True)

# Generate predictions
outs = []
for i, line in enumerate(lines):
    try:
        result = tashkeel(line, model, tokenizer, device)
        outs.append(result)
    except Exception as e:
        outs.append(f"ERROR: {str(e)[:100]}")
    if (i + 1) % 50 == 0:
        print(f"[{i+1}/{len(lines)}]", flush=True)

# Write predictions
out_path = REPO / "models" / "e19" / "wn2024.mishkala.pred.txt"
out_path.write_text("\n".join(outs) + "\n", encoding="utf-8")
print(f"Written to {out_path}", flush=True)
