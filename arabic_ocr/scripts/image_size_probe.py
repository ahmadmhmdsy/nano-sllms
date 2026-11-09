"""How much resolution is there to win? Source line-image size vs the 96px target.

E-99 showed the failure mode is under-emission of SMALL glyphs, which suggests an
acuity limit. That hypothesis only has room to be true if the preprocessing
DOWNSCALES the source images: if the source lines are already <= target_height,
raising target_height only interpolates - no new information - and the hypothesis
is dead. Measured on a stride sample of each set's labels.json.
"""
import json
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SETS = {
    "muharaf_val": ROOT / "data" / "hf" / "muharaf" / "validation",
    "muharaf_test": ROOT / "data" / "hf" / "muharaf" / "test",
    "khatt_teklia_test": ROOT / "data" / "hf" / "khatt_teklia" / "test",
    "isippt_bench": ROOT / "data" / "benchmark" / "isippt",
    "khatt_bench": ROOT / "data" / "benchmark" / "khatt",
}
TARGET = 96

for tag, folder in SETS.items():
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    step = max(1, len(labels) // 250)
    hs, ws, ratios = [], [], []
    for rec in labels[::step]:
        with Image.open(folder / rec["image"]) as im:
            w, h = im.size
        hs.append(h)
        ws.append(w)
        ratios.append(w / max(1, h))
    hs_s, ws_s = sorted(hs), sorted(ws)
    taller = sum(1 for h in hs if h > TARGET)
    print("%-18s n=%4d  src height med %4.0f (p10 %4.0f p90 %4.0f)  src width med %5.0f  "
          "aspect med %5.2f  taller-than-%d: %.0f%%"
          % (tag, len(hs), statistics.median(hs), hs_s[len(hs) // 10], hs_s[9 * len(hs) // 10],
             statistics.median(ws), statistics.median(ratios), TARGET, 100.0 * taller / len(hs)),
          flush=True)
    med_ratio = statistics.median(ratios)
    print("      height %d -> width %5.0f px  |  height 128 -> %5.0f  |  height 192 -> %5.0f"
          % (TARGET, med_ratio * TARGET, med_ratio * 128, med_ratio * 192), flush=True)
