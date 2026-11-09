"""Stage 0.5b - build the FULL-corpus dynamic-width label lists.

PaddleOCR's MultiScaleDataSet with ds_width: true reads a FOUR-field list
(path, text, width, height) because wh_aware() unpacks exactly four fields.
The Stage 0.5 pilot built a 2,000-line list this way (E-77/E-78); this
rebuilds it over all 25,921 training lines so the warm-start lever can be
tested at 13x the pilot's data volume.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def build(split: str, out_path: Path) -> int:
    d = REPO / "data" / "hflines" / split
    data = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    rows = []
    missing = 0
    for r in data:
        name = r.get("image") or r.get("img") or r.get("file")
        text = r.get("text", r.get("label", ""))
        # the label file is TAB delimited: a literal tab or newline inside the
        # text would split the row, so neutralise only those (keep spacing)
        text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        p = d / name
        if not p.exists():
            missing += 1
            continue
        with Image.open(p) as im:
            w, h = im.size
        rows.append("%s\t%s\t%d\t%d" % (p.as_posix(), text, w, h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("%-5s %6d rows (missing %d) -> %s" % (split, len(rows), missing, out_path))
    return len(rows)


def main() -> None:
    out_dir = REPO / "runs" / "stage05b"
    build("train", out_dir / "train_list_dyn.txt")
    build("val", out_dir / "val_list_dyn.txt")


if __name__ == "__main__":
    main()
