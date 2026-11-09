"""Stage 1 gate verification: does the synthetic metadata describe the pixels?

The Stage 1 gate (docs/plans/2026-09-29-arabic-ocr-nano-model.md, V2) is:
200 rendered pages verified + a box-coverage assertion (every rendered glyph
lies inside exactly one PAW box). This checks that mechanically instead of
trusting the renderer's own smoke run.

Reported per line:
  coverage    every ink pixel of the line's render falls inside >= 1 of that
              line's recorded PAW boxes (PAWs recomputed from the ink).
  exactness   the recorded PAW boxes equal the ones recomputed from the ink.
  transcript  the recorded "text" equals what was actually rendered.
  orphans     no glyph with ink is left without a PAW.

Reported per page (ink-level, because PAW boxes are axis-aligned bounding
boxes of connected components and MAY overlap as rectangles without any pixel
being ambiguous - a bbox-overlap count is not a defect):
  multi_covered  ink pixels inside >= 2 PAW boxes of the same page.
  foreign_line   ink pixels of line i that fall inside a box of a DIFFERENT
                 line j. This is the one that would actually hurt a detector.
  degraded       share of ink still inside the boxes after degradation,
                 polarity-aware (inverted pages have white-on-black ink).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr" / "src"))

from data.synthetic import (  # noqa: E402
    INK_LEVEL, assert_coverage, fit_line, paws_from_ink,
)


def draw_montage(recs, gate, out_path):
    """Draw PAW (red) and glyph (blue) boxes on the saved pages, side by side."""
    tiles = []
    for rec in recs:
        img = Image.open(gate / rec["image"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for ln in rec["lines"]:
            for g in ln["glyphs"]:
                d.rectangle(g["box"], outline=(60, 110, 230), width=1)
            for p in ln["paws"]:
                d.rectangle(p, outline=(220, 30, 30), width=2)
        tiles.append(img)
    w = sum(t.width for t in tiles)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(out_path)
    return list(sheet.size)


def main():
    ap = argparse.ArgumentParser(description="Verify the Stage 1 synthetic gate")
    ap.add_argument("--gate", default="runs/stage1_gate")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    ap.add_argument("--montage", type=int, default=0, help="pages to draw into a montage")
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gate = Path(args.gate)
    recs = [json.loads(l) for l in (gate / "pages.jsonl").open(encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    font_dir = Path("C:/Windows/Fonts")

    n_lines = cov_fail = box_mismatch = trunc_lines = trunc_pages = 0
    orphan_total = empty_lines = 0
    ink_total = multi_px = foreign_px = uncovered_px = 0
    degraded_fracs = []
    trunc_examples, cov_examples = [], []

    for rec in recs:
        max_w = rec["width"] - 2 * args.margin
        page_trunc = 0
        canvas = np.full((rec["height"], rec["width"]), 255, dtype=np.uint8)
        line_ink, line_cover = [], []
        for ln in rec["lines"]:
            n_lines += 1
            r = fit_line(ln["text"], font_dir / ln["font"], ln["size"], max_w)
            if r is None:
                empty_lines += 1
                continue
            if r.text != ln["text"]:
                trunc_lines += 1
                page_trunc += 1
                if len(trunc_examples) < 5:
                    trunc_examples.append({"page": rec["image"], "recorded": ln["text"],
                                           "actually_rendered": r.text})
            paws, gp = paws_from_ink(r.mask, r.glyphs)
            c = assert_coverage(r.mask, paws, r.glyphs, gp)
            if not c["ok"]:
                cov_fail += 1
                if len(cov_examples) < 5:
                    cov_examples.append({"page": rec["image"], **c})
            orphan_total += c["orphan_glyphs"]

            x, y = ln["box"][0], ln["box"][1]
            rec_paws = [tuple(p) for p in ln["paws"]]
            got_paws = [(p[0] + x, p[1] + y, p[2] + x, p[3] + y) for p in paws]
            if rec_paws != got_paws:
                box_mismatch += 1

            reg = canvas[y:y + r.height, x:x + r.width]
            np.minimum(reg, r.mask, out=reg)
            ink = np.zeros(canvas.shape, dtype=bool)
            ink[y:y + r.height, x:x + r.width] = r.mask < INK_LEVEL
            cover = np.zeros(canvas.shape, dtype=bool)
            for x0, y0, x1, y1 in rec_paws:
                cover[y0:y1, x0:x1] = True
            line_ink.append(ink)
            line_cover.append(cover)
        if page_trunc:
            trunc_pages += 1

        ink_all = canvas < INK_LEVEL
        counts = np.zeros(canvas.shape, dtype=np.int16)
        for cov in line_cover:
            counts += cov.astype(np.int16)
        n_ink = int(ink_all.sum())
        ink_total += n_ink
        uncovered_px += int((ink_all & (counts == 0)).sum())
        multi_px += int((ink_all & (counts >= 2)).sum())
        for i in range(len(line_ink)):
            other = np.zeros(canvas.shape, dtype=bool)
            for j in range(len(line_cover)):
                if j != i:
                    other |= line_cover[j]
            foreign_px += int((line_ink[i] & other).sum())

        img = np.asarray(Image.open(gate / rec["image"]).convert("L"))
        if "invert" in rec["degradations"]:
            img = 255 - img
        dink = img < INK_LEVEL
        dtot = int(dink.sum())
        if dtot:
            dcov = np.zeros(canvas.shape, dtype=bool)
            for cov in line_cover:
                dcov |= cov
            degraded_fracs.append(float((dink & dcov).sum()) / dtot)

    def pct(a, b):
        return round(100.0 * a / b, 4) if b else None

    report = {
        "gate": str(gate),
        "pages": len(recs),
        "lines": n_lines,
        "coverage_fail_lines": cov_fail,
        "paw_box_mismatch_lines": box_mismatch,
        "orphan_glyphs": orphan_total,
        "empty_render_lines": empty_lines,
        "transcript_mismatch_lines": trunc_lines,
        "transcript_mismatch_pages": trunc_pages,
        "clean_ink_px": ink_total,
        "ink_uncovered_px": uncovered_px,
        "ink_uncovered_pct": pct(uncovered_px, ink_total),
        "ink_multi_covered_px": multi_px,
        "ink_multi_covered_pct": pct(multi_px, ink_total),
        "ink_in_foreign_line_box_px": foreign_px,
        "ink_in_foreign_line_box_pct": pct(foreign_px, ink_total),
        "degraded_ink_inside_paws_median": (round(float(np.median(degraded_fracs)), 4) if degraded_fracs else None),
        "degraded_ink_inside_paws_min": (round(float(np.min(degraded_fracs)), 4) if degraded_fracs else None),
        "transcript_examples": trunc_examples,
        "coverage_examples": cov_examples,
    }
    report["GATE"] = "PASS" if (
        cov_fail == 0 and box_mismatch == 0 and orphan_total == 0
        and trunc_lines == 0 and uncovered_px == 0 and foreign_px == 0
    ) else "FAIL"

    if args.montage:
        size = draw_montage(recs[: args.montage], gate, gate / "verify_montage.png")
        report["montage"] = str(gate / "verify_montage.png")
        report["montage_size"] = size

    print(json.dumps({k: v for k, v in report.items() if not k.endswith("examples")},
                     indent=2, ensure_ascii=False))
    for e in trunc_examples:
        print("TRUNC %s recorded=%r rendered=%r" % (e["page"], e["recorded"], e["actually_rendered"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("report -> %s" % args.out)


if __name__ == "__main__":
    main()
