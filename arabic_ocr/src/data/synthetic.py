"""Synthetic Arabic OCR data generation with exact glyph/PAW boxes.

Stage 1 of the Arabic OCR ladder (docs/plans/2026-09-29-arabic-ocr-nano-model.md).

The previous version of this file was a smoke-test toy: it drew raw Python
strings with Latin fonts (arial / times / calibri) through PIL, which on this
machine is built WITHOUT Raqm (ImageFont.core.HAVE_RAQM is False) and therefore
cannot shape Arabic at all - the output was disconnected and left-to-right.
This module replaces it rather than extending it.

Pipeline
    shaping      HarfBuzz (uharfbuzz); direction/script guessed from the text
    rasterizing  FreeType (freetype-py), one glyph id at a time
    glyph boxes  free, straight out of the HarfBuzz advances
    PAW boxes    connected components of the rendered ink, with components
                 merged when they share a HarfBuzz cluster (so a haraka that
                 floats above its letter still belongs to that letter)
    pages        several lines on one canvas, then photographic degradation

A PAW (part of an Arabic word) is the connected unit of the script - the thing
a detector should learn - not a character. Arabic letters join along the
baseline, so the ink components ARE the PAWs. That makes the segmentation
exact instead of a gap threshold, and it is directly verifiable: every ink
pixel must fall inside exactly one PAW box.

Output contract per page: page image + PAW boxes + glyph boxes + transcript +
font and degradation metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image, ImageFilter
from scipy import ndimage

FONT_DIRS = (Path("C:/Windows/Fonts"),)
PROBE_CODEPOINTS = (0x0627, 0x0644, 0x0645)
INK_LEVEL = 200
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def arabic_ratio(text):
    if not text:
        return 0.0
    n = sum(1 for c in text if any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES))
    return n / len(text)


def has_arabic(path):
    try:
        font = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        return all(font.get_nominal_glyph(cp) for cp in PROBE_CODEPOINTS)
    except Exception:
        return False


def find_arabic_fonts(dirs=FONT_DIRS):
    out = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        pats = ("*.ttf", "*.TTF", "*.otf", "*.OTF")
        cand = []
        for pat in pats:
            cand.extend(d.glob(pat))
        for p in sorted(set(cand)):
            if has_arabic(p):
                out.append(p)
    return out


def load_corpus(hflines_root=None, textpool=None, min_arabic=0.6):
    texts = []
    if hflines_root:
        for split in ("train", "val"):
            p = Path(hflines_root) / split / "labels.json"
            if p.exists():
                for d in json.loads(p.read_text(encoding="utf-8")):
                    t = d.get("text", "")
                    if arabic_ratio(t) >= min_arabic:
                        texts.append(t)
    if textpool and Path(textpool).exists():
        with Path(textpool).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = o.get("text") or o.get("plain") or ""
                if arabic_ratio(t) >= min_arabic:
                    texts.append(t)
    return texts


def length_pool(hflines_root):
    p = Path(hflines_root) / "train" / "labels.json"
    if not p.exists():
        return [40, 55, 70, 90]
    return [len(d["text"]) for d in json.loads(p.read_text(encoding="utf-8"))]


def segment_lines(text, target, rng):
    words = text.split()
    lines, cur, n = [], [], 0
    for w in words:
        if n + len(w) + 1 > target and cur:
            lines.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines


@dataclass
class Glyph:
    gid: int
    cluster: int
    x0: int
    y0: int
    x1: int
    y1: int
    advance: float


@dataclass
class Rendered:
    text: str
    font: str
    size: int
    mask: np.ndarray
    glyphs: list
    width: int
    height: int
    baseline: int


def render_line(text, font_path, size, pad=6):
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    upem = face.upem
    font = hb.Font(face)
    font.scale = (upem, upem)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)

    scale = size / upem
    ft = freetype.Face(str(font_path))
    ft.set_char_size(int(round(size * 64)))

    ascender = int(math.ceil(ft.size.ascender / 64.0))
    descender = int(math.ceil(-ft.size.descender / 64.0))
    height = ascender + descender + 2 * pad
    baseline = pad + ascender

    pen_x = 0.0
    pen_y = 0.0
    placed = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        placed.append((info.codepoint, info.cluster,
                       pen_x + pos.x_offset, pen_y + pos.y_offset,
                       pos.x_advance, pos.y_advance))
        pen_x += pos.x_advance
        pen_y += pos.y_advance

    width = int(math.ceil(pen_x * scale)) + 2 * pad
    mask = np.full((height, width), 255, dtype=np.uint8)
    glyphs = []
    for gid, cluster, gx, gy, adv, _yadv in placed:
        ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        b = ft.glyph.bitmap
        if not b.width or not b.rows:
            continue
        g = np.array(b.buffer, dtype=np.uint8).reshape(b.rows, b.width)
        px = int(round(pad + gx * scale + ft.glyph.bitmap_left))
        py = int(round(baseline - gy * scale - ft.glyph.bitmap_top))
        y0, x0 = max(0, py), max(0, px)
        y1, x1 = min(height, py + b.rows), min(width, px + b.width)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = g[y0 - py:y1 - py, x0 - px:x1 - px]
        mask[y0:y1, x0:x1] = np.minimum(mask[y0:y1, x0:x1], sub)
        glyphs.append(Glyph(gid, cluster, x0, y0, x1, y1, adv * scale))
    return Rendered(text, str(font_path), size, mask, glyphs, width, height, baseline)


def paws_from_ink(mask, glyphs):
    """PAW boxes = ink components, merged when they share a HarfBuzz cluster."""
    ink = mask < INK_LEVEL
    if not ink.any():
        return [], []
    labels, n = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return [], []

    owner = {}
    for i, g in enumerate(glyphs):
        for lab in np.unique(labels[g.y0:g.y1, g.x0:g.x1]):
            if lab:
                owner.setdefault(int(lab), set()).add(i)

    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    cluster_to_lab = {}
    for lab, gids in owner.items():
        cl = glyphs[min(gids)].cluster
        if cl in cluster_to_lab:
            union(cluster_to_lab[cl], lab)
        else:
            cluster_to_lab[cl] = lab

    objs = ndimage.find_objects(labels)
    groups = {}
    for lab in range(1, n + 1):
        sl = objs[lab - 1]
        if sl is None:
            continue
        groups.setdefault(find(lab), []).append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))

    paws, lab_paw = [], {}
    for root, boxes in groups.items():
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        paw_id = len(paws)
        paws.append([int(x0), int(y0), int(x1), int(y1)])
        for lab in range(1, n + 1):
            if find(lab) == root:
                lab_paw[lab] = paw_id

    glyph_paw = []
    for i, g in enumerate(glyphs):
        labs = [int(l) for l in np.unique(labels[g.y0:g.y1, g.x0:g.x1]) if l]
        glyph_paw.append(lab_paw.get(labs[0], -1) if labs else -1)

    order = sorted(range(len(paws)), key=lambda i: paws[i][0])
    remap = {old: new for new, old in enumerate(order)}
    paws = [paws[i] for i in order]
    glyph_paw = [remap.get(p, -1) for p in glyph_paw]
    return paws, glyph_paw


def assert_coverage(mask, paws, glyphs, glyph_paw):
    ink = mask < INK_LEVEL
    total = int(ink.sum())
    if total == 0:
        return {"ok": True, "total_ink": 0, "covered": 0, "orphan_glyphs": 0, "overlap_px": 0}
    counts = np.zeros(mask.shape, dtype=np.int16)
    for x0, y0, x1, y1 in paws:
        counts[y0:y1, x0:x1] += 1
    covered = int((ink & (counts >= 1)).sum())
    overlap = int((ink & (counts >= 2)).sum())
    orphan = sum(1 for i, g in enumerate(glyphs) if glyph_paw[i] < 0 and ink[g.y0:g.y1, g.x0:g.x1].any())
    return {"ok": covered == total and orphan == 0, "total_ink": total,
            "covered": covered, "orphan_glyphs": int(orphan), "overlap_px": overlap}


def degrade(img, rng):
    applied = []
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if rng.random() < 0.35:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4))), dtype=np.float32)
        applied.append("blur")
    if rng.random() < 0.5:
        a = a + np.asarray(Image.effect_noise(img.size, rng.uniform(4, 18)), dtype=np.float32) - 128.0
        applied.append("noise")
    if rng.random() < 0.3:
        a = a * rng.uniform(0.82, 1.05) + rng.uniform(-12, 22)
        applied.append("contrast")
    a = np.clip(a, 0, 255).astype(np.uint8)
    out = Image.fromarray(a)
    if rng.random() < 0.3:
        out = out.rotate(rng.uniform(-1.2, 1.2), resample=Image.BILINEAR, fillcolor=255)
        applied.append("rotate")
    if rng.random() < 0.35:
        import io
        buf = io.BytesIO()
        out.convert("L").save(buf, format="JPEG", quality=int(rng.uniform(35, 85)))
        buf.seek(0)
        out = Image.open(buf).convert("L")
        applied.append("jpeg")
    if rng.random() < 0.12:
        out = Image.fromarray(255 - np.asarray(out.convert("L")))
        applied.append("invert")
    return out.convert("RGB"), applied


def fit_line(text, font_path, size, max_w):
    """Render text, dropping trailing words until it fits max_w pixels."""
    words = text.split()
    while words:
        r = render_line(" ".join(words), font_path, size)
        if r.width <= max_w:
            return r
        keep = max(1, int(len(words) * max_w / float(r.width)) - 1)
        words = words[:keep]
    return None


def render_page(texts, fonts, rng, size_range, page_w, page_h, margin, line_gap):
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    lines = []
    y = margin
    for text in texts:
        size = rng.randint(*size_range)
        font_path = rng.choice(fonts)
        r = fit_line(text, font_path, size, page_w - 2 * margin)
        if r is None:
            continue
        if y + r.height + line_gap > page_h - margin:
            break
        x = rng.randint(margin, max(margin, page_w - r.width - margin))
        rgb = np.repeat(r.mask[:, :, None], 3, axis=2)
        region = canvas[y:y + r.height, x:x + r.width]
        np.minimum(region, rgb, out=region)
        paws, glyph_paw = paws_from_ink(r.mask, r.glyphs)
        lines.append({
            # r.text, NOT text: fit_line() drops trailing words until the line
            # fits the page width, so the pre-fit string would describe ink
            # that was never drawn (25/1600 lines in the 200-page gate).
            "text": r.text,
            "font": Path(r.font).name,
            "size": size,
            "box": [int(x), int(y), int(x + r.width), int(y + r.height)],
            "baseline_y": int(y + r.baseline),
            "paws": [[int(p[0] + x), int(p[1] + y), int(p[2] + x), int(p[3] + y)] for p in paws],
            "glyphs": [{"box": [int(g.x0 + x), int(g.y0 + y), int(g.x1 + x), int(g.y1 + y)],
                        "paw": int(glyph_paw[i])} for i, g in enumerate(r.glyphs)],
        })
        y += r.height + line_gap
    page, applied = degrade(Image.fromarray(canvas), rng)
    return page, lines, applied


def main():
    ap = argparse.ArgumentParser(description="Synthetic Arabic OCR pages with exact glyph/PAW boxes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hflines", default="data/hflines")
    ap.add_argument("--textpool", default="data/ocr_textpool/texts.jsonl")
    ap.add_argument("--page-w", type=int, default=1200)
    ap.add_argument("--page-h", type=int, default=900)
    ap.add_argument("--lines-per-page", type=int, default=8)
    ap.add_argument("--size-min", type=int, default=28)
    ap.add_argument("--size-max", type=int, default=44)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ascii", type=int, default=0)
    ap.add_argument("--font-limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = find_arabic_fonts()
    if args.font_limit:
        fonts = fonts[:args.font_limit]
    if not fonts:
        raise SystemExit("no Arabic-capable fonts found")
    corpus = load_corpus(args.hflines, args.textpool)
    lens = length_pool(args.hflines)
    if not corpus:
        raise SystemExit("empty corpus")
    print("fonts %d | corpus docs %d" % (len(fonts), len(corpus)))

    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    meta_path = out / "pages.jsonl"
    bad = 0
    pool = []
    need = args.pages * args.lines_per_page + 64
    while len(pool) < need:
        doc = rng.choice(corpus)
        for t in segment_lines(doc, rng.choice(lens), rng):
            if arabic_ratio(t) >= 0.6:
                pool.append(t)
    with meta_path.open("w", encoding="utf-8") as mf:
        idx = 0
        for i in range(args.pages):
            texts = pool[idx:idx + args.lines_per_page]
            idx += args.lines_per_page
            if len(texts) < 2:
                break
            page, lines, applied = render_page(texts, fonts, rng, (args.size_min, args.size_max),
                                               args.page_w, args.page_h, 24, 10)
            name = "pages/%06d.png" % i
            page.save(out / name)
            rec = {"image": name, "width": page.width, "height": page.height,
                   "degradations": applied, "lines": lines}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.verify:
                for ln in lines:
                    r = render_line(ln["text"], Path("C:/Windows/Fonts") / ln["font"], ln["size"])
                    paws, gp = paws_from_ink(r.mask, r.glyphs)
                    c = assert_coverage(r.mask, paws, r.glyphs, gp)
                    if not c["ok"]:
                        bad += 1
                        print("COVERAGE FAIL page %d: %s" % (i, c))
            if args.ascii and i < args.ascii:
                a = np.asarray(page.convert("L")) < INK_LEVEL
                step = max(1, a.shape[1] // 150)
                small = a[:, ::step]
                rows = max(1, a.shape[0] // 6)
                print("--- page %d (%d lines, %s) ---" % (i, len(lines), ",".join(applied) or "clean"))
                for rr in range(rows):
                    band = small[rr * 6:(rr + 1) * 6]
                    print("".join("#" if band[:, c].any() else "." for c in range(band.shape[1])))
    print("wrote %d pages -> %s" % (args.pages, meta_path))
    if args.verify:
        print("coverage failures: %d" % bad)
        print("VERIFY %s" % ("PASS" if bad == 0 else "FAIL"))


if __name__ == "__main__":
    main()
