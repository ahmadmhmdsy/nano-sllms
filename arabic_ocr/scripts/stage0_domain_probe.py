
r"""Stage 0 domain probe - objective fingerprint of the three eval sets.

The aggregate CER says hflines_val is far harder than isippt, but not WHY.
This measures plain image statistics (no model) so the claim "the gap is the
domain" is grounded in the pixels: clean modern print, degraded/historical
print, and handwriting differ in ink fraction, contrast, background chroma
(yellowed paper) and high-frequency energy (scanner noise / paper texture).

Usage: & .\.venv\Scripts\python.exe arabic_ocr\scripts\stage0_domain_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SETS = {
    "hflines_val": REPO / "data/hflines/val",
    "isippt": REPO / "data/benchmark/isippt",
    "khatt": REPO / "data/benchmark/khatt",
}
N = 150


def otsu(g: np.ndarray) -> float:
    hist, _ = np.histogram(g, bins=256, range=(0.0, 256.0))
    tot = g.size
    sa = np.cumsum(hist)
    sb = tot - sa
    wa = sa / max(1, tot)
    wb = sb / max(1, tot)
    va = np.cumsum(hist * np.arange(256)) / np.maximum(sa, 1)
    vb = (np.cumsum((hist * np.arange(256))[::-1])[::-1] - hist * np.arange(256)) / np.maximum(sb, 1)
    var = wa * wb * (va - vb) ** 2
    var[~np.isfinite(var)] = 0
    return float(np.argmax(var))


def one(p: Path):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    a = np.asarray(im, dtype=np.float32)
    g = a.mean(axis=2)
    thr = otsu(g)
    ink = float((g < thr).mean())
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean()
    chroma = float(np.abs(a[:, :, 0] - a[:, :, 2]).mean())
    return dict(w=w, h=h, aspect=w / max(1, h), mean=float(g.mean()), std=float(g.std()),
                ink=ink, hf=float(lap), chroma=chroma)


def main() -> int:
    out = {}
    for name, d in SETS.items():
        data = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        imgs = [d / (r.get("image") or r.get("img") or r.get("file")) for r in data[:N]]
        rows = []
        for p in imgs:
            try:
                rows.append(one(p))
            except Exception:  # noqa: BLE001
                pass
        keys = ["w", "h", "aspect", "mean", "std", "ink", "hf", "chroma"]
        med = {k: round(float(np.median([r[k] for r in rows])), 2) for k in keys}
        med["n"] = len(rows)
        out[name] = med
        print("%-12s n=%3d  %s" % (name, len(rows), json.dumps(med)))
    (REPO / "runs/ocr_bakeoff/domain_probe.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
