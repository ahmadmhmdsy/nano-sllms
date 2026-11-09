"""Build a combined line-level Arabic OCR corpus from downloaded HF data.

Merges every corpus under data/hf/, filters to line-like samples, removes
exact duplicates (by image md5), drops any image that also appears in a
benchmark (contamination guard), and writes data/hflines/{train,val}.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def md5_of(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_corpus(root):
    rows = []
    for labels in sorted(Path(root).rglob("labels.json")):
        split = labels.parent.name
        for it in json.loads(labels.read_text(encoding="utf-8")):
            rows.append((labels.parent / it["image"], it["text"].strip(), split))
    return rows


def benchmark_hashes(bench_root):
    hashes = set()
    for labels in Path(bench_root).glob("*/labels.json"):
        for it in json.loads(labels.read_text(encoding="utf-8")):
            p = labels.parent / it["image"]
            if p.exists():
                hashes.add(md5_of(p))
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", default="data/hf")
    ap.add_argument("--extra", default="data/reallines", help="additional corpus to merge")
    ap.add_argument("--bench-root", default="data/benchmark")
    ap.add_argument("--out", default="data/hflines")
    ap.add_argument("--max-height", type=int, default=160)
    ap.add_argument("--max-text-len", type=int, default=256)
    ap.add_argument("--min-text-len", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # Held-out "test" splits are never trained on; they stay available for
    # our own honest evaluation and keep the benchmark guard unambiguous.
    rows = [r for r in load_corpus(args.hf_root) if r[2] != "test"]
    if args.extra and Path(args.extra).exists():
        rows += load_corpus(args.extra)
    print("raw rows:", len(rows))

    bench = benchmark_hashes(args.bench_root)
    print("benchmark image hashes:", len(bench))

    kept, seen, stats = [], set(), Counter()
    for path, text, split in rows:
        if not (args.min_text_len <= len(text) <= args.max_text_len):
            stats["text_len"] += 1
            continue
        if not path.exists():
            stats["missing"] += 1
            continue
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            stats["unreadable"] += 1
            continue
        if h > args.max_height or w < 24:
            stats["too_tall_or_narrow"] += 1
            continue
        d = md5_of(path)
        if d in bench:
            stats["BENCHMARK_OVERLAP"] += 1
            continue
        if d in seen:
            stats["duplicate"] += 1
            continue
        seen.add(d)
        kept.append((path, text, split, w, h))

    print("kept:", len(kept))
    print("dropped:", dict(stats))
    if kept:
        hs = sorted(k[4] for k in kept)
        ls = sorted(len(k[1]) for k in kept)
        print("height  min=%d p50=%d p90=%d max=%d" % (hs[0], hs[len(hs)//2], hs[int(len(hs)*0.9)], hs[-1]))
        print("textlen min=%d p50=%d p90=%d max=%d" % (ls[0], ls[len(ls)//2], ls[int(len(ls)*0.9)], ls[-1]))
        print("splits:", dict(Counter(k[2] for k in kept)))

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac))
    val, train = kept[:n_val], kept[n_val:]

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for name, part in (("train", train), ("val", val)):
        d = out / name
        d.mkdir(parents=True)
        labels = []
        for i, (src, text, _split, _w, _h) in enumerate(part):
            fn = "%06d.png" % i
            shutil.copyfile(src, d / fn)
            labels.append({"image": fn, "text": text})
        (d / "labels.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        print("%s/%s: %d" % (args.out, name, len(labels)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
