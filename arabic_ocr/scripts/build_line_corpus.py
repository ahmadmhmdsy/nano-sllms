"""Build a combined line-level Arabic OCR corpus from local sources.

Sources: the local HF parquet cache (persian line images, QARI-10k, SAQR
handwriting) plus the already-materialised data/hflines real-line corpus.
Applies a benchmark text-contamination guard and cross-source dedup.
"""
import argparse, glob, io, json, os, re, sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

CACHE = Path(os.path.expanduser('~/.cache/huggingface/hub'))
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'linecorpus'


def norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip()


def bench_texts():
    out = set()
    for d in sorted((ROOT / 'data' / 'benchmark').glob('*/labels.json')):
        for r in json.load(open(d, encoding='utf-8')):
            out.add(norm(r['text']))
    return out


def parquets(dirname):
    return sorted(CACHE.glob(dirname + '/**/*.parquet'))


def img_bytes(cell):
    if isinstance(cell, dict):
        return cell.get('bytes')
    return cell


class Writer:
    def __init__(self):
        self.rows = []
        self.seen = set()
        self.banned = bench_texts()
        self.dropped_bench = 0
        self.dropped_dup = 0
        self.n = 0

    def add(self, img, text, source):
        t = norm(text)
        if not t:
            return
        if t in self.banned:
            self.dropped_bench += 1
            return
        key = t
        if key in self.seen:
            self.dropped_dup += 1
            return
        self.seen.add(key)
        self.n += 1
        rel = 'images/%s/%06d.jpg' % (source, self.n)
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.convert('RGB').save(p, 'JPEG', quality=90)
        self.rows.append({'image': rel, 'text': text, 'source': source})


def add_parquet(w, dirname, source, text_col, image_col, limit=0, msg_mode=False):
    files = parquets(dirname)
    if not files:
        print('  [skip] no parquet for', source, flush=True)
        return
    k = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256):
            cols = batch.schema.names
            for i in range(batch.num_rows):
                if msg_mode:
                    msgs = batch.column('messages')[i].as_py() or []
                    raw = None
                    txt = None
                    for m in msgs:
                        if m.get('images'):
                            raw = img_bytes(m['images'][0])
                        c = str(m.get('content', ''))
                        if c and '<image>' not in c:
                            txt = c
                    if raw is None or not txt:
                        continue
                else:
                    raw = img_bytes(batch.column(image_col)[i].as_py())
                    txt = batch.column(text_col)[i].as_py()
                    if raw is None or not txt:
                        continue
                try:
                    im = Image.open(io.BytesIO(raw))
                    im.load()
                except Exception:
                    continue
                w.add(im, txt, source)
                k += 1
                if limit and k >= limit:
                    print('  %-10s %d added (limit)' % (source, k), flush=True)
                    return
        print('  %-10s %d added so far' % (source, k), flush=True)
    print('  %-10s %d added' % (source, k), flush=True)


def add_hflines(w):
    for split in ('train', 'val'):
        lp = ROOT / 'data' / 'hflines' / split / 'labels.json'
        if not lp.exists():
            print('  [skip] no', lp, flush=True)
            continue
        n = 0
        for r in json.load(open(lp, encoding='utf-8')):
            t = norm(r['text'])
            if not t or t in w.banned or t in w.seen:
                if t in w.banned:
                    w.dropped_bench += 1
                elif t:
                    w.dropped_dup += 1
                continue
            w.seen.add(t)
            w.rows.append({'image': '../hflines/%s/%s' % (split, r['image']),
                           'text': r['text'], 'source': 'hflines'})
            n += 1
        print('  hflines/%s %d added' % (split, n), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-per-source', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.02)
    args = ap.parse_args()
    print('building line corpus ->', OUT, flush=True)
    w = Writer()
    add_hflines(w)
    add_parquet(w, 'datasets--mohajesmaeili--Persian_Arabic_TextLine_Image_Ocr_Small',
                'persian', 'Text', 'Image', args.limit_per_source)
    add_parquet(w, 'datasets--melsiddieg--qari-arabic-ocr-10k',
                'qari10k', None, None, args.limit_per_source, msg_mode=True)
    add_parquet(w, 'datasets--mahmoudsalah01--SAQR',
                'saqr_hw', 'text', 'hw_image', args.limit_per_source)
    add_parquet(w, 'datasets--mahmoudsalah01--SAQR',
                'saqr_gt', 'text', 'gt_image', args.limit_per_source)
    rows = w.rows
    import hashlib
    train, val = [], []
    for r in rows:
        h = int(hashlib.md5(norm(r['text']).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < args.val_frac else train).append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(OUT / 'labels.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(val, open(OUT / 'labels_val.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print()
    print('TOTAL kept      :', len(rows))
    print('dropped (bench) :', w.dropped_bench)
    print('dropped (dup)   :', w.dropped_dup)
    print('train / val     : %d / %d' % (len(train), len(val)))
    from collections import Counter
    print('by source       :', dict(Counter(r['source'] for r in rows)))


if __name__ == '__main__':
    main()
