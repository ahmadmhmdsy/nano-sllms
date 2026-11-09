"""Build a clean, deduplicated line-level OCR dataset from all local sources.

Rationale: the ISIPPT and KHATT benchmarks are single text LINE images
(median 38 and 64 chars). Training was previously done on full-page images
(median ~600 chars) truncated to 64 tokens, so the model never saw the task
it was being scored on.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image

PROMPT = 'ما هو النص المكتوب فى الصورة؟ برجاء الاجابة بالنص فقط'

REAL_SOURCES = [
    'data/all_data/train', 'data/all_data/val',
    'data/large/train', 'data/large/val',
    'data/qari_10k',
    'data/combined/train', 'data/combined/val',
    'data/all_combined',
    'data/small/train', 'data/small/val',
]
SYNTH_SOURCES = ['data/full/train', 'data/full/val']


def collect(sources, max_h, min_len, max_len, seen, limit=None):
    rows = []
    for d in sources:
        lp = Path(d) / 'labels.json'
        if not lp.exists():
            print('  skip (no labels):', d)
            continue
        lab = json.loads(lp.read_text(encoding='utf-8'))
        kept = dup = filt = bad = 0
        for it in lab:
            t = it['text'].strip()
            if not (min_len <= len(t) <= max_len):
                filt += 1
                continue
            if t == PROMPT or t.startswith('ما هو النص المكتوب'):
                filt += 1
                continue
            p = Path(d) / it['image']
            if not p.exists():
                bad += 1
                continue
            try:
                w, h = Image.open(p).size
            except Exception:
                bad += 1
                continue
            if h > max_h or w < 40 or h < 8:
                filt += 1
                continue
            try:
                digest = hashlib.md5(p.read_bytes()).hexdigest()
            except Exception:
                bad += 1
                continue
            if digest in seen:
                dup += 1
                continue
            seen.add(digest)
            rows.append((str(p), t, w, h))
            kept += 1
        print('  %-24s kept=%-6d dup=%-6d filtered=%-6d missing=%d' % (d, kept, dup, filt, bad))
    if limit and len(rows) > limit:
        random.Random(0).shuffle(rows)
        rows = rows[:limit]
    return rows


def write_split(rows, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    labels = []
    for i, (src, text, w, h) in enumerate(rows):
        name = '%06d.png' % i
        shutil.copyfile(src, out_dir / name)
        labels.append({'image': name, 'text': text})
    (out_dir / 'labels.json').write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding='utf-8')
    return len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-height', type=int, default=120)
    ap.add_argument('--min-len', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--val-frac', type=float, default=0.08)
    ap.add_argument('--include-synthetic', action='store_true')
    ap.add_argument('--synthetic-limit', type=int, default=12000)
    ap.add_argument('--out', default='data/lines')
    args = ap.parse_args()

    print('Collecting real line-level images:')
    seen = set()
    rows = collect(REAL_SOURCES, args.max_height, args.min_len, args.max_len, seen)
    print('real line-level total:', len(rows))

    if args.include_synthetic:
        print('Collecting synthetic word images:')
        syn = collect(SYNTH_SOURCES, args.max_height, args.min_len, args.max_len,
                      seen, limit=args.synthetic_limit)
        print('synthetic total:', len(syn))
        rows.extend(syn)

    if not rows:
        raise SystemExit('no rows collected')

    random.Random(1234).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    n_tr = write_split(train, Path(args.out) / 'train')
    n_va = write_split(val, Path(args.out) / 'val')
    print('WROTE train=%d val=%d -> %s' % (n_tr, n_va, args.out))


if __name__ == '__main__':
    main()
