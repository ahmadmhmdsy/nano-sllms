"""Reorganise the built corpus into train/val dirs and build a compact vocab.

The corpus builder writes one flat labels.json plus labels_val.json; the
dataset expects <root>/labels.json, so this rewrites the paths into train/ and
val/ subdirectories and derives a frequency-pruned character vocabulary.
"""
import argparse, collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'linecorpus'


def remap(p):
    if p.startswith('images/'):
        return '../' + p
    if p.startswith('../hflines/'):
        return '../' + p
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-count', type=int, default=20)
    args = ap.parse_args()
    train = json.load(open(OUT / 'labels.json', encoding='utf-8'))
    val = json.load(open(OUT / 'labels_val.json', encoding='utf-8'))
    for split, rows in (('train', train), ('val', val)):
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            r['image'] = remap(r['image'])
        json.dump(rows, open(d / 'labels.json', 'w', encoding='utf-8'), ensure_ascii=False)
        print('%-5s %6d rows -> %s' % (split, len(rows), d / 'labels.json'))

    freq = collections.Counter()
    for r in train + val:
        freq.update(r['text'])
    specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    chars = [c for c, n in freq.most_common() if n >= args.min_count and c not in specials]
    char_to_idx = {c: i for i, c in enumerate(specials)}
    for c in chars:
        char_to_idx[c] = len(char_to_idx)
    idx_to_char = {str(v): k for k, v in char_to_idx.items()}
    vp = ROOT / 'data' / 'linecorpus_vocab.json'
    json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char},
              open(vp, 'w', encoding='utf-8'), ensure_ascii=False)
    total = sum(freq.values())
    cov = sum(freq[c] for c in chars) / max(total, 1)
    print()
    print('distinct chars seen : %d' % len(freq))
    print('vocab kept (>=%d)   : %d  (4 specials + %d chars)' % (args.min_count, len(char_to_idx), len(chars)))
    print('char coverage       : %.4f%%' % (cov * 100))
    print('vocab written       : %s' % vp)
    rare = [(c, n) for c, n in freq.most_common() if n < args.min_count]
    if rare:
        print('pruned (rare)       : %d types, %.4f%% of chars' % (
            len(rare), 100 * sum(n for _, n in rare) / max(total, 1)))


if __name__ == '__main__':
    main()
