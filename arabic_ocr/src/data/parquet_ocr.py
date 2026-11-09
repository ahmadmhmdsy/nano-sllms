"""Parquet-backed Arabic OCR dataset over the local HF cache.

Avoids duplicating ~26 GB of images by reading the cached parquet files in
place.  Applies a text-contamination guard against the benchmark label sets
so a benchmark line can never enter training.
"""
import glob, json, os, re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

CACHE = Path(os.path.expanduser('~/.cache/huggingface/hub'))

# source name -> hub dir name.  Benchmarks are deliberately absent.
SOURCES = {
    'synth2':  'datasets--aallail--arabic_ocr_synth_2',
    'persian': 'datasets--mohajesmaeili--Persian_Arabic_TextLine_Image_Ocr_Small',
    'aryad':   'datasets--aryadacademie--Arabic_Ocr_set',
    'qari10k': 'datasets--melsiddieg--qari-arabic-ocr-10k',
    'saqr':    'datasets--mahmoudsalah01--SAQR',
    'salehc4': 'datasets--saleh-c4--arabic-ocr-images',
}

TEXT_KEYS = ('text', 'transcription', 'label', 'sentence', 'ground_truth')
IMAGE_KEYS = ('image', 'img', 'image_bytes')


def _norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip()


def benchmark_texts():
    out = set()
    for d in glob.glob('data/benchmark/*/labels.json'):
        for r in json.load(open(d, encoding='utf-8')):
            out.add(_norm(r['text']))
    return out


def parquet_files(source):
    base = CACHE / SOURCES[source]
    return sorted(glob.glob(str(base / '**' / '*.parquet'), recursive=True))


def _pick(cols, keys):
    for k in keys:
        if k in cols:
            return k
    for c in cols:
        if any(k in c.lower() for k in keys):
            return c
    return None


class ParquetOCRCorpus(Dataset):
    """Concatenates several cached parquet OCR corpora as (PIL image, text)."""

    def __init__(self, sources, limit_per_source=0, drop_benchmark=True):
        self.rows = []          # (file_idx, row_idx, text)
        self.files = []
        self.text_key = None
        self.image_key = None
        banned = benchmark_texts() if drop_benchmark else set()
        self.dropped = 0
        for src in sources:
            fs = parquet_files(src)
            if not fs:
                print('  [warn] no parquet for', src)
                continue
            base_idx = len(self.files)
            for f in fs:
                self.files.append(f)
            kept = 0
            for fi, f in enumerate(fs):
                pf = pq.ParquetFile(f)
                cols = pf.schema_arrow.names
                tk = _pick(cols, TEXT_KEYS)
                ik = _pick(cols, IMAGE_KEYS)
                if tk is None or ik is None:
                    print('  [warn] %s cols=%s -> no text/image' % (src, cols))
                    continue
                self.text_key, self.image_key = tk, ik
                tbl = pf.read(columns=[tk])
                texts = tbl.column(tk).to_pylist()
                for ri, t in enumerate(texts):
                    if not t or not str(t).strip():
                        continue
                    if drop_benchmark and _norm(t) in banned:
                        self.dropped += 1
                        continue
                    self.rows.append((base_idx + fi, ri, str(t)))
                    kept += 1
                    if limit_per_source and kept >= limit_per_source:
                        break
                if limit_per_source and kept >= limit_per_source:
                    break
            print('  %-9s %d rows kept' % (src, kept))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fi, ri, text = self.rows[i]
        pf = pq.ParquetFile(self.files[fi])
        cell = pf.read_row_group(0, columns=[self.image_key]) if False else None
        tbl = pf.read(columns=[self.image_key])
        raw = tbl.column(self.image_key)[ri].as_py()
        if isinstance(raw, dict):
            raw = raw.get('bytes')
        import io
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        return {'image': img, 'text': text}
