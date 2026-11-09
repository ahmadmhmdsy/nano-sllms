"""mex/scripts/build_mu2_v3q.py - E-56: rebuild gold v3q SFT corpus in the mu g1
stream contract (uint32 PackedDataset shards, CharVocab ids).

Source: data/diac/v3q/tokens/{train,val}_{ids,y}.npy: [rows,128] diacritizer TK
ids (bare text) + per-base-position 15-class labels. Each row is decoded with
TK, marks_for_label inserts the canonical mark char(s) after each Arabic base
char (shadda-first), giving vocalized classical text: the same domain gold
trained on. Re-encoded with the shared 97-id CharVocab, packed like pack_mu2_g1.
CPU only.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diacritizer" / "src"))
from mex.src.vocab import CharVocab
import tokenizer as TK
from labels import marks_for_label

SHARD_TOKENS = 2_000_000
SRC = ROOT / "data" / "diac" / "v3q" / "tokens"
OUT = ROOT / "data" / "mex" / "mu2" / "v3q"
MAX_TRAIN_ROWS = 800_000  # ~25M tokens of gold 45M-char corpus: 2 epochs at 4000 steps


def row_text(ids_row, y_row):
    s = TK.decode([int(i) for i in ids_row])
    out = []
    k = 0
    for ch in s:
        out.append(ch)
        o = ord(ch)
        if 0x0621 <= o <= 0x064A or o == 0x0671:
            lab = int(y_row[k]) if k < len(y_row) else 0
            if lab >= 1:
                out.append(marks_for_label(lab))
            k += 1
    return "".join(out)


def pack(split, cap):
    voc = CharVocab()
    ids = np.load(str(SRC / f"{split}_ids.npy"), mmap_mode="r")
    ys = np.load(str(SRC / f"{split}_y.npy"), mmap_mode="r")
    n_rows = len(ids) if cap is None else min(cap, len(ids))
    buf = []
    sh = 0
    tok = 0
    for r in range(n_rows):
        enc = voc.encode(row_text(ids[r], ys[r]) + "\n")
        buf.extend(enc)
        tok += len(enc)
        if len(buf) >= SHARD_TOKENS:
            np.asarray(buf, dtype=np.uint32).tofile(OUT / f"{split}_{sh:04d}.bin")
            sh += 1
            buf = []
        if (r + 1) % 100000 == 0:
            print(split, r + 1, tok, flush=True)
    if buf:
        np.asarray(buf, dtype=np.uint32).tofile(OUT / f"{split}_{sh:04d}.bin")
    print(split, "done rows", n_rows, "tokens", tok, "shards", sh + 1, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pack("val", None)
    pack("train", MAX_TRAIN_ROWS)

main()
