"""Build the E-097 full-character-set ctx-256 token bed (arm B).

Reads data/diac/prepared_rebuilt/{train,val}.jsonl - the RECORDED source of the
control bed data/diac/v3ctx256/tokens (research/experiments.yaml:1110 records
'tokenize_data.py --phase v3ctx256 --ctx 256 --pad-label 0 --prepared-dir
data/diac/prepared_rebuilt') - re-parses each row's 'text' with the FULL
character set, and writes the four .npy arrays directly.

WHY NOT prepare_data.py + tokenize_data.py: a full-character prepared corpus is
a ~7.5 GB jsonl intermediate (data/diac/prepared_rebuilt is 8.04 GB), and the
bed itself is another ~6 GB. This builder streams instead, so the bed costs only
its own ~6 GB. It also needs no dedupe/split work: prepared_rebuilt already
carries the exact kept rows and the exact train/val split.

ROW ORDER IS MADE IDENTICAL TO THE CONTROL BED, ON PURPOSE. tokenize_data.pack_rows
shuffles the prepared rows with seed 20260911 before filling, so row j of
v3ctx256 is train.jsonl line order[j] where
    order = list(range(N)); random.Random(20260911).shuffle(order)
This builder reproduces that permutation exactly (a Python list shuffle, so the
permutation is identical by construction, not by argument). Result: bed B row j
is the SAME underlying text as bed A row j, so arm A and arm B differ in the
input alphabet alone and train.py's fixed data order (rng = default_rng(777))
draws the same batch composition for both.

The control bed's PAD label convention is 0 (measured: 0 negative cells, range
[0, 14] over 200k sampled cells of v3ctx256/train_y.npy), and --pad-label
defaults to 0 here to match. Non-target positions are labelled BARE (0) exactly
as prepare_data._parse_pair does under PREP_KEEP_ALL=1, so the labels are
byte-comparable with the stripped bed at every base-letter position.

CPU only. Usage (smoke first):
    ./.venv/Scripts/python.exe diacritizer/scripts/e97_build_bed.py --limit 5000
    ./.venv/Scripts/python.exe diacritizer/scripts/e97_build_bed.py
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import passthrough as PT  # noqa: E402
import tokenizer as TK  # noqa: E402

PACK_SHUFFLE_SEED = 20260911   # tokenize_data.pack_rows(train, ..., shuffle_seed=...)
PACK_SHUFFLE_SEED_VAL = 20260912


def count_lines(path):
    """Fast binary line count; the file is ~7 GB, so do not read it as text."""
    n = 0
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            n += b.count(b"\n")
    return n


def full_charset_row(text):
    """text -> (bases, labels) over EVERY character, labels expanded PER CHARACTER.

    passthrough.parse groups a run of non-target characters into ONE Unit whose
    .raw can be many characters long, while tokenizer.encode emits one id per
    CHARACTER. Expanding labels per character is therefore mandatory; otherwise
    len(bases) != len(labels) and the y array silently misaligns.
    """
    units = PT.parse(text)
    parts, labs = [], []
    for u in units:
        parts.append(u.raw)
        if u.is_target:
            labs.append(u.label)
        else:
            labs.extend([0] * len(u.raw))
    bases = "".join(parts)
    if len(bases) != len(labs):
        raise AssertionError("len(bases)=%d != len(labels)=%d" % (len(bases), len(labs)))
    return bases, labs


def build_split(jsonl_path, out_ids, out_y, ctx, pad_label, shuffle_seed,
                limit=None, log_every=200000, no_shuffle=False):
    n_src = count_lines(jsonl_path)
    if limit is not None:
        n_src = min(n_src, limit)
    # The permutation must be built over the rows we will actually write, and
    # pack_rows shuffles the FULL list then fills, so with a --limit smoke build
    # the permutation is over the limited list (a smoke bed is not the real bed).
    order = list(range(n_src))
    if not no_shuffle:
        random.Random(shuffle_seed).shuffle(order)
    inv = [0] * n_src
    for j, i in enumerate(order):
        inv[i] = j

    ids = np.lib.format.open_memmap(str(out_ids), mode="w+", dtype=np.int64,
                                    shape=(n_src, ctx))
    ids[:] = TK.PAD
    y = np.lib.format.open_memmap(str(out_y), mode="w+", dtype=np.int8,
                                  shape=(n_src, ctx))
    y[:] = pad_label

    n_written = 0
    n_trunc = 0
    max_len = 0
    t0 = time.time()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_src:
                break
            row = json.loads(line)
            bases, labs = full_charset_row(row["text"])
            seq = TK.encode(bases)
            L = min(len(seq), ctx)
            if len(seq) > ctx:
                n_trunc += 1
            max_len = max(max_len, len(seq))
            dst = i if no_shuffle else inv[i]
            ids[dst, :L] = seq[:L]
            y[dst, :L] = labs[:L]
            n_written += 1
            if log_every and n_written % log_every == 0:
                print("  %s rows=%d/%d  %.1fs" % (jsonl_path.name, n_written, n_src,
                                                  time.time() - t0), flush=True)
    ids.flush()
    y.flush()
    del ids, y
    if n_written != n_src:
        raise AssertionError("wrote %d rows, expected %d" % (n_written, n_src))
    return {"rows": n_written, "truncated_rows": n_trunc, "max_char_len": max_len,
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default=str(REPO / "data" / "diac" / "prepared_rebuilt"))
    ap.add_argument("--out-dir", default=str(REPO / "data" / "diac" / "v3full" / "tokens"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--pad-label", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: build from the first N rows (not the real bed)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="diagnostic: write rows in file order instead of the pack order")
    args = ap.parse_args()

    src = Path(args.prepared_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print({"prepared_dir": str(src), "out_dir": str(out), "ctx": args.ctx,
           "pad_label": args.pad_label, "limit": args.limit,
           "no_shuffle": args.no_shuffle, "vocab_size": TK.VOCAB_SIZE}, flush=True)

    stats = {"source": str(src.relative_to(REPO)).replace("\\", "/"),
             "ctx": args.ctx, "pad_label": args.pad_label, "limit": args.limit,
             "no_shuffle": args.no_shuffle, "vocab_size": TK.VOCAB_SIZE,
             "keep_all": True, "splits": {}}
    for name, seed in (("train", PACK_SHUFFLE_SEED), ("val", PACK_SHUFFLE_SEED_VAL)):
        print("[%s] start" % name, flush=True)
        st = build_split(src / (name + ".jsonl"),
                         out / (name + "_ids.npy"), out / (name + "_y.npy"),
                         args.ctx, args.pad_label, seed, limit=args.limit,
                         no_shuffle=args.no_shuffle)
        stats["splits"][name] = st
        print("[%s] %s" % (name, st), flush=True)

    (out / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
