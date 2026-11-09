"""Pack prepared windows into fixed-shape token blocks for training.

Output data/diac/<phase>/tokens/*.npy:
  train_ids / val_ids : int64 [n, ctx]  PAD-padded char-token streams
  train_y / val_y     : int8  [n, ctx]  15-class labels, -1 where no target

Deterministic: same seed + inputs -> byte-identical files.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import tokenizer as TK  # noqa: E402

PREPARED = REPO / "data" / "diac" / "prepared"


def read_jsonl(path):
    """Stream-parse lines into (bases, labels) pairs to keep RAM bounded.

    The v2 corpus has 1.5M windows over ~84 MB raw text; loading full row
    dicts into Python objects reached multi-GB and MemoryError'd.
    """
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append((r["bases"], r["labels"]))
    return out


def pack_rows(rows, ctx, shuffle_seed, head=None, pad_label=-1):
    import random
    rng = random.Random(shuffle_seed)
    rng.shuffle(rows)
    if head is not None:
        rows = rows[:head]
    n = len(rows)
    ids = np.full((n, ctx), TK.PAD, dtype=np.int64)
    y = np.full((n, ctx), pad_label, dtype=np.int8)
    for i, (bases, labels) in enumerate(rows):
        seq = TK.encode(bases)[:ctx]
        L = len(seq)
        ids[i, :L] = seq
        y[i, :L] = labels[:L]
    return ids, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "pilot", "pilot128", "v2", "v2b",
                                        "fadel_spec", "b65", "v3", "v3s10ctx256",
                                        "v3ctx256"],
                    default="smoke")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=None,
                    help="truncate BEFORE the shuffle")
    # --head-train/--head-val slice AFTER the shuffle, which is a DIFFERENT
    # row set. The v3s10 trial bed is a PREFIX OF THE v3 ARRAYS
    # (v3s10[train_ids] == v3[train_ids][:243343] and v3s10[val_ids] ==
    # v3[val_ids][:9613], verified byte-for-byte 2026-10-06), i.e. it was built
    # by shuffling the whole corpus and then slicing. --max-train does NOT
    # reproduce it: the shuffle of a 243343-row list is not the head of the
    # shuffle of a 2433430-row list. Use --head-train/--head-val for that.
    ap.add_argument("--head-train", type=int, default=None,
                    help="keep the first N train rows AFTER the shuffle")
    ap.add_argument("--head-val", type=int, default=None,
                    help="keep the first N val rows AFTER the shuffle")
    # PAD convention (data/diac/v3/tokens/PAD_CONVENTION.md): this script emits
    # PAD=-1, but every pilot corpus (v3s10) and the deployed gold use PAD=0, and
    # the two differ ONLY at PAD positions. The doc flags as an OPEN HAZARD that
    # a plain re-run silently reintroduces the drift - which is exactly what
    # happened on 2026-10-06 when the first post-doc bed (v3s10ctx256) was built.
    # Default stays -1 so no existing artifact's reproduction changes; pass
    # --pad-label 0 to produce the lineage convention in ONE command.
    ap.add_argument("--pad-label", type=int, default=-1,
                    help="value for label positions beyond the text; the "
                         "lineage convention is 0 (see PAD_CONVENTION.md)")
    ap.add_argument("--prepared-dir", default=None,
                    help="alternate prepared corpus dir (default data/diac/prepared)")
    args = ap.parse_args()
    out_dir = REPO / "data" / "diac" / args.phase / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.prepared_dir) if args.prepared_dir else PREPARED
    train = read_jsonl(src / "train.jsonl")
    val = read_jsonl(src / "val.jsonl")
    if args.max_train:
        train = train[:args.max_train]
    tr_ids, tr_y = pack_rows(train, args.ctx, shuffle_seed=20260911,
                             head=args.head_train, pad_label=args.pad_label)
    va_ids, va_y = pack_rows(val, args.ctx, shuffle_seed=20260912,
                             head=args.head_val, pad_label=args.pad_label)
    np.save(str(out_dir / "train_ids.npy"), tr_ids)
    np.save(str(out_dir / "train_y.npy"), tr_y)
    np.save(str(out_dir / "val_ids.npy"), va_ids)
    np.save(str(out_dir / "val_y.npy"), va_y)
    print({"phase": args.phase, "ctx": args.ctx, "train": len(tr_ids),
           "val": len(va_ids), "vocab_size": TK.VOCAB_SIZE, "out": str(out_dir)})


if __name__ == "__main__":
    main()
