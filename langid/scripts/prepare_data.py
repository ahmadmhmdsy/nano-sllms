"""Build train/val TSVs from the Tatoeba export (CPU-only, deterministic)."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import load_tatoeba


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default="data/langid/raw/sentences.tar.bz2")
    ap.add_argument("--out", default="data/langid")
    ap.add_argument("--cap", type=int, default=50000)
    ap.add_argument("--val-per-lang", type=int, default=500)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = load_tatoeba(
        args.tar, cap=args.cap, val_per_lang=args.val_per_lang)
    for name, rows in (("train.tsv", train_rows), ("val.tsv", val_rows)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as f:
            for lang, text in rows:
                f.write(lang + "\t" + text.replace("\t", " ") + "\n")
        print(f"[prepare] {name}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
