"""mex/scripts/build_mu2_g2.py — rung G2 corpus (E-31, row-first committed).

Mark-drop fill-in on the G1 corpus, SINGLE VOCab (no new ids): every harakat
mark char becomes '|' (the separator glyph, in-vocab since mu0) while the
marked positions 1:1 with the clean stream, so the same shifted-CE
PackedDataset stays aligned: input stream has '|' where the label stream has
the actual mark. Outputs:
  data/mex/mu2/g2/train_mask.txt   85% of G1 train chars, marks->'|'
  data/mex/mu2/g2/train_replay.txt 15% retention tail, marks KEPT (replay)
  data/mex/mu2/g2/val_task.txt     mask-formatted val (task gate)
  data/mex/mu2/g2/holdout_retent.txt  clean-format val (retention probe,
                                   kept OUT of tokens_dir on purpose)
Reads G1 files read-only; deterministic char-threshold split (no RNG need).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G1 = ROOT / "data" / "mex" / "mu2" / "g1"
OUT = ROOT / "data" / "mex" / "mu2" / "g2"
REPLAY_CHARS = 0.15
MARKS = set("ًٌٍَُِّّْ")


def mask(text: str) -> str:
    return "".join(("|" if (c in MARKS) else c) for c in text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = (G1 / "train.txt").read_text(encoding="utf-8", errors="ignore")
    lines = tr.splitlines()
    # deterministic 85/15 split: every 7th line (~14% chars) -> replay tail
    n_rep = n_mask = 0
    with (OUT / "train_mask.txt").open("w", encoding="utf-8", newline="\n") as f_m, \
         (OUT / "train_replay.txt").open("w", encoding="utf-8", newline="\n") as f_r:
        for i, line in enumerate(lines):
            if i % 7 == 3:      # every 7th clean block
                f_r.write(line + "\n"); n_rep += len(line) + 1
            else:
                f_m.write(mask(line) + "\n"); n_mask += len(line) + 1
    va = (G1 / "val.txt").read_text(encoding="utf-8", errors="ignore")
    (OUT / "val_task.txt").write_text(mask(va), encoding="utf-8", newline="\n")
    (OUT / "holdout_retent.txt").write_text(va, encoding="utf-8", newline="\n")
    print(f"G2: mask {n_mask} chars | replay {n_rep} chars "
          f"({n_rep / max(n_rep + n_mask, 1):.3f} replay rate) | val task/ret "
          f"chars={len(va)}")


if __name__ == "__main__":
    main()
