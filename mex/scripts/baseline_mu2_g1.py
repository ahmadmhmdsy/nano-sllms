"""mex/scripts/baseline_mu2_g1.py — E-30 trivial baseline (backoff 3-gram).

Character 3-gram with additive smoothing + backoff, built from train.txt,
evaluated on val.txt with the SAME next-char cross-entropy as the LM
(newline in-vocab, same CharVocab). Deterministic, CPU-only.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mex.src.vocab import CharVocab

voc = CharVocab()
V = len(voc.vocab)
b0 = defaultdict(int); b1 = defaultdict(lambda: defaultdict(int)); b2 = defaultdict(lambda: defaultdict(int))
txt = (ROOT / "data/mex/mu2/g1/train.txt").read_text(encoding="utf-8", errors="ignore")
ids = []
for line in txt.splitlines():
    ids.extend(voc.encode(line + "\n"))
a = b = -1
for i in ids:
    b2[(a, b)][i] += 1; b1[b][i] += 1; b0[i] += 1; a, b = b, i
HP = 0.3
sum0 = sum(b0.values())
nll = 0.0; n = 0
vt = (ROOT / "data/mex/mu2/g1/val.txt").read_text(encoding="utf-8", errors="ignore")
vids = []
for line in vt.splitlines():
    vids.extend(voc.encode(line + "\n"))
a = b = -1
for i in vids:
    s2 = sum(b2[(a, b)].values()); s1 = sum(b1[b].values())
    c0 = (b0[i] + 1.0) / (sum0 + V)
    c1 = (b1[b][i] + HP * V * c0) / (s1 + HP * V)
    c2 = (b2[(a, b)][i] + HP * V * c1) / (s2 + HP * V) if s2 else c1
    nll -= math.log(c2); n += 1
    a, b = b, i
print(f"backoff-3gram val cross-entropy: {nll / n:.4f} nats/char over {n} positions (V={V})")
