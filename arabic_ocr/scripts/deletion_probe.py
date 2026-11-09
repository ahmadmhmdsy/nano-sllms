"""WHAT is the model dropping? The 65%-deletion error mode, decomposed.

Aligns greedy hypotheses to references and reports: which characters are deleted
(and their per-occurrence deletion RATE), how many are spaces, the per-line
emission ratio histogram (uniform under-emission vs a few collapsed lines), and
the CER with whitespace removed from both sides (how much is segmentation).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "arabic_ocr" / "models" / "muharaf_ocr"))
sys.path.insert(0, str(ROOT / "arabic_ocr" / "scripts"))

from submission_code.ocr_helpers import greedy_ctc_decode  # noqa: E402
from finetune_muharaf import CKPT_PATH, load_ctc_backbone  # noqa: E402


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def deletions(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        prev, cur = D[i - 1], D[i]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == hyp[j - 1] else 1))
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and D[i][j] == D[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append(ref[i - 1])
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            j -= 1
        else:
            raise RuntimeError("stuck")
    return out


_, charset, _ = load_ctc_backbone(str(CKPT_PATH), device="cpu")
L = ROOT / "arabic_ocr/runs/stage05e/logits/teacher"

for set_name in ["muharaf_test", "khatt_teklia_test", "isippt_bench"]:
    npz = np.load(L / (set_name + ".npz"), allow_pickle=True)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    hyps, dels = [], Counter()
    refc, buckets = Counter(), Counter()
    for i in range(len(offs) - 1):
        raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
        h = charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id))
        hyps.append(h)
        refc.update(refs[i])
        dels.update(deletions(refs[i], h))
        if refs[i]:
            r = len(h) / len(refs[i])
            b = "<0.5" if r < 0.5 else "0.5-0.8" if r < 0.8 else "0.8-0.95" if r < 0.95 else "0.95-1.05" if r <= 1.05 else ">1.05"
            buckets[b] += 1
    nd = sum(dels.values())
    nsp = sum(1 for r in refs for c in r if c == " ")
    print("== %s | deletions %d | ref spaces %d (deleted %d = %.0f%% of spaces)" %
          (set_name, nd, nsp, dels[" "], 100.0 * dels[" "] / max(1, nsp)))
    print("   per-line hyp/ref ratio: %s" % dict(buckets))
    top = [c for c, _ in refc.most_common(40)]
    print("   deletion RATE for the 18 most common ref chars: " +
          " ".join("%s %.0f%%(%d/%d)" % (repr(c), 100.0 * dels[c] / max(1, refc[c]), dels[c], refc[c]) for c in top[:18]))
    print("   most-deleted chars overall: " +
          " ".join("%s:%d" % (repr(c), n) for c, n in dels.most_common(14)), flush=True)
    nows = 100.0 * sum(lev(h.replace(" ", ""), r.replace(" ", "")) for h, r in zip(hyps, refs)) / \
        max(1, sum(len(r.replace(" ", "")) for r in refs))
    raw = 100.0 * sum(lev(h, r) for h, r in zip(hyps, refs)) / max(1, sum(len(r) for r in refs))
    print("   CER raw %.2f | with whitespace removed from both sides %.2f\n" % (raw, nows), flush=True)
