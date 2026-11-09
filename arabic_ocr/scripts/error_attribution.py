"""Attribute the raw CER: how much is model error vs evaluation convention vs data?

(1) raw CER vs norm_ar CER (the convention published OCR cards use) vs CER on
    emit-able reference characters only.
(2) deletion rate per character, and the mean for dot-bearing vs dotless letters
    (a visual-acuity signature would show small/dotted characters dropped more).
"""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "arabic_ocr" / "models" / "muharaf_ocr"))
sys.path.insert(0, str(ROOT / "arabic_ocr" / "scripts"))

from submission_code.ocr_helpers import greedy_ctc_decode  # noqa: E402
from finetune_muharaf import CKPT_PATH, load_ctc_backbone  # noqa: E402

DOT = set("بتثجخذرزشضظغفقني")
NODOT = set("احلومهةكسصطعرد")
DIAC = set(chr(c) for c in list(range(0x064B, 0x0660)) + [0x0670, 0x0640])


def norm_ar(t):
    t = unicodedata.normalize("NFKC", t).replace("\u0640", "")
    t = "".join(c for c in t if not (0x064B <= ord(c) <= 0x0652))
    t = t.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    t = t.replace("\u0649", "\u064A")
    return " ".join(t.split()).strip()


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
VOCAB = set(charset.characters)
L = ROOT / "arabic_ocr/runs/stage05e/logits/teacher"

for set_name in ["muharaf_test", "muharaf_val", "khatt_teklia_test"]:
    npz = np.load(L / (set_name + ".npz"), allow_pickle=True)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    hyps = []
    for i in range(len(offs) - 1):
        raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
        hyps.append(charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id)))
    nr = max(1, sum(len(r) for r in refs))
    raw_c = 100.0 * sum(lev(h, r) for h, r in zip(hyps, refs)) / nr
    norm_c = 100.0 * sum(lev(norm_ar(h), norm_ar(r)) for h, r in zip(hyps, refs)) / nr
    emit_refs = ["".join(c for c in r if c in VOCAB) for r in refs]
    em_c = 100.0 * sum(lev(h, r) for h, r in zip(hyps, emit_refs)) / max(1, sum(len(r) for r in emit_refs))
    print("== %s | raw CER %.2f | norm_ar %.2f (%+.2f) | emit-able refs only %.2f (%+.2f)"
          % (set_name, raw_c, norm_c, norm_c - raw_c, em_c, em_c - raw_c), flush=True)
    refc, dels = Counter(), Counter()
    for r, h in zip(refs, hyps):
        refc.update(r)
        dels.update(deletions(r, h))
    rows = [(c, dels[c] / refc[c], refc[c]) for c in refc if refc[c] >= 200 and c != " "]
    rows.sort(key=lambda x: x[1])
    print("   deletion rate (n>=200): " + " ".join("%s%.0f%%" % (c, 100 * r) for c, r, _ in rows))
    dg = [r for c, r, _ in rows if c in DOT]
    ng = [r for c, r, _ in rows if c in NODOT]
    if dg and ng:
        print("   mean deletion rate: dot-bearing %.1f%% (%d chars) vs dotless %.1f%% (%d chars)"
              % (100 * sum(dg) / len(dg), len(dg), 100 * sum(ng) / len(ng), len(ng)), flush=True)
