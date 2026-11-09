"""WHY does the 4.25M Muharaf model fail? Classify its errors by KIND.

Greedy-decodes cached CTC logits, aligns each hypothesis to its reference, and
reports the edit breakdown (substitution / insertion / deletion), how much of it
involves a SPACE (segmentation errors), how much of the substitution mass is a
DOT-ONLY confusion (same base glyph, different dots - i.e. a purely visual error),
and the top substitution pairs. This separates acoustic/visual errors from
linguistic ones.
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

# same base glyph, different dots / hamza -> a purely VISUAL confusion
GROUPS = ["بتثني", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق",
          "هة", "وؤ", "اأإآٱ", "يئ", "ىي"]
GID = {}
for gi, g in enumerate(GROUPS):
    for ch in g:
        GID.setdefault(ch, gi)


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        prev, cur = D[i - 1], D[i]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
    subs, inss, dels = Counter(), Counter(), Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and D[i][j] == D[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            subs[(ref[i - 1], hyp[j - 1])] += 1
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            dels[ref[i - 1]] += 1
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            inss[hyp[j - 1]] += 1
            j -= 1
        else:
            raise RuntimeError("backtrace stuck at %d,%d" % (i, j))
    return subs, inss, dels


def run(model_tag, set_name, path, charset):
    if not path.exists():
        print("%s/%s: MISSING" % (model_tag, set_name))
        return
    npz = np.load(path, allow_pickle=True)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    S, I, D = Counter(), Counter(), Counter()
    for i in range(len(offs) - 1):
        raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
        h = charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id))
        s, ins, de = align(refs[i], h)
        S.update(s)
        I.update(ins)
        D.update(de)
    ns, ni, nd = sum(S.values()), sum(I.values()), sum(D.values())
    tot = ns + ni + nd
    nref = sum(len(r) for r in refs)
    space = (sum(c for (a, b), c in S.items() if a == " " or b == " ")
             + I[" "] + D[" "])
    dot = sum(c for (a, b), c in S.items() if GID.get(a, -1) == GID.get(b, -2))
    print("%-14s %-18s CER %5.2f | edits %5d = sub %4d (%.0f%%) ins %4d (%.0f%%) del %4d (%.0f%%)"
          % (model_tag, set_name, 100.0 * tot / max(1, nref), tot, ns, 100.0 * ns / max(1, tot),
             ni, 100.0 * ni / max(1, tot), nd, 100.0 * nd / max(1, tot)))
    print("%-14s %-18s   space-involving %4d (%.1f%% of edits) | dot-only substitutions %4d (%.1f%% of subs)"
          % ("", "", space, 100.0 * space / max(1, tot), dot, 100.0 * dot / max(1, ns)))
    top = " ".join("%s->%s:%d" % (a, b, c) for (a, b), c in S.most_common(12))
    print("%-14s %-18s   top subs: %s" % ("", "", top), flush=True)


_, charset, _ = load_ctc_backbone(str(CKPT_PATH), device="cpu")
L = ROOT / "arabic_ocr/runs/stage05e/logits"
for tag, sub in [("teacher", "teacher"), ("adapt11700", "adapt11700")]:
    for st in ["muharaf_test", "khatt_teklia_test", "isippt_bench", "khatt_bench"]:
        run(tag, st, L / sub / (st + ".npz"), charset)
