"""Why is raw CER 22.11 when deletions are 65% of the edits?

Two questions:
(1) Do the REFERENCES contain characters the 165-char charset cannot emit at all
    (diacritics, tatweel, rare forms)? Those are guaranteed deletions - a metric
    artifact, not a model error. Reports raw CER vs CER on emit-able ref chars only.
(2) Is there a frame budget? T = w/4 output frames per line; if a line has more
    reference characters than frames, CTC physically cannot emit them.
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

DIAC = set(chr(c) for c in list(range(0x064B, 0x0660)) + [0x0670, 0x0640, 0x0653, 0x0654, 0x0655])


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


_, charset, _ = load_ctc_backbone(str(CKPT_PATH), device="cpu")
VOCAB = set(charset.characters)
print("charset: %d symbols, blank_id %s" % (len(VOCAB), charset.blank_id), flush=True)
L = ROOT / "arabic_ocr/runs/stage05e/logits/teacher"

for set_name in ["muharaf_test", "muharaf_val", "khatt_teklia_test", "isippt_bench", "khatt_bench"]:
    p = L / (set_name + ".npz")
    if not p.exists():
        print("%-18s MISSING" % set_name)
        continue
    npz = np.load(p, allow_pickle=True)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    hyps, rows = [], []
    for i in range(len(offs) - 1):
        raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
        h = charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id))
        hyps.append(h)
        rows.append((offs[i + 1] - offs[i], len(refs[i]), len(h)))
    nref = sum(len(r) for r in refs)
    nchars = Counter(c for r in refs for c in r)
    oov = Counter({c: n for c, n in nchars.items() if c not in VOCAB})
    noov = sum(oov.values())
    diac = sum(n for c, n in nchars.items() if c in DIAC)
    raw_cer = 100.0 * sum(lev(h, r) for h, r in zip(hyps, refs)) / max(1, nref)
    stripped = ["".join(c for c in r if c in VOCAB) for r in refs]
    st_cer = 100.0 * sum(lev(h, r) for h, r in zip(hyps, stripped)) / max(1, sum(len(r) for r in stripped))
    T = sum(r[0] for r in rows)
    hypc = sum(r[2] for r in rows)
    print("%-18s ref chars %6d | NOT in charset %5d (%.2f%%) | diacritic-ish %5d (%.2f%%)" %
          (set_name, nref, noov, 100.0 * noov / max(1, nref), diac, 100.0 * diac / max(1, nref)))
    print("%-18s CER raw %5.2f -> CER on emit-able ref chars only %5.2f  | ref/hyp chars %d/%d (%.1f%% emitted)" %
          ("", raw_cer, st_cer, nref, hypc, 100.0 * hypc / max(1, nref)))
    print("%-18s frames %6d for %6d ref chars = %.2f chars/frame | top non-charset ref chars: %s" %
          ("", T, nref, nref / max(1, T),
           " ".join("%r:%d" % (c, n) for c, n in oov.most_common(8))), flush=True)
