"""The model under-emits (65% of its edits are deletions). Test the direct fix:
penalise the CTC BLANK logit before the greedy argmax and sweep the penalty.
CPU only, on the cached logits. Reports CER and the emitted-character ratio.
"""
import sys
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


_, charset, _ = load_ctc_backbone(str(CKPT_PATH), device="cpu")
blank = charset.blank_id
L = ROOT / "arabic_ocr/runs/stage05e/logits/teacher"

for set_name in ["muharaf_test", "khatt_teklia_test"]:
    npz = np.load(L / (set_name + ".npz"), allow_pickle=True)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    nref = sum(len(r) for r in refs)
    print("== %s (ref chars %d)" % (set_name, nref), flush=True)
    for delta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        logp = data.astype(np.float32)
        if delta:
            logp[:, blank] -= delta
        hyps = []
        for i in range(len(offs) - 1):
            raw = np.argmax(logp[offs[i]:offs[i + 1]], axis=-1).tolist()
            hyps.append(charset.decode(greedy_ctc_decode(raw, blank_id=blank)))
        cer = 100.0 * sum(lev(h, r) for h, r in zip(hyps, refs)) / max(1, nref)
        emit = 100.0 * sum(len(h) for h in hyps) / max(1, nref)
        print("   blank penalty %.1f -> CER %5.2f | emitted %5.1f%% of ref chars" % (delta, cer, emit), flush=True)
