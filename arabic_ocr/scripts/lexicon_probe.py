"""Is a word-lexicon post-correction lever alive? Measure before building it.

Greedy-decodes the cached teacher logits for muharaf_test (CPU only, no GPU) and
reports: the greedy CER recomputed here (self-check - must reproduce the harness
22.11), the share of REFERENCE words missing from a train-only lexicon (the coverage
ceiling), the share of HYPOTHESIS words that are OOV, and - over the most frequent
OOV words - how many have a distance-1 lexicon neighbour and what the CER becomes
when they are replaced by it. The distance-1 search generates all variants of each
OOV word and looks them up, so its cost does not grow with lexicon size.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "arabic_ocr" / "models" / "muharaf_ocr"))
sys.path.insert(0, str(ROOT / "arabic_ocr" / "scripts"))

from submission_code.ocr_helpers import greedy_ctc_decode  # noqa: E402
from finetune_muharaf import CKPT_PATH, load_ctc_backbone  # noqa: E402


def load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load("clc", "arabic_ocr/scripts/char_lm_common.py")


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


def cer(hyps, refs):
    d = sum(lev(h, r) for h, r in zip(hyps, refs))
    n = sum(len(r) for r in refs)
    return 100.0 * d / max(1, n)


def words(text):
    return [C.strip_diacritics(w) for w in text.split() if w]


lex = Counter()
for rel in ["data/hf/muharaf/train/labels.json", "data/hf/khatt_teklia/train/labels.json"]:
    for rec in json.loads((ROOT / rel).read_text(encoding="utf-8")):
        for w in words(rec.get("text", "")):
            lex[w] += 1
lexset = set(lex)
alph = sorted({c for w in lex for c in w})
print("lexicon: %d types / %d tokens, alphabet %d" % (len(lex), sum(lex.values()), len(alph)), flush=True)

_, charset, _ = load_ctc_backbone(str(CKPT_PATH), device="cpu")
npz = np.load(ROOT / "arabic_ocr/runs/stage05e/logits/teacher/muharaf_test.npz", allow_pickle=True)
data, offs = npz["data"], npz["offsets"]
refs = [str(r) for r in npz["refs"]]
hyps = []
for i in range(len(offs) - 1):
    raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
    hyps.append(charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id)))
base = cer(hyps, refs)
print("greedy CER here %.2f (harness expects 22.11)  n=%d" % (base, len(refs)), flush=True)

ref_words = [w for r in refs for w in words(r)]
hyp_words = [w for h in hyps for w in words(h)]
ref_oov = sum(1 for w in ref_words if w not in lexset)
hyp_oov = sum(1 for w in hyp_words if w not in lexset)
print("ref words %d OOV %d (%.2f%%)" % (len(ref_words), ref_oov, 100.0 * ref_oov / max(1, len(ref_words))), flush=True)
print("hyp words %d OOV %d (%.2f%%)" % (len(hyp_words), hyp_oov, 100.0 * hyp_oov / max(1, len(hyp_words))), flush=True)

oov_counts = Counter(w for w in hyp_words if w not in lexset)
CAP = 4000
todo = [w for w, _ in oov_counts.most_common(CAP)]
fix = {}
for w in todo:
    hits = set()
    n = len(w)
    for i in range(n):
        hits.add(w[:i] + w[i + 1:])
    for i in range(n):
        pre, post = w[:i], w[i + 1:]
        for c in alph:
            if c != w[i]:
                hits.add(pre + c + post)
    for i in range(n + 1):
        pre, post = w[:i], w[i:]
        for c in alph:
            hits.add(pre + c + post)
    cands = [h for h in hits if h in lexset]
    if cands:
        fix[w] = max(cands, key=lambda c: lex[c])
covered = sum(oov_counts[w] for w in fix)
print("scanned %d/%d distinct OOV (%.1f%% of OOV tokens); distance-1 reachable %d (%.1f%% of scanned), covering %d tokens"
      % (len(todo), len(oov_counts), 100.0 * sum(oov_counts[w] for w in todo) / max(1, hyp_oov),
         len(fix), 100.0 * len(fix) / max(1, len(todo)), covered), flush=True)

corrected = [" ".join(fix.get(w, w) for w in words(h)) for h in hyps]
after = cer(corrected, refs)
print("CER after the distance-1 lexicon rule on the covered words: %.2f (delta %+.2f)" % (after, after - base), flush=True)

out = {
    "greedy_cer": round(base, 2), "lexicon_types": len(lex),
    "ref_oov_pct": round(100.0 * ref_oov / max(1, len(ref_words)), 2),
    "hyp_oov_pct": round(100.0 * hyp_oov / max(1, len(hyp_words)), 2),
    "distinct_oov": len(oov_counts), "scanned": len(todo),
    "reachable_d1": len(fix), "covered_oov_tokens": covered,
    "cer_after_rule": round(after, 2),
}
(ROOT / "arabic_ocr/runs/stage05e/lexicon_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote arabic_ocr/runs/stage05e/lexicon_probe.json", flush=True)
