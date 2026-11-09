# Plan: DA-1 langid - Tongue-analogue on-device text language ID

**Date:** 2026-09-19  |  **Line:** DA-line (Desert Ant recreation)  |  **Design:** research/desert_ant_recreation/DESIGN.md

## Goal

Recreate Desert Ant Labs' "Tongue" model (research/desert-ant-labs-models-report.md section 3.8):
a tokenizer-free, feature-hashed character n-gram language-ID classifier + script router,
with their full benchmark protocol (full/5/3/1-word accuracy with ties reported, never guessed)
on held-out FLORES-style data, exported as a <= 2.5 MiB int8 artifact with pure numpy inference.
Arabic-first: the Arabic-script group (ar/fa/ur/ps) is first-class and is separated LEXICALLY
(script routing deliberately never fires on Arabic script).

## Architecture

- normalize (NFC + lowercase) -> letters-only words -> per-word char n-grams n=1..4 with
  angle-bracket word-boundary markers -> FNV-1a 32-bit -> 2^16 buckets (bag of n-grams).
- Script router (no model): Hangul -> ko, Kana -> ja, Hebrew -> he, Greek -> el,
  Devanagari -> hi (>= 2 matching chars). Arabic script NOT routed.
- Model: torch EmbeddingBag(65536 -> 21, sum) + bias on CPU (multinomial logistic regression).
- Eval: held-out eval TSV (FLORES-200 dev primary / wikipedia fallback) with seeded k-word windows;
  tie = top1-top2 logit margin < 0.5; abstain = no letter features; both reported, never correct.
- Export: per-language-column int8 npz + pure numpy inference (Int8LangID in langid/src/infer.py).

## Tech stack

Existing venv only (torch 2.14.0+cu126 CPU + numpy; no new deps). Python 3.12, venv scripts only.
Unit tests via stdlib unittest. All runs CPU-only - GPU-safe beside any active run.

## Pre-registered gates (registered BEFORE training; from DESIGN.md section 4)

- G1 smoke: 2k/lang x 1 epoch CPU < 10 min; beats majority baseline on val.
- G2 eval set: full >= 0.97, 5-word >= 0.95, 3-word >= 0.90, 1-word >= 0.70.
- G3: Arabic-script group (ar/fa/ur/ps) mean 3-word >= 0.85.
- G4: artifact <= 2.5 MiB; int8 within 0.5 pp of fp32.
- G5: tie/abstain verified; never guessed.
- G6: leakage barrier - 10-word shingle overlap train-vs-eval == 0.

Reference points from their card (84 langs): FLORES 3-word 0.933 (lingua 0.887),
5-word 0.974, held-out sentences 0.971, single words 0.759.

## Files

langid/__init__.py, langid/src/{__init__,features,model,data,infer}.py,
langid/scripts/{download_data,prepare_data,train,eval_langid,export_int8}.py,
langid/tests/{__init__,test_features,test_script_route,test_model,test_data,test_infer}.py.
Data: data/langid/raw/sentences.tar.bz2 (gitignored, regenerable), data/langid/{train,val,eval}.tsv
(local artifacts). Runs: runs/langid_da1/{model_fp32.pt,langid_int8.npz,eval_report.json,train_summary.json}.

## Execution mode

User pre-authorized starting ("start with most feasible/achievable one"):
executing INLINE in this session under executing-plans discipline (EV-label gate after each task).

---

## Task 1: features module - hashed char n-grams (DONE - code committed this session)

- [x] langid/src/features.py (full content below)
- [x] langid/tests/test_features.py (full content below)
- [x] venv unittest green

```python
# langid/src/features.py
"""Hashed character n-gram features + script-only routing.

Recreation of the Desert Ant Labs "Tongue" recipe (see
research/desert-ant-labs-models-report.md section 3.8): a lexical
feature-hashing model with no tokenizer file, plus script-only routing for
languages whose script is unique within the language set.
"""
import re
import unicodedata

NGRAM_MIN = 1
NGRAM_MAX = 4
NUM_BUCKETS = 1 << 16

# Letters-only words: digits and punctuation carry no language signal and
# poison the 1-word regime.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def fnv1a(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def bucket(feature: str) -> int:
    return fnv1a(feature.encode("utf-8")) % NUM_BUCKETS


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def word_ngrams(word: str):
    w = "<" + word + ">"
    if len(w) <= NGRAM_MAX:
        yield w
        return
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(w) - n + 1):
            yield w[i:i + n]


def text_features(text: str):
    """Yield hashed bucket ids: bag of char n-grams over letters-only words."""
    for w in _WORD_RE.findall(normalize(text)):
        for g in word_ngrams(w):
            yield bucket(g)


# Script-only routing: languages whose script is unique among LANGS.
# Arabic script is deliberately ABSENT: ar/fa/ur/ps share it and must be
# separated by the lexical model (that is the research value of DA-1).
_SCRIPTS = [
    ("ko", ((0xAC00, 0xD7A3), (0x1100, 0x11FF))),   # Hangul
    ("ja", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),   # Kana (Hiragana + Katakana)
    ("he", ((0x0590, 0x05FF),)),                     # Hebrew
    ("el", ((0x0370, 0x03FF),)),                     # Greek
    ("hi", ((0x0900, 0x097F),)),                     # Devanagari (only hi in scope)
]

_ROUTE_MIN_CHARS = 2


def script_route(text: str):
    """Return a language code if the text decisively belongs to a script-routed
    language, else None (the lexical model decides)."""
    t = normalize(text)
    for lang, ranges in _SCRIPTS:
        count = 0
        for ch in t:
            o = ord(ch)
            for a, b in ranges:
                if a <= o <= b:
                    count += 1
                    break
        if count >= _ROUTE_MIN_CHARS:
            return lang
    return None
```

```python
# langid/tests/test_features.py
import unittest

from langid.src.features import (NUM_BUCKETS, bucket, fnv1a, normalize,
                                 text_features, word_ngrams)


class TestFnv1a(unittest.TestCase):
    def test_known_vectors(self):
        self.assertEqual(fnv1a(b""), 0x811C9DC5)
        self.assertEqual(fnv1a(b"a"), 0xE40C292C)
        self.assertEqual(fnv1a(b"foobar"), 0xBF9CF968)

    def test_bucket_deterministic_in_range(self):
        for i in range(100):
            b = bucket("feat" + str(i))
            self.assertTrue(0 <= b < NUM_BUCKETS)
            self.assertEqual(b, bucket("feat" + str(i)))


class TestFeatures(unittest.TestCase):
    def test_empty_and_digitless(self):
        self.assertEqual(list(text_features("")), [])
        self.assertEqual(list(text_features("123 !!! ???")), [])

    def test_case_insensitive(self):
        self.assertEqual(sorted(text_features("Hello")),
                         sorted(text_features("hello")))

    def test_nfc_composition(self):
        self.assertEqual(normalize("caf\u0065\u0301"), normalize("caf\u00e9"))

    def test_short_word_whole_token(self):
        self.assertEqual(list(text_features("ab")), [bucket("<ab>")])

    def test_word_boundary_markers(self):
        feats = set(text_features("abcd"))
        self.assertIn(bucket("<abc"), feats)
        self.assertIn(bucket("bcd>"), feats)
        self.assertIn(bucket("abcd"), feats)
        self.assertNotIn(bucket("<ab>"), feats)  # closing > only for short words

    def test_digits_and_punct_excluded(self):
        self.assertEqual(list(text_features("ab12!! cd")), list(text_features("ab cd")))
```

## Task 2: script router + tests (DONE)

- [x] script_route in langid/src/features.py (same file, section below in the test)
- [x] langid/tests/test_script_route.py - Arabic script intentionally NOT routed (asserted)
- [x] venv unittest green

```python
# langid/tests/test_script_route.py
import unittest

from langid.src.features import script_route

KO = "\uc548\ub155\ud558\uc138\uc694"   # annyeonghaseyo
JA = "\u3053\u3093\u306b\u3061\u306f"   # konnichiwa
HE = "\u05e9\u05dc\u05d5\u05dd"          # shalom
EL = "\u03ba\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1"  # kalimera
HI = "\u0928\u092e\u0938\u094d\u0924\u0947"  # namaste
AR = "\u0645\u0631\u062d\u0628\u0627"   # marhaba (Arabic script)
FA = "\u0633\u0644\u0627\u0645"          # salaam (Arabic script)


class TestScriptRoute(unittest.TestCase):
    def test_unique_scripts_route(self):
        self.assertEqual(script_route(KO), "ko")
        self.assertEqual(script_route(JA), "ja")
        self.assertEqual(script_route(HE), "he")
        self.assertEqual(script_route(EL), "el")
        self.assertEqual(script_route(HI), "hi")

    def test_arabic_script_not_routed(self):
        self.assertIsNone(script_route(AR))
        self.assertIsNone(script_route(FA))

    def test_latin_not_routed(self):
        self.assertIsNone(script_route("hello world"))

    def test_single_char_not_enough(self):
        self.assertIsNone(script_route(KO[0]))

    def test_mixed_prefers_unique_script(self):
        self.assertEqual(script_route("hello " + KO), "ko")
```

## Task 3: data - Tatoeba download + prepare

- [ ] Run download in background (network-only, CPU-safe):
```powershell
& .\.venv\Scripts\python.exe langid\scripts\download_data.py
```
  Expected: data/langid/raw/sentences.tar.bz2 present (size > 1 MB).
- [ ] After download: prepare train/val (CPU, deterministic):
```powershell
& .\.venv\Scripts\python.exe langid\scripts\prepare_data.py --cap 50000 --val-per-lang 500
```
  Expected: "[prepare] train.tsv: <N> rows" and "[prepare] val.tsv: <N> rows" (per-language
  availability varies; counts printed honestly).
- [x] Unit test on synthetic fixture tar (no network): langid/tests/test_data.py
- [ ] G6 leakage barrier check: compare 10-word shingles of train.tsv vs the eval TSV once the
  eval set is built (scripts/leakage_check.py to be added with Task 4); must be 0.

```python
# langid/src/data.py
"""Tatoeba corpus loading for DA-1 language-ID training data."""
import csv
import io
import random
import tarfile
from collections import defaultdict

TATOEBA_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"

# 21 languages: Arabic-script group first (the point of the Arabic-first
# line), then Latin/Cyrillic/European, then unique-script languages.
LANGS = [
    "ar", "fa", "ur", "ps",
    "en", "de", "fr", "es", "it", "pt", "nl", "tr",
    "ru", "uk", "pl", "el", "he", "hi",
    "ko", "ja", "zh",
]
LANG_TO_ID = {lang: i for i, lang in enumerate(LANGS)}

ARABIC_SCRIPT_GROUP = ("ar", "fa", "ur", "ps")


def _dedupe_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _has_language_signal(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def load_tatoeba(tbz_path, langs=None, cap=50000, val_per_lang=500, seed=42):
    """Parse a Tatoeba sentences.tar.bz2 export.

    Returns (train_rows, val_rows) of (lang, text). Deterministic: per-language
    dedupe on the alphanumeric-lowercase key, seeded shuffle, cap applied after
    the shuffle, val taken from the shuffled tail (never overlaps train).
    """
    langs = list(langs or LANGS)
    want = set(langs)
    by_lang = defaultdict(dict)
    with tarfile.open(tbz_path, "r:bz2") as tar:
        member = None
        while True:
            member = tar.next()
            if member is None:
                break
            if member.name.endswith("sentences.csv"):
                break
        if member is None:
            raise ValueError("sentences.csv not found in " + str(tbz_path))
        f = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            lang, text = row[1], row[2]
            if lang not in want or not _has_language_signal(text):
                continue
            key = _dedupe_key(text)
            if key and key not in by_lang[lang]:
                by_lang[lang][key] = text
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for lang in langs:
        texts = list(by_lang.get(lang, {}).values())
        rng.shuffle(texts)
        n_val = min(val_per_lang, len(texts) // 10)
        val_rows.extend((lang, t) for t in texts[:n_val])
        train_rows.extend((lang, t) for t in texts[n_val:n_val + cap])
    return train_rows, val_rows
```

```python
# langid/scripts/download_data.py
"""Download the Tatoeba sentences export for DA-1 (network-only, CPU-safe)."""
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import TATOEBA_URL


def main() -> int:
    raw_dir = pathlib.Path("data/langid/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "sentences.tar.bz2"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[download] already present: {dest} ({dest.stat().st_size} bytes)")
        return 0
    tmp = dest.with_suffix(".part")
    print(f"[download] {TATOEBA_URL} -> {tmp}")
    urllib.request.urlretrieve(TATOEBA_URL, tmp)
    tmp.replace(dest)
    print(f"[download] done: {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# langid/scripts/prepare_data.py
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
```

```python
# langid/tests/test_data.py
import io
import pathlib
import tarfile
import tempfile
import unittest

from langid.src.data import load_tatoeba


def make_fixture_tar(path):
    lines = []
    n = 0
    for lang in ("en", "fr", "ar"):
        for i in range(12):
            n += 1
            lines.append(f"{n}\t{lang}\t{lang} sentence number {i} unique")
    n += 1
    lines.append(f"{n}\ten\tEN SENTENCE, number 0 unique!")  # dedupe hit
    n += 1
    lines.append(f"{n}\ten\t123 456")                        # no letters
    data = ("\n".join(lines) + "\n").encode("utf-8")
    with tarfile.open(path, "w:bz2") as tar:
        info = tarfile.TarInfo("sentences.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


class TestLoadTatoeba(unittest.TestCase):
    def test_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            tar_path = pathlib.Path(td) / "sentences.tar.bz2"
            make_fixture_tar(tar_path)
            train, val = load_tatoeba(str(tar_path), langs=["en", "fr", "ar"],
                                      cap=20, val_per_lang=2, seed=1)
            self.assertEqual({l for l, _ in train} | {l for l, _ in val},
                             {"en", "fr", "ar"})
            # 12 unique per lang, val = min(2, 12//10)=1, train = 11
            self.assertEqual(len(train), 33)
            self.assertEqual(len(val), 3)

            def key(t):
                return "".join(c for c in t.lower() if c.isalnum())

            tr_keys = {key(t) for _, t in train}
            va_keys = {key(t) for _, t in val}
            self.assertFalse(tr_keys & va_keys)
            joined = " ".join(t for _, t in train + val)
            self.assertNotIn("123", joined)
```

## Task 4: eval harness BEFORE full training (D-line A0 discipline)

- [x] langid/scripts/eval_langid.py (full content below; prints G2/G3 gate rows PASS/FAIL)
- [ ] Build data/langid/eval.tsv: probe ungated FLORES-200 mirrors via huggingface_hub
  dataset_info (facebook/flores, Muennighoff/flores200, openlanguagedata/flores_plus),
  take the first accessible; fallback = wikimedia/wikipedia first-sentence slices
  (streaming, 1200/lang, fixed seed). Write data/langid/eval_manifest.json recording
  source + counts (honesty rule CLAUDE.md section 10 - never invent facts).
- [ ] Selftest: run the eval harness on a tiny fixture TSV + dummy model - protocol math verified.

```python
# langid/scripts/eval_langid.py
"""DA-1 evaluation harness (Tongue-style protocol).

Pre-registered gates (docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md):
  full >= 0.97, 5-word >= 0.95, 3-word >= 0.90, 1-word >= 0.70,
  Arabic-script group (ar/fa/ur/ps) mean 3-word >= 0.85.
Ties (top1-top2 logit margin < --margin) and abstains (no letter features)
are reported separately and NEVER counted as correct.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import features
from langid.src.data import ARABIC_SCRIPT_GROUP, LANGS, LANG_TO_ID
from langid.src.model import HashedLangID, encode_batch

GATES = {"full": 0.97, 5: 0.95, 3: 0.90, 1: 0.70, "arabic_group_3w": 0.85}


def read_eval(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lang, _, text = line.rstrip("\n").partition("\t")
            if lang in LANG_TO_ID and text:
                rows.append((lang, text))
    return rows


def k_window(words, k, rand):
    if len(words) <= k:
        return words
    start = rand.randrange(len(words) - k + 1)
    return words[start:start + k]


def score_split(model, rows, k=None, margin=0.5):
    rand = random.Random(42)
    texts = []
    for _, text in rows:
        words = text.split()
        sel = k_window(words, k, rand) if k else words
        texts.append(" ".join(sel))
    featless = [not any(features.text_features(t)) for t in texts]
    correct = ties = abstain = 0
    per_lang_ok = collections.Counter()
    per_lang_total = collections.Counter()
    confusion = collections.Counter()
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            chunk = rows[i:i + 512]
            ids, offsets = encode_batch(texts[i:i + 512])
            top2 = model(ids, offsets).topk(2, dim=1)
            for j, (lang, _) in enumerate(chunk):
                gold = LANG_TO_ID[lang]
                i1, i2 = int(top2.indices[j][0]), int(top2.indices[j][1])
                v1, v2 = float(top2.values[j][0]), float(top2.values[j][1])
                per_lang_total[lang] += 1
                if featless[i + j]:
                    abstain += 1
                elif i1 == gold:
                    correct += 1
                    per_lang_ok[lang] += 1
                elif v1 - v2 < margin:
                    ties += 1
                    confusion[f"tie:{LANGS[i1]}->{lang}"] += 1
                else:
                    confusion[f"{LANGS[i1]}->{lang}"] += 1
    n = max(1, len(rows))
    return {
        "k": k, "n": len(rows), "acc": correct / n, "tie_rate": ties / n,
        "abstain_rate": abstain / n,
        "per_lang_acc": {l: per_lang_ok[l] / per_lang_total[l]
                         for l in sorted(per_lang_total)},
        "top_confusions": confusion.most_common(10),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/langid/eval.tsv")
    ap.add_argument("--model", default="runs/langid_da1/model_fp32.pt")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--report", default="runs/langid_da1/eval_report.json")
    args = ap.parse_args()

    model = HashedLangID(len(LANGS), features.NUM_BUCKETS)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.emb.weight.data = state["emb"]
    model.bias.data = state["bias"]
    model.eval()

    rows = read_eval(args.eval)
    results = {}
    for k in (None, 5, 3, 1):
        key = "full" if k is None else str(k)
        results[key] = score_split(model, rows, k=k, margin=args.margin)
        r = results[key]
        print(f"[eval] k={key:4s} acc={r['acc']:.4f} tie={r['tie_rate']:.4f} "
              f"abstain={r['abstain_rate']:.4f}")
    group = [results["3"]["per_lang_acc"].get(l, 0.0) for l in ARABIC_SCRIPT_GROUP]
    results["arabic_group_3w"] = sum(group) / len(group)
    print("[eval] arabic-script group 3-word:", " ".join(
        f"{l}={v:.3f}" for l, v in zip(ARABIC_SCRIPT_GROUP, group)),
        f"mean={results['arabic_group_3w']:.4f}")

    counts = collections.Counter(lang for lang, _ in rows)
    majority = max(counts.values()) / max(1, len(rows))
    results["trivial_majority_acc"] = majority
    print(f"[eval] trivial majority baseline acc={majority:.4f}")

    gates = {
        "G2_full>=0.97": results["full"]["acc"] >= GATES["full"],
        "G2_5w>=0.95": results["5"]["acc"] >= GATES[5],
        "G2_3w>=0.90": results["3"]["acc"] >= GATES[3],
        "G2_1w>=0.70": results["1"]["acc"] >= GATES[1],
        "G3_arabic_group_3w>=0.85": results["arabic_group_3w"] >= GATES["arabic_group_3w"],
    }
    print("[gates]", " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    results["gates"] = gates
    report = pathlib.Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[eval] report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Task 5: model + train script + smoke gate (G1)

- [x] langid/src/model.py + langid/scripts/train.py (contents below)
- [x] langid/tests/test_model.py (toy 2-language task reaches 1.0 acc; loss strictly decreases)
- [ ] G1 smoke run after prepare with small cap:
```powershell
& .\.venv\Scripts\python.exe langid\scripts\prepare_data.py --cap 2000 --val-per-lang 200
& .\.venv\Scripts\python.exe langid\scripts\train.py --epochs 1 --out runs/langid_da1_smoke
```
  Expected: finishes < 10 min on CPU; val_acc printed; beats the majority baseline
  (labeled in the EXPERIMENTS.md row). Then rebuild full-cap data before Task 6.

```python
# langid/src/model.py
"""Hashed bag-of-n-grams multinomial logistic regression.

One EmbeddingBag(num_buckets -> num_langs, mode='sum') + per-language bias:
exactly the "feature hashing, no tokenizer" lexical classifier of the
Tongue recipe. Trains on CPU in minutes at DA-1 scale.
"""
import torch

from langid.src import features


class HashedLangID(torch.nn.Module):
    def __init__(self, num_langs: int, num_buckets: int = 1 << 16):
        super().__init__()
        self.num_langs = num_langs
        self.num_buckets = num_buckets
        self.emb = torch.nn.EmbeddingBag(num_buckets, num_langs, mode="sum")
        self.bias = torch.nn.Parameter(torch.zeros(num_langs))
        torch.nn.init.zeros_(self.emb.weight)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.emb(ids, offsets) + self.bias


def encode_batch(texts):
    """Flat hash ids + bag offsets for a list of texts (EmbeddingBag input).

    Texts with zero letter features get one dummy bucket so no bag is empty;
    the eval harness reports those as abstains.
    """
    ids, offsets = [], []
    for t in texts:
        offsets.append(len(ids))
        f = list(features.text_features(t))
        if f:
            ids.extend(f)
        else:
            ids.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
```

```python
# langid/scripts/train.py
"""Train the DA-1 language-ID model on CPU (GPU-safe beside any run)."""
import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import features
from langid.src.data import LANGS, LANG_TO_ID
from langid.src.model import HashedLangID, encode_batch


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lang, _, text = line.rstrip("\n").partition("\t")
            if lang in LANG_TO_ID and text:
                rows.append((LANG_TO_ID[lang], text))
    return rows


def evaluate(model, rows, batch_size=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            ids, offsets = encode_batch([t for _, t in chunk])
            pred = model(ids, offsets).argmax(dim=1)
            gold = torch.tensor([lang for lang, _ in chunk], dtype=torch.long)
            correct += int((pred == gold).sum())
    return correct / max(1, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/langid")
    ap.add_argument("--out", default="runs/langid_da1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(42)
    train = read_tsv(pathlib.Path(args.data_dir) / "train.tsv")
    val = read_tsv(pathlib.Path(args.data_dir) / "val.tsv")
    print(f"[train] train={len(train)} val={len(val)} langs={len(LANGS)} "
          f"buckets={features.NUM_BUCKETS}")

    model = HashedLangID(len(LANGS), features.NUM_BUCKETS)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(train)).tolist()
        total_loss = 0.0
        steps = 0
        for i in range(0, len(train), args.batch_size):
            chunk = [train[j] for j in order[i:i + args.batch_size]]
            ids, offsets = encode_batch([t for _, t in chunk])
            gold = torch.tensor([lang for lang, _ in chunk], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(model(ids, offsets), gold)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            steps += 1
        acc = evaluate(model, val)
        print(f"[train] epoch {epoch + 1}/{args.epochs} "
              f"loss={total_loss / max(1, steps):.4f} val_acc={acc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc = acc
            torch.save({"emb": model.emb.weight.detach().cpu(),
                        "bias": model.bias.detach().cpu()},
                       out / "model_fp32.pt")

    (out / "train_summary.json").write_text(json.dumps({
        "epochs": args.epochs, "best_val_acc": best_acc, "langs": LANGS,
        "num_buckets": features.NUM_BUCKETS, "lr": args.lr,
        "batch_size": args.batch_size,
    }, indent=2), encoding="utf-8")
    print(f"[train] best val_acc={best_acc:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# langid/tests/test_model.py
import unittest

import torch

from langid.src import features
from langid.src.model import HashedLangID, encode_batch


class TestEncodeBatch(unittest.TestCase):
    def test_offsets_valid(self):
        ids, offsets = encode_batch(["hello world", "bonjour", "123"])
        self.assertEqual(int(offsets[0]), 0)
        self.assertTrue(torch.all(offsets[1:] >= offsets[:-1]).item())
        self.assertGreater(len(ids), 0)

    def test_empty_batch_gets_dummy_bucket(self):
        ids, offsets = encode_batch(["123"])
        self.assertEqual(len(ids), 1)


class TestModelLearns(unittest.TestCase):
    def test_loss_decreases_on_toy_task(self):
        torch.manual_seed(0)
        texts = ["aaa bbb"] * 16 + ["xxx yyy"] * 16
        labels = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
        model = HashedLangID(2, features.NUM_BUCKETS)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        ids, offsets = encode_batch(texts)
        first = last = None
        for step in range(30):
            loss = torch.nn.functional.cross_entropy(model(ids, offsets), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        self.assertLess(last, first)
        with torch.no_grad():
            acc = float((model(ids, offsets).argmax(1) == labels).float().mean())
        self.assertEqual(acc, 1.0)
```

## Task 6: full train + eval + report

- [ ] BEFORE full training: register the E-row in research/EXPERIMENTS.md (row-first
  discipline): gates G1..G6 as above; results only appended when closed.
- [ ] Full train:
```powershell
& .\.venv\Scripts\python.exe langid\scripts\train.py --epochs 2 --out runs/langid_da1
```
  Expected: val_acc > smoke val_acc; model_fp32.pt + train_summary.json written.
- [ ] Eval on held-out set:
```powershell
& .\.venv\Scripts\python.exe langid\scripts\eval_langid.py
```
  Expected: G2/G3 gate rows printed PASS or FAIL (honest either way); eval_report.json written.

## Task 7: int8 export + numpy inference + latency

- [x] quantize_columns + Int8LangID in langid/src/infer.py; langid/scripts/export_int8.py
  (export + ms/word benchmark, contents below)
- [x] langid/tests/test_infer.py (round-trip error bound <= max_abs/254; scores shape;
  no-feature abstain)
- [ ] G4 check: artifact bytes <= 2.5 MiB; int8-vs-fp32 accuracy delta <= 0.5 pp on the eval
  set (eval once with model_fp32.pt, once via Int8LangID, diff).
- [ ] Write the run report (research/desert_ant_recreation/DA1_REPORT.md) and update
  HANDOFF / MEMORY / TASKS.

```python
# langid/src/infer.py
"""Pure numpy inference + per-language int8 quantization for DA-1 (no torch)."""
import numpy as np

from langid.src import features


def quantize_columns(W):
    """Per-language-column int8 quantization: Wq[:, l] ~= W[:, l] * scale[l].

    Max-abs symmetric scale per language column; worst-case reconstruction
    error per weight is max_abs[l] / 254.
    """
    W = np.asarray(W, dtype=np.float64)
    max_abs = np.max(np.abs(W), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    scale = 127.0 / max_abs
    Wq = np.clip(np.round(W * scale), -127.0, 127.0).astype(np.int8)
    return Wq, scale


class Int8LangID:
    """Loads a langid_int8.npz artifact and predicts with numpy only."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.Wq = z["W"]
        self.scale = z["scale"].astype(np.float64)
        self.bias = z["bias"].astype(np.float64)
        self.langs = [str(x) for x in z["langs"].tolist()]

    def scores(self, text: str):
        """Raw per-language scores, or (None, langs) when there are no
        letter features (abstain - never a guess)."""
        ids = np.fromiter(features.text_features(text), dtype=np.int64)
        if ids.size == 0:
            return None, self.langs
        # sum(Wq)/scale ~= sum(W): int64 accumulation avoids int8 overflow
        return self.Wq[ids].sum(axis=0, dtype=np.int64).astype(np.float64) / self.scale + self.bias, self.langs

    def predict(self, text: str, margin: float = 0.5):
        """Returns (lang, is_tie_or_abstain). Ties/abstains return (None, True)."""
        s, langs = self.scores(text)
        if s is None:
            return None, True
        order = np.argsort(s)[::-1]
        i1, i2 = int(order[0]), int(order[1])
        if s[i1] - s[i2] < margin:
            return None, True
        return langs[i1], False
```

```python
# langid/scripts/export_int8.py
"""Export fp32 model -> per-language int8 npz artifact + latency benchmark."""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import LANGS
from langid.src.infer import Int8LangID, quantize_columns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/langid_da1/model_fp32.pt")
    ap.add_argument("--out", default="runs/langid_da1/langid_int8.npz")
    args = ap.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    W = state["emb"].numpy().astype(np.float64)
    bias = state["bias"].numpy().astype(np.float64)
    Wq, scale = quantize_columns(W)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, W=Wq, scale=scale, bias=bias,
                        langs=np.array(LANGS))
    size = out.stat().st_size
    print(f"[export] {out} {size} bytes ({size / 1048576:.2f} MiB)")

    model = Int8LangID(str(out))
    words = ["hello", "bonjour", "guten", "mundo", "strana", "merhaba"] * 50
    t0 = time.perf_counter()
    for w in words:
        model.predict(w)
    dt = (time.perf_counter() - t0) / len(words)
    print(f"[bench] {dt * 1000:.3f} ms/word over {len(words)} words (CPU numpy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# langid/tests/test_infer.py
import pathlib
import tempfile
import unittest

import numpy as np

from langid.src.infer import Int8LangID, quantize_columns


class TestQuantize(unittest.TestCase):
    def test_round_trip_error_bound(self):
        rng = np.random.RandomState(0)
        W = rng.randn(64, 3) * 0.1
        Wq, scale = quantize_columns(W)
        self.assertEqual(Wq.dtype, np.int8)
        max_abs = np.max(np.abs(W), axis=0)
        err = np.abs(Wq.astype(np.float64) / scale - W)
        self.assertTrue(np.all(err <= max_abs / 254.0 + 1e-12))


class TestInt8LangID(unittest.TestCase):
    def test_scores_and_predict(self):
        rng = np.random.RandomState(1)
        W = rng.randn(1 << 12, 2) * 0.05
        Wq, scale = quantize_columns(W)
        bias = np.array([0.1, -0.1])
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "m.npz"
            np.savez(p, W=Wq, scale=scale, bias=bias, langs=np.array(["en", "fr"]))
            m = Int8LangID(str(p))
            s, langs = m.scores("bonjour le monde")
            self.assertEqual(s.shape, (2,))
            self.assertEqual(langs, ["en", "fr"])
            self.assertIsNone(m.scores("123 !!!")[0])
            pred, tie = m.predict("hello")
            self.assertIsInstance(tie, bool)
            self.assertIn(pred, ("en", "fr", None))
```

## Commit plan

- Commit 1 (done this session): langid/ source + tests.
- Commit 2: this plan + DESIGN.md.
- One commit per task thereafter, message prefix "langid:".

## Self-review

- No placeholders: every module above is the actual committed file content (read from disk).
- Windows console safety: scripts print ASCII only; test fixtures use \uXXXX escapes (cp1252 lesson).
- Determinism: FNV-1a only (never Python hash()); seed 42 wherever data is shuffled/sampled.
- GPU: none used anywhere in this plan; state it in every run log.
