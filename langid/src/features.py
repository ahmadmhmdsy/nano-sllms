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
