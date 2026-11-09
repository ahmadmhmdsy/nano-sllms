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

# The Tatoeba exports switched to ISO 639-3 codes (and keep a couple of
# legacy ones: Arabic='ara', Mandarin='cmn'). Map our ISO-639-1 training
# tags to the codes actually present in sentences.csv (verified 2026-09-19
# by tallying the real export: e.g. 'arb'=0, 'ara'=68516).
TATOEBA_CODE = {
    "ar": "ara", "fa": "pes", "ur": "urd", "ps": "pus",
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "tr": "tur", "ru": "rus", "uk": "ukr",
    "pl": "pol", "el": "ell", "he": "heb", "hi": "hin",
    "ko": "kor", "ja": "jpn", "zh": "cmn",
}

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
    want = {TATOEBA_CODE[l]: l for l in langs}
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
            code, text = row[1], row[2]
            lang = want.get(code)
            if lang is None or not _has_language_signal(text):
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
