"""Post-processing for D-line diacritizer outputs (E-18).

Two candidate transforms, each ADMITTED ONLY IF it improves the standing
gate DER (user rule 2026-09-16: "if worse or no valid result -> drop it"):

- dedup_marks: collapse runs of IDENTICAL marks; fold fatha+tanween-fatha
  pairs into the tanween. Safe by construction.
- legality_repair: the other agent risky "keep last haraka in a mixed run"
  suggestion (their regex steps 4+5). EXPECTED to break valid shadda+haraka
  stacks -> tested in E-18, dropped if worse.

Applied to PREDICTION files only, never refs. Marks: U+064B..652 + U+0670.
"""
import re

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "\u064e", "\u064f", "\u0650", "\u0652", "\u0651"
FATHATAN, DAMMATAN, KASRATAN = "\u064b", "\u064c", "\u064d"
HARAKAT = FATHA + DAMMA + KASRA + SUKUN + SHADDA + FATHATAN + DAMMATAN + KASRATAN


def dedup_marks(text):
    t = re.sub("([" + HARAKAT + "])\\1+", r"\1", text)
    for a, b in ((FATHA, FATHATAN), (DAMMA, DAMMATAN), (KASRA, KASRATAN)):
        t = re.sub(a + b, b, t)
    return t


def legality_repair(text):
    t = dedup_marks(text)
    t = re.sub("([" + FATHA + DAMMA + KASRA + "])([" + FATHA + DAMMA + KASRA + "])", r"\2", t)
    t = re.sub("([" + FATHA + DAMMA + KASRA + FATHATAN + DAMMATAN + KASRATAN + "])" + SHADDA, r"\1", t)
    return t
