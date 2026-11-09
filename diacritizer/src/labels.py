"""15-class diacritic label schema (DESIGN.md sections 2/4).

Labels are per-Arabic-base-letter classes over the combining marks that
follow the letter in the source text. Non-classifiable marks (Quranic
annotation, superscript alef, hamza combinants...) are NOT labels: the
parser routes them to passthrough units (see passthrough.py).
Stdlib only, no dependencies.
"""

import unicodedata

# --- canonical class order (index == class id) -----------------------------
CLASS_NAMES = (
    "bare", "fatha", "damma", "kasra", "sukun",
    "fathatan", "dammatan", "kasratan",
    "shadda", "shadda_fatha", "shadda_damma", "shadda_kasra",
    "shadda_fathatan", "shadda_dammatan", "shadda_kasratan",
)
(C_BARE, C_FATHA, C_DAMMA, C_KASRA, C_SUKUN,
 C_FATHATAN, C_DAMMATAN, C_KASRATAN,
 C_SHADDA, C_SHADDA_FATHA, C_SHADDA_DAMMA, C_SHADDA_KASRA,
 C_SHADDA_FATHATAN, C_SHADDA_DAMMATAN, C_SHADDA_KASRATAN) = range(15)
N_CLASSES = 15

FATHA = "\u064e"; DAMMA = "\u064f"; KASRA = "\u0650"
FATHATAN = "\u064b"; DAMMATAN = "\u064c"; KASRATAN = "\u064d"
SUKUN = "\u0652"; SHADDA = "\u0651"

VOWELS = {FATHA: C_FATHA, DAMMA: C_DAMMA, KASRA: C_KASRA}
TANWIN = {FATHATAN: C_FATHATAN, DAMMATAN: C_DAMMATAN, KASRATAN: C_KASRATAN}

# Marks that can be consumed as labels (shadda is handled specially).
_CONSUMABLE = {FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN, SHADDA}

# Marks never consumed: U+0653..U+065F (hamza combinants etc.),
# Quranic annotation U+06D6..U+06ED, superscript alef U+0670, tatweel U+0640.
def is_passthrough_mark(ch):
    o = ord(ch)
    return (0x0653 <= o <= 0x065f) or o == 0x0670 or (0x06d6 <= o <= 0x06ed) or o == 0x0640

def label_for_marks(marks):
    """Map an ordered tuple of consumable marks to a class id.

    Valid forms: () / one vowel / one tanwin / sukun / shadda followed by
    exactly one of {vowel, tanwin}. Anything else raises ValueError
    (sample quarantine path).
    """
    if not marks:
        return C_BARE
    if len(marks) == 1:
        m = marks[0]
        if m in VOWELS: return VOWELS[m]
        if m in TANWIN: return TANWIN[m]
        if m == SUKUN: return C_SUKUN
        if m == SHADDA: return C_SHADDA
        raise ValueError("single non-vowel mark %r" % m)
    if len(marks) == 2:
        # Real corpora use BOTH orders: shadda+vowel (canonical) and
        # vowel+shadda (raw Tashkeela order). Both are the same gemination
        # class; decode is canonicalized to shadda-first by marks_for_label.
        for a, b in ((marks[0], marks[1]), (marks[1], marks[0])):
            if a == SHADDA and b in VOWELS:
                return C_SHADDA + VOWELS[b]                                        # 9..11
            if a == SHADDA and b in TANWIN:
                return C_SHADDA_FATHATAN + (TANWIN[b] - C_FATHATAN)                # 12..14
    raise ValueError("invalid mark sequence %r" % (marks,))

def marks_for_label(label):
    """Decode a class id back to its combining-mark string (detok/tests)."""
    if label == C_BARE: return ""
    if C_FATHA <= label <= C_SUKUN:
        return (FATHA, DAMMA, KASRA, SUKUN)[label - C_FATHA]
    if C_FATHATAN <= label <= C_KASRATAN:
        return (FATHATAN, DAMMATAN, KASRATAN)[label - C_FATHATAN]
    if label == C_SHADDA: return SHADDA
    if C_SHADDA_FATHA <= label <= C_SHADDA_KASRATAN:
        # NFC ORDER, NOT shadda-first. The harness NFC-normalizes every
        # reference (bench.load_pairs) while the decoder never normalized its
        # own output, so the two were compared raw. FatHa (ccc 30) sorts
        # before shadda (ccc 33), so a shadda-first render disagrees with
        # EVERY NFC reference on EVERY geminate: measured 4.5 pct of all
        # units and 25-43 pct of the raw error on all five gate corpora, and
        # 42 pct of the registered gate. NFC makes the output canonical text
        # and matches the reference.
        return unicodedata.normalize(
            "NFC",
            SHADDA + (FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN)[
                label - C_SHADDA_FATHA])
    raise ValueError("bad label %d" % label)
