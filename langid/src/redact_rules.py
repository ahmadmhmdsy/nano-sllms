"""DA-7b (E-55) deterministic Arabic redaction rules.

Token-level regex layer mirroring the hybrid design: numeric runs (Arabic and
Latin digits), dates (year / day-month patterns), phone numbers, URLs, emails,
and %-quantities. Returns the character span class per token.
"""
import re

AR_DIGITS = "\u0660-\u0669\u06f0-\u06f9"
D = "[" + AR_DIGITS + "0-9]"
M = "|".join(["\u064a\u0646\u0627\u064a\u0631", "\u0641\u0628\u0631\u0627\u064a\u0631", "\u0645\u0627\u0631\u0633", "\u0623\u0628\u0631\u064a\u0644", "\u0645\u0627\u064a\u0648", "\u064a\u0648\u0646\u064a\u0648", "\u064a\u0648\u0644\u064a\u0648", "\u0623\u063a\u0633\u0637\u0633", "\u0633\u0628\u062a\u0645\u0628\u0631", "\u0623\u0643\u062a\u0648\u0628\u0631", "\u0646\u0648\u0641\u0645\u0628\u0631", "\u062d\u062f\u064a\u0633\u0645\u0628\u0631"])
uk = "\u062f\u064a\u0633\u0645\u0628\u0631"
WD = "|".join(['\u0627\u0644\u0627\u062d\u062f','\u0627\u0644\u0627\u062b\u0646\u064a\u0646','\u0627\u0644\u062b\u0644\u0627\u062b\u0627\u0621','\u0627\u0644\u0627\u0631\u0628\u0639\u0627\u0621','\u0627\u0644\u062e\u0645\u064a\u0633','\u0627\u0644\u062c\u0645\u0639\u0629','\u0627\u0644\u0633\u0628\u062a','\u064a\u0648\u0645','\u0634\u0647\u0631','\u0639\u0627\u0645'])

PATTERNS = [
    ("URL", re.compile(r"https?://|www\.|\.com\b|\.net\b|\.org\b", re.I)),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")),
    ("PHONE", re.compile("^(" + D + "[\s\-.]?){7,}$")),
    ("ID", re.compile("^" + D + "{8,}$")),
    ("DATE", re.compile("^(" + D + "{1,2})?\\s*(" + M + "|" + uk + "|" + WD + "|" + D + "{4})$")),
    ("NUM", re.compile("^" + D + "+([.,:/]?\s*" + D + "+)?([%\u066a])?$")),
]


def match(tok):
    """Class name if the token matches a deterministic rule, else None."""
    t = tok.strip()
    if not t:
        return None
    for name, rx in PATTERNS:
        if rx.search(t):
            return name
    return None


REGEX_CLASSES = {"NUM", "DATE", "PHONE", "ID", "URL", "EMAIL"}

GOLD_MAP = {  # MAFAT gold tags covered by the deterministic layer
    "B-TIMEX": "DATE", "I-TIMEX": "DATE",
    "B-ANG": "NUM", "I-ANG": "NUM",
    "B-DUC": "NUM", "I-DUC": "NUM",
}
