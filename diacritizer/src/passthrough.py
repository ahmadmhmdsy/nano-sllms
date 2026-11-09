"""Segmentation + byte-exact reassembly (DESIGN.md section 4, decision D6).

parse(text) walks the string once and produces a list of Unit entries:
  - base units   : one Arabic base letter + its 15-class label (training target)
  - passthrough  : everything else (digits, latin, emoji, tatweel, Quranic
                   marks, invisibles, lone marks), context-visible, never a
                   target, and re-inserted byte-exact by reconstruct().

Failure surface: invalid mark combinations (e.g. fatha+fatha, shadda+sukun)
raise QuarantineError with the offending index -- the data pipeline (R51)
turns that into sample quarantine, never a silent guess.

Stdlib only.
"""
from labels import _CONSUMABLE, label_for_marks

# --- codepoint classes -------------------------------------------------------
def is_arabic_base(ch):
    o = ord(ch)
    if o == 0x0640:  # tatweel: passthrough by explicit decision, never a base
        return False
    return 0x0621 <= o <= 0x064A or o == 0x0671  # primary block + alef wasla

def classify_codepoint(ch):
    """Coarse category tag for reporting/edge-case bookkeeping."""
    o = ord(ch)
    if o == 0x0640: return "TATWEEL"
    if is_arabic_base(ch): return "ARABIC_BASE"
    if ch in _CONSUMABLE: return "DIACRITIC"
    if 0x0653 <= o <= 0x065f: return "ARABIC_COMBINANT"
    if o == 0x0670: return "SUPERSCRIPT_ALEF"
    if 0x06D6 <= o <= 0x06ED: return "QURANIC_MARK"
    if o in (0x200E, 0x200F, 0x200C, 0x200D, 0x2060, 0xFEFF):
        return "INVISIBLE"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFEFF:
        return "ARABIC_OTHER"
    return "OTHER"

class QuarantineError(ValueError):
    def __init__(self, index, reason):
        self.index = index
        super().__init__("quarantine at char %d: %s" % (index, reason))

class Unit:
    __slots__ = ("raw", "is_target", "label")
    def __init__(self, raw, is_target, label=None):
        self.raw = raw
        self.is_target = is_target
        self.label = label
    def __repr__(self):
        return "Unit(%r, target=%s, label=%s)" % (self.raw, self.is_target, self.label)

def parse(text):
    """text -> list[Unit]. Reconstruct(join) returns the input byte-exact."""
    units = []
    buf = []
    i = 0
    n = len(text)

    def flush():
        if buf:
            units.append(Unit("".join(buf), is_target=False))
            buf.clear()

    while i < n:
        ch = text[i]
        if is_arabic_base(ch):
            marks = []
            j = i + 1
            while j < n and text[j] in _CONSUMABLE:
                marks.append(text[j])
                j += 1
            try:
                label = label_for_marks(tuple(marks))
            except ValueError as e:
                raise QuarantineError(i, "after base %r: %s" % (ch, e)) from e
            flush()
            units.append(Unit(ch, is_target=True, label=label))
            i = j
        else:
            buf.append(ch)
            i += 1
    flush()
    return units

def reconstruct(units):
    return "".join(u.raw for u in units)

def labels(units):
    """Label per unit; None where the unit carries no prediction head."""
    return [u.label if u.is_target else None for u in units]
