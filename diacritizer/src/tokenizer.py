"""Char-level tokenizer (DESIGN.md section 3).

Deterministic vocabulary of Arabic base letters, Arabic-Indic + Western
digits, tatweel, ASCII printable run, and specials. Anything else (emoji,
CJK, rare script, passthrough marks) encodes as UNK: it is context-only
because passthrough.reconstruct() reinserts the original bytes, so UNK lossy
encoding is harmless BY CONSTRUCTION (selftest proves it).

Stdlib only.
"""
from __future__ import annotations

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def _build_vocab():
    chars = []
    def add(s):
        for ch in s:
            if ch not in chars:
                chars.append(ch)
    # Arabic base letters (primary block) + alef wasla, in codepoint order.
    add("".join(chr(o) for o in range(0x0621, 0x064B)))
    add(chr(0x0671))
    add(chr(0x0640))  # tatweel (context token; never diacritized)
    # Western digits + Arabic-Indic + Extended Arabic-Indic digits.
    add("0123456789")
    add("".join(chr(o) for o in range(0x0660, 0x066A)))
    add("".join(chr(o) for o in range(0x06F0, 0x06FA)))
    # Punctuation sentinels + common whitespace/newline.
    add(" \t\n.,!?;:\u060C\u061B\u061F()[]{}<>\"'\u2026\u00AB\u00BB-/\u0640=+*&@#%$~^|\\\u00A7")
    # ASCII printable remainder (covers most misc context chars).
    add("".join(chr(o) for o in range(0x21, 0x7F)))
    return chars

_BASE_CHARS = _build_vocab()
# Order: specials first (stable ids), then _BASE_CHARS, then nothing else.
ID_OF = {}
VOCAB = []
for ch in SPECIALS:
    ID_OF[ch] = len(VOCAB); VOCAB.append(ch)
for ch in _BASE_CHARS:
    if ch not in ID_OF:
        ID_OF[ch] = len(VOCAB); VOCAB.append(ch)

PAD, BOS, EOS, UNK = (ID_OF[s] for s in SPECIALS)
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE < 400, VOCAB_SIZE  # keep the char vocab small by design

def encode(text):
    """str -> list[int]; non-vocab chars become UNK (never fatal)."""
    return [ID_OF.get(ch, UNK) for ch in text]

def decode(ids):
    return "".join(VOCAB[i] for i in ids)
