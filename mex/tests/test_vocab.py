# mex/tests/test_vocab.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.vocab import CharVocab, char_ids, MAX_IDS, SPECIALS

def test_vocab_capped_and_deterministic():
    v = char_ids()
    assert len(v) <= 128
    assert [t for t in SPECIALS if t in v] == SPECIALS
    assert list(v.items())[:2] == [("<pad>", 0), ("<unk>", 1)]
    ids = sorted(v.values())
    assert ids == list(range(len(v))), "ids must be a dense 0..N-1 range"

def test_roundtrip_diacritic():
    voc = CharVocab()
    s = "مكتب|مَكْتَب"
    assert voc.decode(voc.encode(s)) == s

def test_unknown_char_maps_unk():
    voc = CharVocab()
    assert voc.encode("Ω")[-1] == voc.vocab["<unk>"]

def test_saved_tokenizer_loads(tmp_path):
    voc = CharVocab()
    voc.save(tmp_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tmp_path)
    assert tok.vocab_size <= 128
    ids = tok.encode("مكتب|مَكْتَب")
    assert tok.decode(ids) == "مكتب|مَكْتَب"


import unittest  # noqa: E402


class TestVocab(unittest.TestCase):
    """unittest harness over the module-level test functions (AGENTS.md §3
    prescribes unittest for mex tests; the functions stay callable as-is)."""

    def test_vocab_capped_and_deterministic(self):
        test_vocab_capped_and_deterministic()

    def test_roundtrip_diacritic(self):
        test_roundtrip_diacritic()

    def test_unknown_char_maps_unk(self):
        test_unknown_char_maps_unk()

    def test_saved_tokenizer_loads(self, tmp_path=None):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            test_saved_tokenizer_loads(Path(tmp))
