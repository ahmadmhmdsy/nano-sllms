import io
import pathlib
import tarfile
import tempfile
import unittest

from langid.src.data import load_tatoeba


def make_fixture_tar(path):
    lines = []
    n = 0
    # use the real Tatoeba ISO-639-3 export codes (data.TATOEBA_CODE)
    codes = {"en": "eng", "fr": "fra", "ar": "ara"}
    for lang, code in codes.items():
        for i in range(12):
            n += 1
            lines.append(f"{n}\t{code}\t{lang} sentence number {i} unique")
    n += 1
    lines.append(f"{n}\teng\tEN SENTENCE, number 0 unique!")  # dedupe hit
    n += 1
    lines.append(f"{n}\teng\t123 456")                        # no letters
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
