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
