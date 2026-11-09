import unittest

from langid.src.features import script_route

KO = "\uc548\ub155\ud558\uc138\uc694"   # annyeonghaseyo
JA = "\u3053\u3093\u306b\u3061\u306f"   # konnichiwa
HE = "\u05e9\u05dc\u05d5\u05dd"          # shalom
EL = "\u03ba\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1"  # kalimera
HI = "\u0928\u092e\u0938\u094d\u0924\u0947"  # namaste
AR = "\u0645\u0631\u062d\u0628\u0627"   # marhaba (Arabic script)
FA = "\u0633\u0644\u0627\u0645"          # salaam (Arabic script)


class TestScriptRoute(unittest.TestCase):
    def test_unique_scripts_route(self):
        self.assertEqual(script_route(KO), "ko")
        self.assertEqual(script_route(JA), "ja")
        self.assertEqual(script_route(HE), "he")
        self.assertEqual(script_route(EL), "el")
        self.assertEqual(script_route(HI), "hi")

    def test_arabic_script_not_routed(self):
        self.assertIsNone(script_route(AR))
        self.assertIsNone(script_route(FA))

    def test_latin_not_routed(self):
        self.assertIsNone(script_route("hello world"))

    def test_single_char_not_enough(self):
        self.assertIsNone(script_route(KO[0]))

    def test_mixed_prefers_unique_script(self):
        self.assertEqual(script_route("hello " + KO), "ko")
