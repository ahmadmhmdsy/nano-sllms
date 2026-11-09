"""M4.2: chat-parameter validation tests for webui/app.py.

The Gradio sliders bound what the UI can send, but the chat endpoint is
also reachable through the Gradio API with hand-built payloads, so every
parameter is validated again in app.validate_chat_params BEFORE any model
call. Out-of-range values are REJECTED with a clear error message, never
silently clamped: a clamped value would hide the user's mistake and run
a generation they did not ask for.

These tests are pure: no GPU, no checkpoint, no torch. The validation
function is called directly with boundary values, and chat_stream is
called only far enough to prove the validation fires before the model
check (no model is loaded, so a valid request still yields the
no-model placeholder).

Run from the repo root:
    .venv/Scripts/python.exe -W error::SyntaxWarning -m unittest discover -s webui/tests -t .
"""
import sys
import unittest
from pathlib import Path

WEBUI_DIR = Path(__file__).resolve().parents[1]
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))

import app as app_module  # noqa: E402  (webui/ put on sys.path above)

VALID = ("def fibonacci(n):", 128, 0.8, 0.95, 50)  # message, max_new, temp, top_p, top_k
_UNSET = object()  # sentinel: an argument defaulted is replaced by a valid one


class ValidateChatParamsTest(unittest.TestCase):
    """Boundary behaviour of the pure validation function."""

    def validate(self, message=_UNSET, max_new_tokens=_UNSET,
                 temperature=_UNSET, top_p=_UNSET, top_k=_UNSET):
        """Call validate_chat_params, defaulting every arg to a valid one."""
        args = [VALID[0], VALID[1], VALID[2], VALID[3], VALID[4]]
        for i, v in enumerate((message, max_new_tokens, temperature,
                               top_p, top_k)):
            if v is not _UNSET:
                args[i] = v
        return app_module.validate_chat_params(*args)

    # ---------------------------------------------------------- acceptance

    def test_valid_params_accepted(self):
        self.assertIsNone(app_module.validate_chat_params(*VALID))

    def test_max_new_tokens_boundaries_accepted(self):
        for ok in (1, 4096):
            with self.subTest(ok=ok):
                self.assertIsNone(self.validate(max_new_tokens=ok))

    def test_temperature_boundaries_accepted(self):
        for ok in (0.01, 2.0):
            with self.subTest(ok=ok):
                self.assertIsNone(self.validate(temperature=ok))

    def test_top_p_boundaries_accepted(self):
        for ok in (0.01, 1.0):
            with self.subTest(ok=ok):
                self.assertIsNone(self.validate(top_p=ok))

    def test_top_k_boundaries_accepted(self):
        for ok in (1, 200):
            with self.subTest(ok=ok):
                self.assertIsNone(self.validate(top_k=ok))

    # ---------------------------------------------------------- rejection

    def test_max_new_tokens_zero_is_rejected(self):
        error = self.validate(max_new_tokens=0)
        self.assertIsNotNone(error)
        self.assertIn("max_new_tokens", error)

    def test_max_new_tokens_10000_is_clamped(self):
        """Brief wording says "is clamped"; the operative reading is
        REJECTED - a silent clamp would hide the user's mistake (see the
        M4.2 report). The test name stays per the brief."""
        error = self.validate(max_new_tokens=10000)
        self.assertIsNotNone(error)
        self.assertIn("max_new_tokens", error)

    def test_temperature_zero_is_rejected(self):
        error = self.validate(temperature=0)
        self.assertIsNotNone(error)
        self.assertIn("temperature", error)

    def test_empty_message_is_rejected(self):
        error = self.validate(message="")
        self.assertIsNotNone(error)
        self.assertIn("message", error)

    def test_whitespace_message_is_rejected(self):
        error = self.validate(message="   ")
        self.assertIsNotNone(error)
        self.assertIn("message", error)

    def test_non_string_message_is_rejected(self):
        for bad in (None, 123, ["hi"]):
            with self.subTest(bad=bad):
                error = self.validate(message=bad)
                self.assertIsNotNone(error)
                self.assertIn("message", error)

    def test_top_p_out_of_range_rejected(self):
        for bad in (0, 0.001, 1.01, 2.0):
            with self.subTest(bad=bad):
                error = self.validate(top_p=bad)
                self.assertIsNotNone(error)
                self.assertIn("top_p", error)

    def test_top_k_out_of_range_rejected(self):
        for bad in (0, -5, 201, 10000):
            with self.subTest(bad=bad):
                error = self.validate(top_k=bad)
                self.assertIsNotNone(error)
                self.assertIn("top_k", error)

    def test_negative_max_new_tokens_rejected(self):
        error = self.validate(max_new_tokens=-1)
        self.assertIsNotNone(error)
        self.assertIn("max_new_tokens", error)

    def test_non_numeric_values_rejected(self):
        for bad in (None, "abc"):
            with self.subTest(bad=bad):
                error = self.validate(temperature=bad)
                self.assertIsNotNone(error)
                self.assertIn("temperature", error)

    # ------------------------------------------------------ error quality

    def test_error_message_states_the_allowed_range(self):
        error = self.validate(max_new_tokens=10000)
        self.assertIn("1", error)
        self.assertIn("4096", error)
        self.assertIn("10000", error)  # the offending value is echoed

    def test_multiple_errors_are_all_reported(self):
        error = self.validate(message="", max_new_tokens=0, temperature=0,
                              top_p=0, top_k=0)
        for field in ("message", "max_new_tokens", "temperature", "top_p",
                      "top_k"):
            self.assertIn(field, error)


class ChatStreamValidationTest(unittest.TestCase):
    """The validation is wired into chat_stream BEFORE any model call."""

    def test_chat_stream_rejects_invalid_params_before_model_call(self):
        """No model is loaded: had validation not fired first, the stream
        would have yielded the no-model placeholder instead."""
        pieces = list(app_module.chat_stream("hi", [], False, False, 0, 0.8,
                                             0.95, 50))
        self.assertEqual(len(pieces), 1)
        self.assertTrue(pieces[0].startswith("❌"), pieces[0])
        self.assertIn("max_new_tokens", pieces[0])

    def test_chat_stream_rejects_out_of_range_temperature(self):
        pieces = list(app_module.chat_stream("hi", [], False, False, 128, 0,
                                             0.95, 50))
        self.assertEqual(len(pieces), 1)
        self.assertTrue(pieces[0].startswith("❌"), pieces[0])
        self.assertIn("temperature", pieces[0])

    def test_chat_stream_blank_message_keeps_placeholder(self):
        """A blank box keeps its long-standing placeholder (pinned by the
        M4.1 tests); validation covers every other bad input."""
        pieces = list(app_module.chat_stream("   ", [], False, False, 128,
                                             0.8, 0.95, 50))
        self.assertEqual(pieces,
                         ["_(load a checkpoint above, then send a code prefix)_"])

    def test_chat_stream_valid_params_pass_validation(self):
        """Valid parameters must not be over-rejected: with no model loaded
        the stream still reaches the no-model placeholder."""
        pieces = list(app_module.chat_stream("hi", [], False, False, 128, 0.8,
                                             0.95, 50))
        self.assertEqual(pieces,
                         ["_(load a checkpoint above, then send a code prefix)_"])

    def test_chat_stream_none_message_rejected_without_crash(self):
        """A hand-built API payload can pass a non-string message; the
        validation rejects it instead of crashing message.strip()."""
        pieces = list(app_module.chat_stream(None, [], False, False, 128, 0.8,
                                             0.95, 50))
        self.assertEqual(len(pieces), 1)
        self.assertTrue(pieces[0].startswith("❌"), pieces[0])
        self.assertIn("message", pieces[0])


if __name__ == "__main__":
    unittest.main()
