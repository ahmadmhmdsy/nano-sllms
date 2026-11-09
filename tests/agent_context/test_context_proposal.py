"""Tests for the gated, non-adopting context improvement proposal tool.

The tool under test must never edit live policy files (SOUL.md, AGENTS.md,
CLAUDE.md, rules/**), never stage adoption, and write only an explicitly
requested staged report and diff. Fixtures live in temporary directories so
the repository's own context files are never mutated across tests.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_context_proposal.py"

_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

GOOD_TASKS = '''tasks:
  - id: mock/lesson-smoke
    status: done
    evidence: []
experiments: []
'''

BAD_TASKS = '''tasks:
  - id: mock/lesson-smoke
    status: nonsense
    evidence: []
experiments: []
'''


def _make_context(root: Path, tasks_text: str, readme_text: str = "# Fixture\n") -> None:
    """Create a minimal policy-complete context tree under *root*."""
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "tasks.yaml").write_text(tasks_text, encoding="utf-8")
    (root / "README.md").write_text(readme_text, encoding="utf-8")
    for name in ("AGENTS.md", "CLAUDE.md", "SOUL.md"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    rules = root / "rules"
    rules.mkdir(exist_ok=True)
    (rules / "fixture.md").write_text("# Fixture rule\n", encoding="utf-8")
    # A policy-complete context tree also carries a valid, empty exception
    # registry; a missing registry is itself a controlled lint issue.
    (root / "state" / "context_exceptions.yaml").write_text(
        "schema_version: 1\nexceptions: []\n", encoding="utf-8"
    )


class ContextProposalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.baseline = base / "baseline"
        self.candidate = base / "candidate"
        self.staging = base / "staging"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.staging.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _report(self, **overrides):
        from validate_context_proposal import validate_proposal

        kwargs = {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "root": self.baseline,
        }
        kwargs.update(overrides)
        return validate_proposal(**kwargs)

    # -- classification ----------------------------------------------------

    def test_baseline_clean_candidate_clean_passes(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        report = self._report()
        self.assertTrue(report.baseline_ok)
        self.assertTrue(report.candidate_ok)
        self.assertEqual(report.regressions, ())
        self.assertEqual(report.improvements, ())

    def test_candidate_regression_is_reported_separately(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        report = self._report()
        self.assertTrue(report.baseline_ok)
        self.assertFalse(report.candidate_ok)
        self.assertTrue(report.regressions)
        self.assertEqual(report.improvements, ())
        self.assertIn("invalid task status", "\n".join(report.regressions))

    def test_candidate_fixing_baseline_issue_is_an_improvement(self) -> None:
        broken_readme = "# Fixture\nbroken_markdown_link: [x](missing-file.md)\n"
        _make_context(self.baseline, GOOD_TASKS, readme_text=broken_readme)
        _make_context(self.candidate, GOOD_TASKS, readme_text="# Fixture\n")
        report = self._report()
        self.assertFalse(report.baseline_ok)
        self.assertTrue(report.candidate_ok)
        self.assertEqual(report.regressions, ())
        self.assertTrue(report.improvements)
        self.assertIn("broken markdown link", "\n".join(report.improvements))

    def test_frozen_report(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        report = self._report()
        with self.assertRaises(Exception):
            report.regressions = ("t",)  # type: ignore[misc]

    # -- diff ---------------------------------------------------------------

    def test_render_diff_unified(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        from validate_context_proposal import render_diff

        diff = render_diff(self.baseline, self.candidate)
        self.assertIn("state/tasks.yaml", diff)
        self.assertIn("-    status: done", diff)
        self.assertIn("+    status: nonsense", diff)

    def test_render_diff_identical_is_empty(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        from validate_context_proposal import render_diff

        self.assertEqual(render_diff(self.baseline, self.candidate), "")

    def test_render_diff_not_vacuous_with_added_and_removed_files(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        (Path(self.baseline) / "dropped.md").write_text("goodbye\n", encoding="utf-8")
        (Path(self.candidate) / "added.md").write_text("hello\n", encoding="utf-8")
        from validate_context_proposal import render_diff

        diff = render_diff(self.baseline, self.candidate)
        self.assertIn("a/dropped.md", diff)
        self.assertIn("b/added.md", diff)
        self.assertIn("+++ b/added.md", diff)

    def test_render_diff_renders_subtree_relocation_as_delete_and_add(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        old_file = Path(self.baseline) / "old" / "x.txt"
        new_file = Path(self.candidate) / "new" / "x.txt"
        old_file.parent.mkdir(parents=True)
        new_file.parent.mkdir(parents=True)
        old_file.write_text("same\n", encoding="utf-8")
        new_file.write_text("same\n", encoding="utf-8")
        from validate_context_proposal import render_diff

        diff = render_diff(self.baseline, self.candidate)
        self.assertIn("--- a/old/x.txt", diff)
        self.assertIn("+++ b/old/x.txt", diff)
        self.assertIn("--- a/new/x.txt", diff)
        self.assertIn("+++ b/new/x.txt", diff)

    def test_render_diff_excludes_vcs_and_environment_directories(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        ignored = Path(self.baseline) / ".git" / "config"
        ignored.parent.mkdir(parents=True)
        ignored.write_text("secret-like repository metadata\n", encoding="utf-8")
        from validate_context_proposal import render_diff

        diff = render_diff(self.baseline, self.candidate)
        self.assertNotIn(".git/config", diff)
        self.assertNotIn("secret-like repository metadata", diff)

    def test_render_diff_excludes_sensitive_and_generated_stores(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        sentinel = "GENERATED_STORE_SENTINEL_7f3a"
        ignored_dirs = (
            "data",
            "runs",
            "outputs",
            "artifacts",
            "logs",
            "models",
            "raw",
            "scratch",
            "checkpoint_backup",
            "wandb",
            "resources",
        )
        for name in ignored_dirs:
            before = self.baseline / name / "private.txt"
            after = self.candidate / name / "private.txt"
            before.parent.mkdir(parents=True)
            after.parent.mkdir(parents=True)
            before.write_text(f"{sentinel}: before\n", encoding="utf-8")
            after.write_text(f"{sentinel}: after\n", encoding="utf-8")

        from validate_context_proposal import render_diff

        diff = render_diff(self.baseline, self.candidate)
        self.assertEqual(diff, "")
        self.assertNotIn(sentinel, diff)

    def test_documented_credential_placeholder_is_allowed(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        text = (
            "- Secret safety: values such as `password=...` are examples; "
            "live-looking values remain rejected.\n"
        )
        sample = self.candidate / "placeholder-example.md"
        sample.write_text(text, encoding="utf-8")

        diff = render_diff(self.baseline, self.candidate)
        self.assertIn("password=...", diff)
        safe = _safe_diagnostic(text)
        self.assertIn("password=...", safe)
        self.assertNotIn("<redacted:credential assignment>", safe)

    def test_render_diff_rejects_likely_secret_without_echoing_content(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        sentinel = "CREDENTIAL_SENTINEL_7f3a_do_not_echo"
        (self.candidate / "notes.txt").write_text(
            f"API_KEY={sentinel}\n",
            encoding="utf-8",
        )
        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("notes.txt", message)
        self.assertIn("secret", message.lower())
        self.assertNotIn(sentinel, message)

        report_path = self.staging / "secret-report.json"
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, _ = self._run_main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(report_path),
            ])
        self.assertEqual(code, 2)
        self.assertIn("notes.txt", stderr.getvalue())
        self.assertIn("secret", stderr.getvalue().lower())
        self.assertNotIn(sentinel, stderr.getvalue())
        self.assertFalse(report_path.exists())

    def test_render_diff_rejects_prefixed_credentials_and_provider_tokens(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        samples = (
            (
                "OPENAI_API_KEY",
                "OPENAI_API_KEY=prefixed-openai-credential-value-7f3a",
                "prefixed-openai-credential-value-7f3a",
            ),
            (
                "GITHUB_TOKEN",
                "GITHUB_TOKEN=prefixed-github-credential-value-7f3a",
                "prefixed-github-credential-value-7f3a",
            ),
            (
                "HF_TOKEN",
                "HF_TOKEN=prefixed-huggingface-credential-value-7f3a",
                "prefixed-huggingface-credential-value-7f3a",
            ),
            ("sk-proj-", "sk-proj-" + ("A" * 32), "sk-proj-" + ("A" * 32)),
            ("glpat-", "glpat-" + ("B" * 30), "glpat-" + ("B" * 30)),
            ("AIza", "AIza" + ("C" * 35), "AIza" + ("C" * 35)),
            ("hf_", "hf_" + ("D" * 30), "hf_" + ("D" * 30)),
        )
        from validate_context_proposal import render_diff

        for index, (label, text, value) in enumerate(samples):
            with self.subTest(label=label):
                sample = self.candidate / f"credential-{index}.txt"
                sample.write_text(f"{text}\n", encoding="utf-8")
                try:
                    with self.assertRaises(ValueError) as ctx:
                        render_diff(self.baseline, self.candidate)
                    message = str(ctx.exception)
                    self.assertIn(sample.name, message)
                    self.assertIn("secret", message.lower())
                    self.assertNotIn(value, message)
                finally:
                    sample.unlink(missing_ok=True)

    def test_render_diff_rejects_provider_token_split_by_allowed_controls(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("Z" * 24)
        candidate_file = self.candidate / "split-token.txt"
        candidate_file.write_text(
            f"before {token[:3]}\n{token[3:]} after\n",
            encoding="utf-8",
        )

        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn(candidate_file.name, message)
        self.assertIn("secret", message.lower())
        self.assertNotIn(token, message)

    def test_control_split_bearer_and_placeholder_prefix_values_are_redacted(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        secret = "SECRET_VALUE_123"
        for separator in (
            "\n", "\r", "\t", "\x00", "\u200b",
            "\u00a0", "\u2002", "\u2003", "\u2009", "\u202f",
        ):
            with self.subTest(separator=repr(separator)):
                safe = _safe_diagnostic(f"Bearer{separator}{secret}")
                self.assertIn("Bearer", safe)
                self.assertNotIn(secret, safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        for index, separator in enumerate(("\n", "\r", "\t")):
            with self.subTest(content_separator=repr(separator)):
                sample = self.candidate / f"split-bearer-{index}.txt"
                sample.write_text(
                    f"before Bearer{separator}{secret} after\n",
                    encoding="utf-8",
                )
                try:
                    with self.assertRaises(ValueError) as ctx:
                        render_diff(self.baseline, self.candidate)
                    self.assertIn(sample.name, str(ctx.exception))
                    self.assertNotIn(secret, str(ctx.exception))
                finally:
                    sample.unlink(missing_ok=True)

        for value in (
            "...", "${SECRET_VALUE}", "{{secret}}", "`...`", "'...'", '"..."',
        ):
            with self.subTest(placeholder=value):
                text = f"password={value}"
                self.assertEqual(_safe_diagnostic(text), text)

        for index, value in enumerate(("...SECRET_VALUE_123", "...secret")):
            with self.subTest(live_value=value):
                text = f"password={value}"
                safe = _safe_diagnostic(text)
                self.assertIn("<redacted:credential assignment>", safe)
                self.assertNotIn(value, safe)
                sample = self.candidate / f"prefixed-placeholder-{index}.txt"
                sample.write_text(f"{text}\n", encoding="utf-8")
                try:
                    with self.assertRaises(ValueError) as ctx:
                        render_diff(self.baseline, self.candidate)
                    self.assertIn(sample.name, str(ctx.exception))
                    self.assertNotIn(value, str(ctx.exception))
                finally:
                    sample.unlink(missing_ok=True)

    def test_unicode_whitespace_credentials_are_redacted_in_diagnostics_and_paths(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        secret = "SECRET_VALUE_123"
        separators = ("\u00a0", "\u2002", "\u2003", "\u2009", "\u202f")
        for separator in separators:
            with self.subTest(diagnostic_separator=repr(separator)):
                text = f"password{separator}={separator}{secret}"
                safe = _safe_diagnostic(text)
                self.assertNotIn(secret, safe)
                self.assertIn("password", safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        for index, separator in enumerate(separators):
            with self.subTest(content_separator=repr(separator)):
                sample = self.candidate / f"unicode-content-{index}.txt"
                sample.write_text(
                    f"password{separator}={separator}{secret}\n",
                    encoding="utf-8",
                )
                try:
                    with self.assertRaises(ValueError) as ctx:
                        render_diff(self.baseline, self.candidate)
                    self.assertNotIn(secret, str(ctx.exception))
                finally:
                    sample.unlink(missing_ok=True)

            with self.subTest(path_separator=repr(separator)):
                sample = self.candidate / f"password{separator}={secret}.txt"
                sample.write_text("ordinary\n", encoding="utf-8")
                try:
                    with self.assertRaises(ValueError) as ctx:
                        render_diff(self.baseline, self.candidate)
                    self.assertNotIn(secret, str(ctx.exception))
                finally:
                    sample.unlink(missing_ok=True)

    def test_direct_render_diff_wraps_unexpected_reader_errors(self) -> None:
        from validate_context_proposal import render_diff

        token = "github_pat_" + ("Q" * 30)
        with patch(
            "validate_context_proposal._read_tree",
            side_effect=RuntimeError(f"raw reader failed for {token}"),
        ), self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)

        message = str(ctx.exception)
        self.assertIn("diff", message.lower())
        self.assertNotIn(token, message)
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_direct_validate_proposal_wraps_pathlike_conversion_errors(self) -> None:
        from validate_context_proposal import validate_proposal

        token = "github_pat_" + ("P" * 30)

        class ExplodingPath:
            def __fspath__(self):
                raise RuntimeError(f"raw path conversion failed for {token}")

        with self.assertRaises(ValueError) as ctx:
            validate_proposal(ExplodingPath(), self.candidate, self.baseline)

        message = str(ctx.exception)
        self.assertIn("baseline", message)
        self.assertNotIn(token, message)
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_render_diff_wraps_malformed_public_paths(self) -> None:
        from validate_context_proposal import render_diff

        cases = ((None, self.baseline, "baseline"), (self.baseline, None, "candidate"))
        for baseline, candidate, role in cases:
            with self.subTest(role=role):
                with self.assertRaises(ValueError) as ctx:
                    render_diff(baseline, candidate)
                message = str(ctx.exception)
                self.assertIn(role, message)
                self.assertNotIn("Traceback", message)
                self.assertNotIn("NoneType", message)
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    def test_diagnostic_redacts_spaced_credential_values(self) -> None:
        from validate_context_proposal import _safe_diagnostic

        samples = (
            ("Authorization: Bearer SUPER_SECRET_VALUE_123", "Authorization", "SUPER_SECRET_VALUE_123"),
            ("token: abc def", "token", "abc def"),
        )
        for text, role, secret in samples:
            with self.subTest(role=role):
                safe = _safe_diagnostic(text)
                self.assertIn(role, safe)
                self.assertIn("<redacted:credential assignment>", safe)
                self.assertNotIn(secret, safe)

    def test_diagnostic_redacts_standalone_bearer_and_jwt_values(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        bearer = "SUPER_SECRET_BEARER_123"
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjMifQ."
            "SUPER_SIGNATURE_123"
        )
        jwe = (
            "eyJhbGciOiJka2IifQ."
            "QUJDREVGR0."
            "QUJDREVG0."
            "QUJDREVGU."
            "QUJDREVGVA"
        )
        for text, label, secret in (
            (f"Bearer {bearer}", "Bearer", bearer),
            (f"Bearer:{bearer}", "Bearer", bearer),
            (f"Bearer={bearer}", "Bearer", bearer),
            (f"authentication header contained {jwt}", "authentication", jwt),
            (f"authentication header contained {jwe}", "authentication", jwe),
        ):
            with self.subTest(label=label):
                safe = _safe_diagnostic(text)
                self.assertIn(label, safe)
                self.assertIn("<redacted:", safe)
                self.assertNotIn(secret, safe)
                if secret is jwe:
                    self.assertNotIn(".".join(jwe.split(".")[3:]), safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        for index, secret in enumerate((jwt, jwe)):
            sample = self.candidate / f"standalone-jwt-{index}.txt"
            sample.write_text(f"payload {secret}\n", encoding="utf-8")
            try:
                with self.assertRaises(ValueError) as ctx:
                    render_diff(self.baseline, self.candidate)
                self.assertIn(sample.name, str(ctx.exception))
                self.assertNotIn(secret, str(ctx.exception))
            finally:
                sample.unlink(missing_ok=True)

    def test_diagnostic_redacts_json_credentials_and_short_or_unsigned_tokens(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        credential = "GENERIC_CREDENTIAL_VALUE_123"
        short_bearer = "short-secret"
        unsigned_jwt = "eyJhbGciOiJub25lIn0.."
        for text, label, secret in (
            (f'{{"token": "{credential}"}}', "token", credential),
            (f"Bearer {short_bearer}", "Bearer", short_bearer),
            (f"authentication header contained {unsigned_jwt}", "authentication", unsigned_jwt),
        ):
            with self.subTest(label=label):
                safe = _safe_diagnostic(text)
                self.assertIn(label, safe)
                self.assertIn("<redacted:", safe)
                self.assertNotIn(secret, safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        for index, (text, secret) in enumerate((
            (f'{{"token": "{credential}"}}', credential),
            (f"Bearer {short_bearer}", short_bearer),
            (f"payload {unsigned_jwt}", unsigned_jwt),
        )):
            sample = self.candidate / f"credential-case-{index}.txt"
            sample.write_text(text + "\n", encoding="utf-8")
            try:
                with self.assertRaises(ValueError) as ctx:
                    render_diff(self.baseline, self.candidate)
                self.assertIn(sample.name, str(ctx.exception))
                self.assertNotIn(secret, str(ctx.exception))
            finally:
                sample.unlink(missing_ok=True)

    def test_diagnostic_redacts_common_camel_case_credential_keys(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        samples = (
            ("refreshToken=GENERIC_REFRESH_VALUE_123", "refreshToken"),
            ("accessToken: GENERIC_ACCESS_VALUE_123", "accessToken"),
            ("credentials = {\"session\": \"GENERIC_CREDENTIALS_VALUE_123\"}", "credentials"),
            ("passphrase: GENERIC_PASSPHRASE_VALUE_123", "passphrase"),
            ("apiToken=GENERIC_API_TOKEN_VALUE_123", "apiToken"),
            ("bearerToken=GENERIC_BEARER_TOKEN_VALUE_123", "bearerToken"),
            ("_authToken=GENERIC_AUTH_TOKEN_VALUE_123", "_authToken"),
        )
        for text, label in samples:
            with self.subTest(label=label):
                safe = _safe_diagnostic(text)
                self.assertIn(label, safe)
                self.assertIn("<redacted:", safe)
                self.assertNotIn("GENERIC_", safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        for index, (text, label) in enumerate(samples):
            sample = self.candidate / f"credential-key-{index}.txt"
            sample.write_text(text + "\n", encoding="utf-8")
            try:
                with self.assertRaises(ValueError) as ctx:
                    render_diff(self.baseline, self.candidate)
                self.assertIn(sample.name, str(ctx.exception))
                self.assertNotIn("GENERIC_", str(ctx.exception))
            finally:
                sample.unlink(missing_ok=True)

    def test_diagnostic_redacts_complete_private_key_blocks(self) -> None:
        from validate_context_proposal import _safe_diagnostic

        body = "MII_PRIVATE_BODY_SENTINEL_7f3a"
        text = (
            "diagnostic before\n"
            "-----BEGIN PRIVATE KEY-----\n"
            f"{body}\n"
            "-----END PRIVATE KEY-----\n"
            "diagnostic after"
        )
        safe = _safe_diagnostic(text)
        self.assertIn("diagnostic before", safe)
        self.assertIn("<redacted:private key block>", safe)
        self.assertNotIn(body, safe)
        self.assertNotIn("BEGIN PRIVATE KEY", safe)
        self.assertNotIn("END PRIVATE KEY", safe)

    def test_diagnostic_and_diff_reject_unicode_tag_controls(self) -> None:
        from validate_context_proposal import _safe_diagnostic, render_diff

        tag = "\U000e0000"
        token = "sk-" + ("J" * 24)
        safe = _safe_diagnostic(
            f"diagnostic {token[:3]}{tag}{token[3:]}tail"
        )
        self.assertNotIn(tag, safe)
        self.assertNotIn(token, safe)

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        content = f"ordinary{tag}content{tag}\n"
        (self.candidate / "tagged.txt").write_text(content, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("tagged.txt", message)
        self.assertNotIn(tag, message)
        self.assertNotIn("ordinary", message)

    def test_render_diff_rejects_unicode_format_controls_without_echoing_content(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        marker = "FORMAT_SENTINEL_7f3a"
        format_control = "\u202e"
        candidate_file = self.candidate / "format.txt"
        candidate_file.write_text(
            f"{marker}{format_control}ordinary text\n",
            encoding="utf-8",
        )

        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("format.txt", message)
        self.assertIn("control", message.lower())
        self.assertNotIn(format_control, message)
        self.assertNotIn(marker, message)

    def test_render_diff_rejects_control_characters_without_echoing_content(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        sentinel = "CONTROL_SENTINEL_7f3a"
        (self.candidate / "control.txt").write_text(
            f"{sentinel}\x00payload",
            encoding="utf-8",
        )
        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("control.txt", message)
        self.assertIn("control", message.lower())
        self.assertNotIn(sentinel, message)

    def test_render_diff_rejects_oversized_file_before_output(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        oversized = self.candidate / "large.txt"
        oversized.write_bytes(b"x" * ((1 << 20) + 1))
        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("large.txt", message)
        self.assertIn("size", message.lower())
        self.assertNotIn(str(self.candidate), message)

    def test_control_characters_in_untrusted_filenames_are_rejected(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        try:
            unsafe = self.candidate / ("name\x1f.txt")
            unsafe.write_text("payload\n", encoding="utf-8")
        except OSError as exc:
            self.skipTest(f"control-character filename unavailable: {exc}")
        from validate_context_proposal import render_diff
        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("control", message.lower())
        self.assertNotIn("name\x1f.txt", message)

    def test_secret_like_untrusted_filename_is_rejected_without_echo(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("A" * 24)
        unsafe = self.candidate / f"{token}.txt"
        unsafe.write_text("safe payload\n", encoding="utf-8")

        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        message = str(ctx.exception)
        self.assertIn("secret", message.lower())
        self.assertNotIn(token, message)
        self.assertNotIn(str(self.candidate), message)

        report_path = self.staging / "secret-filename-report.json"
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, _ = self._run_main(["--output-report", str(report_path)])
        self.assertEqual(code, 2)
        self.assertIn("secret", stderr.getvalue().lower())
        self.assertNotIn(token, stderr.getvalue())
        self.assertNotIn(str(self.candidate), stderr.getvalue())
        self.assertFalse(report_path.exists())

    def test_preflight_tree_diagnostics_redact_absolute_paths(self) -> None:
        from validate_context_proposal import _preflight_tree

        missing = Path(self._tmp.name) / "private-root-marker" / "missing"
        with self.assertRaises(ValueError) as ctx:
            _preflight_tree(missing, "candidate")
        message = str(ctx.exception)
        self.assertIn("missing candidate root", message)
        self.assertNotIn(str(missing), message)
        self.assertNotIn("private-root-marker", message)
        self.assertNotIn("Traceback", message)

    def test_input_path_load_errors_redact_secret_values_and_keep_role(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("A" * 24)
        original_abspath = os.path.abspath

        for role in ("baseline", "candidate"):
            target = getattr(self, role)
            target_name = os.path.normcase(os.fspath(target))

            def fail_for_role(path):
                if os.path.normcase(os.fspath(path)) == target_name:
                    raise OSError(
                        f"cannot load {role} path {token}.yaml: metadata denied"
                    )
                return original_abspath(path)

            with self.subTest(role=role), patch(
                "validate_context_proposal.os.path.abspath",
                side_effect=fail_for_role,
            ):
                with self.assertRaises((OSError, ValueError)) as ctx:
                    self._report()
            message = str(ctx.exception)
            self.assertIn(role, message)
            self.assertIn("metadata denied", message)
            self.assertNotIn(token, message)

    def test_input_path_role_label_is_bounded_and_redacted(self) -> None:
        from validate_context_proposal import _absolute_input_path

        token = "sk-" + ("G" * 24)
        label = f"role-{token}; " + ("l" * 700)
        error_text = f"metadata denied for {token}; " + ("e" * 700)
        with patch(
            "validate_context_proposal.os.path.abspath",
            side_effect=OSError(error_text),
        ), self.assertRaises(ValueError) as ctx:
            _absolute_input_path(self.staging, label)

        message = str(ctx.exception)
        self.assertIn("role-", message)
        self.assertIn("metadata denied", message)
        self.assertNotIn(token, message)
        self.assertLessEqual(len(message), 512)

    def test_linter_error_diagnostic_is_globally_bounded(self) -> None:
        from validate_context_proposal import _lint_issues

        class FailingLinter:
            def lint_tree(self, root, *, include_policy, evidence_root):
                raise OSError("metadata " + ("x" * 1200))

        with patch(
            "validate_context_proposal._load_linter",
            return_value=FailingLinter(),
        ), self.assertRaises(ValueError) as ctx:
            _lint_issues(self.baseline, self.baseline)

        message = str(ctx.exception)
        self.assertIn("static check", message)
        self.assertLessEqual(len(message), 512)
        self.assertNotIn("x" * 600, message)

    def test_read_tree_error_diagnostic_is_globally_bounded(self) -> None:
        from validate_context_proposal import _read_tree

        long_directory = self.candidate
        try:
            for index in range(8):
                long_directory = long_directory / (f"segment-{index}-" + ("d" * 70))
                long_directory.mkdir()
        except OSError as exc:
            self.skipTest(f"long diagnostic path unavailable: {exc}")

        original_scandir = os.scandir

        def fail_for_long_directory(path):
            if Path(path) == long_directory:
                raise OSError("directory metadata " + ("e" * 1200))
            return original_scandir(path)

        with patch(
            "validate_context_proposal._preflight_tree",
            return_value=None,
        ), patch(
            "validate_context_proposal.os.scandir",
            side_effect=fail_for_long_directory,
        ), self.assertRaises(ValueError) as ctx:
            _read_tree(self.candidate)

        message = str(ctx.exception)
        self.assertIn("directory metadata", message)
        self.assertLessEqual(len(message), 512)
        self.assertNotIn("e" * 600, message)

    def test_wrapped_input_error_traceback_does_not_echo_secret(self) -> None:
        import traceback

        from validate_context_proposal import _absolute_input_path

        token = "sk-" + ("H" * 24)
        cause = None
        suppress_context = False
        with patch(
            "validate_context_proposal.os.path.abspath",
            side_effect=OSError(f"metadata denied for {token}"),
        ):
            try:
                _absolute_input_path(self.staging, "candidate")
            except ValueError as exc:
                rendered = "".join(traceback.format_exception(exc))
                cause = exc.__cause__
                suppress_context = exc.__suppress_context__

        self.assertIn("candidate", rendered)
        self.assertNotIn(token, rendered)
        self.assertIsNone(cause)
        self.assertTrue(suppress_context)

    def test_staging_error_message_is_bounded_at_construction(self) -> None:
        from validate_context_proposal import StagingError

        error = StagingError("prefix " + ("x" * 1200))
        self.assertLessEqual(len(error.args[0]), 512)
        self.assertLessEqual(len(str(error)), 512)

    def test_linter_diagnostics_redact_secret_shaped_filenames(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("B" * 24)

        def lint_with_secret_path(root, _evidence_root):
            if Path(root) == self.candidate:
                return [
                    f"state/{token}.yaml: invalid task status\x1f\u202e"
                    + ("x" * 600)
                ]
            return []

        with patch(
            "validate_context_proposal._lint_issues",
            side_effect=lint_with_secret_path,
        ):
            report = self._report()

        diagnostic = "\n".join(report.regressions)
        self.assertIn("state/", diagnostic)
        self.assertIn("invalid task status", diagnostic)
        self.assertNotIn(token, diagnostic)
        self.assertNotIn("\x1f", diagnostic)
        self.assertNotIn("\u202e", diagnostic)
        self.assertLess(len(diagnostic), 600)
        self.assertFalse(report.candidate_ok)

    def test_candidate_ok_precedes_bounded_diagnostic_redaction(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        shared = "ordinary static diagnostic " + ("x" * 600)

        def lint_with_distinct_long_issues(root, _evidence_root):
            suffix = "baseline" if Path(root) == self.baseline else "candidate"
            return [shared + suffix]

        with patch(
            "validate_context_proposal._lint_issues",
            side_effect=lint_with_distinct_long_issues,
        ):
            report = self._report()

        self.assertTrue(report.regressions)
        self.assertFalse(report.candidate_ok)
        self.assertLess(len(report.regressions[0]), 600)

    def test_static_check_metadata_and_utf8_errors_are_redacted(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("C" * 24)
        errors = (
            OSError(f"candidate metadata failed for state/{token}.yaml"),
            UnicodeError(f"candidate state/{token}.yaml is not valid UTF-8"),
        )

        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                stderr = io.StringIO()
                with patch(
                    "validate_context_proposal._lint_issues",
                    side_effect=error,
                ), redirect_stderr_to(stderr):
                    code, stdout = self._run_main([])
                message = stderr.getvalue()
                self.assertEqual(code, 2)
                self.assertIn("candidate", message)
                self.assertIn("state/", message)
                self.assertNotIn(token, message)
                self.assertNotIn(token, stdout)
                self.assertNotIn("Traceback", message)

    def test_static_check_runtime_errors_are_wrapped_and_redacted(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("I" * 24)

        class FailingLinter:
            def lint_tree(self, root, *, include_policy, evidence_root):
                raise RuntimeError(f"candidate parser failed for {token}.yaml")

        stderr = io.StringIO()
        with patch(
            "validate_context_proposal._load_linter",
            return_value=FailingLinter(),
        ), redirect_stderr_to(stderr):
            code, stdout = self._run_main([])
        message = stderr.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("candidate", message)
        self.assertNotIn(token, message)
        self.assertNotIn(token, stdout)
        self.assertNotIn("Traceback", message)

    def test_main_wraps_unexpected_validation_and_staging_errors(self) -> None:
        from validate_context_proposal import ProposalReport

        report = ProposalReport(True, True, (), (), "")
        cases = (
            ("validation", "validate_context_proposal.validate_proposal", ()),
            ("serialization", "validate_context_proposal._serialize_report", ()),
            (
                "staging",
                "validate_context_proposal._stage",
                ("--output-report", str(self.staging / "unexpected.json")),
            ),
        )
        for role, target, extra in cases:
            with self.subTest(role=role):
                token = "github_pat_" + ("R" * 30) + "\x00"
                stderr = io.StringIO()
                patches = [
                    patch(target, side_effect=TypeError(f"raw {token}"))
                ]
                if role in ("staging", "serialization"):
                    patches.extend((
                        patch("validate_context_proposal._preflight_staging", return_value=None),
                        patch("validate_context_proposal.validate_proposal", return_value=report),
                    ))
                with redirect_stderr_to(stderr):
                    for context in patches:
                        context.start()
                    try:
                        code, stdout = self._run_main(list(extra))
                    finally:
                        for context in reversed(patches):
                            context.stop()
                message = stderr.getvalue()
                self.assertEqual(code, 2)
                self.assertIn("error", message.lower())
                self.assertNotIn(token, message)
                self.assertNotIn("\x00", message)
                self.assertNotIn("Traceback", message)
                self.assertLessEqual(len(message.rstrip("\n")), 512)
                self.assertNotIn(token, stdout)

    def test_main_redacts_unknown_paths_from_exception_diagnostics(self) -> None:
        from validate_context_proposal import ProposalReport

        report = ProposalReport(True, True, (), (), "")
        cases = (
            (
                "validation",
                "validate_context_proposal.validate_proposal",
                f"failed input {self.baseline}; remote C:/private/customer/secret.txt",
                str(self.baseline),
                "C:/private/customer/secret.txt",
                [],
            ),
            (
                "serialization",
                "validate_context_proposal._serialize_report",
                r"failed report \\server\share\customer\secret.txt",
                r"\\server\share\customer\secret.txt",
                r"\\server\share\customer\secret.txt",
                ["--output-report", str(self.staging / "report.json")],
            ),
            (
                "staging",
                "validate_context_proposal._stage",
                "failed diff /private/customer/secret.txt",
                "/private/customer/secret.txt",
                "/private/customer/secret.txt",
                ["--output-diff", str(self.staging / "proposal.diff")],
            ),
        )
        for phase, target, raw, known, unknown, extra in cases:
            with self.subTest(phase=phase):
                stderr = io.StringIO()
                patches = [patch(target, side_effect=ValueError(raw))]
                if phase in ("serialization", "staging"):
                    patches.extend((
                        patch(
                            "validate_context_proposal._preflight_staging",
                            return_value=None,
                        ),
                        patch(
                            "validate_context_proposal.validate_proposal",
                            return_value=report,
                        ),
                    ))
                with redirect_stderr_to(stderr):
                    for context in patches:
                        context.start()
                    try:
                        code, _ = self._run_main(extra)
                    finally:
                        for context in reversed(patches):
                            context.stop()
                message = stderr.getvalue()
                self.assertEqual(code, 2)
                self.assertIn("error", message.lower())
                self.assertNotIn(unknown, message)
                self.assertNotIn("Traceback", message)
                if phase == "validation":
                    self.assertIn(known, message)

    def test_serialize_report_preserves_live_secret_rejection_reason(self) -> None:
        from validate_context_proposal import StagingError, _serialize_report

        payload = {
            "baseline_ok": True,
            "candidate_ok": True,
            "regressions": ["sk-" + ("R" * 24)],
            "improvements": [],
            "diff": "",
        }
        with self.assertRaises(StagingError) as ctx:
            _serialize_report(payload, trailing_newline=False)
        message = str(ctx.exception)
        self.assertIn("live secret", message)
        self.assertNotIn("cannot be encoded", message)

    def test_main_sanitizes_non_dataclass_report_serialization_failure(self) -> None:
        stderr = io.StringIO()
        with patch(
            "validate_context_proposal.validate_proposal", return_value=None
        ):
            with redirect_stderr_to(stderr):
                code, _ = self._run_main([])

        message = stderr.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("error", message.lower())
        self.assertNotIn("Traceback", message)

    def test_staging_diagnostics_redact_secret_shaped_output_filename(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        token = "sk-" + ("D" * 24)
        report_path = self.staging / f"{token}.json"

        code, first_stdout = self._run_main([
            "--output-report", str(report_path),
        ])
        self.assertEqual(code, 0)
        self.assertTrue(report_path.exists())

        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            second_code, second_stdout = self._run_main([
                "--output-report", str(report_path),
            ])
        message = first_stdout + second_stdout + stderr.getvalue()
        self.assertEqual(second_code, 2)
        self.assertIn("staged report", first_stdout)
        self.assertIn("replace", stderr.getvalue().lower())
        self.assertIn(str(self.staging), message)
        self.assertNotIn(token, message)

    def test_reparse_metadata_exceptions_are_sanitized_without_context(self) -> None:
        from validate_context_proposal import (
            StagingError,
            _reparse_component,
        )

        token = "sk-" + ("K" * 24)
        secret_path = self.staging / f"{token}.txt"
        with patch(
            "validate_context_proposal._symlink_component",
            return_value=False,
        ), patch("validate_context_proposal._WIN", True), patch(
            "validate_context_proposal._windows_file_attributes",
            side_effect=OSError(f"reparse metadata denied for {token}"),
        ), self.assertRaises(StagingError) as ctx:
            _reparse_component(secret_path)
        message = str(ctx.exception)
        self.assertIn("reparse", message.lower())
        self.assertNotIn(token, message)
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_direct_stage_write_errors_are_sanitized_without_context(self) -> None:
        from validate_context_proposal import StagingError, _stage

        token = "sk-" + ("N" * 24)
        target = self.staging / "report.json"
        for method in ("mkdir", "write_bytes"):
            for error in (
                OSError(f"output write denied for {token}"),
                TypeError(f"output writer rejected {token}"),
            ):
                with self.subTest(method=method, error=type(error).__name__), patch.object(
                    Path,
                    method,
                    side_effect=error,
                ), self.assertRaises(StagingError) as ctx:
                    _stage(target, "payload\n", replace=False)
                message = str(ctx.exception)
                self.assertIn("staged", message.lower())
                self.assertNotIn(token, message)
                self.assertLessEqual(len(message), 512)
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    def test_staging_error_sanitizes_all_constructor_arguments(self) -> None:
        from validate_context_proposal import StagingError

        token = "sk-" + ("L" * 24)
        error = StagingError("safe prefix", token, "x" * 1200)
        for argument in error.args:
            self.assertLessEqual(len(str(argument)), 512)
            self.assertNotIn(token, str(argument))
        self.assertNotIn(token, str(error))

    def test_staging_error_sanitizes_unknown_path_arguments(self) -> None:
        from validate_context_proposal import StagingError

        unknown = "C:/private/customer/secret.txt"
        error = StagingError("safe prefix", unknown)
        self.assertNotIn(unknown, str(error))
        self.assertNotIn(unknown, str(error.args[1]))
        self.assertLessEqual(len(str(error)), 512)

    def test_staging_path_comparison_diagnostics_redact_secret_values(self) -> None:
        from validate_context_proposal import (
            StagingError,
            _canonical_staging_path,
            _is_within,
            _is_within_or_different_drive,
        )

        token = "sk-" + ("E" * 24)
        secret_path = self.staging / f"{token}.json"
        original_abspath = os.path.abspath

        def fail_for_secret(path):
            if os.path.normcase(os.fspath(path)) == os.path.normcase(
                os.fspath(secret_path)
            ):
                raise OSError(f"metadata denied for {token}")
            return original_abspath(path)

        operations = (
            ("canonical", lambda: _canonical_staging_path(secret_path)),
            ("within", lambda: _is_within(secret_path, self.staging)),
            (
                "within-or-different-drive",
                lambda: _is_within_or_different_drive(secret_path, self.staging),
            ),
        )
        for name, operation in operations:
            with self.subTest(operation=name), patch(
                "validate_context_proposal.os.path.abspath",
                side_effect=fail_for_secret,
            ), self.assertRaises(StagingError) as ctx:
                operation()
            message = str(ctx.exception.args[0])
            self.assertNotIn(token, message)
            self.assertNotIn("\x1f", message)
            self.assertLessEqual(len(message), 512)
            self.assertTrue(message)

    def test_staging_validation_diagnostics_redact_secret_paths_and_errors(self) -> None:
        from validate_context_proposal import (
            StagingError,
            _INVALID_FILE_ATTRIBUTES,
            _check_staging,
            _destination_metadata,
            _reparse_component,
            _symlink_component,
        )

        token = "sk-" + ("F" * 24)
        secret_name = f"{token}.json"
        secret_path = self.staging / secret_name

        def assert_redacted(ctx, expected: str) -> None:
            message = str(ctx.exception.args[0])
            self.assertIn(expected, message)
            self.assertNotIn(token, message)
            self.assertNotIn("\x1f", message)
            self.assertLessEqual(len(message), 512)

        with patch(
            "validate_context_proposal._reparse_component",
            return_value=None,
        ), patch(
            "validate_context_proposal.os.path.realpath",
            side_effect=OSError(f"resolution metadata denied for {token}"),
        ), self.assertRaises(StagingError) as ctx:
            _check_staging(secret_path, False)
        assert_redacted(ctx, "resolution")

        with patch(
            "validate_context_proposal.os.path.abspath",
            side_effect=OSError(f"abspath denied for {token}"),
        ), self.assertRaises(StagingError) as ctx:
            _check_staging(secret_path, False)
        assert_redacted(ctx, "abspath")

        protected_path = self.staging / token / "SOUL.md"
        with self.assertRaises(StagingError) as ctx:
            _check_staging(protected_path, False)
        assert_redacted(ctx, "protected path")

        live_path = REPO_ROOT / token / "proposal.json"
        with self.assertRaises(StagingError) as ctx:
            _check_staging(live_path, False)
        assert_redacted(ctx, "live repository")

        existing_directory = self.staging / f"{token}-directory"
        existing_directory.mkdir()
        with self.assertRaises(StagingError) as ctx:
            _check_staging(existing_directory, True)
        assert_redacted(ctx, "unsafe existing destination")

        existing_file = self.staging / f"{token}-overwrite.json"
        existing_file.write_text("fixture\n", encoding="utf-8")
        with self.assertRaises(StagingError) as ctx:
            _check_staging(existing_file, False)
        assert_redacted(ctx, "without --replace")

        with patch(
            "validate_context_proposal.os.stat",
            side_effect=OSError(f"destination metadata denied for {token}"),
        ), self.assertRaises(StagingError) as ctx:
            _destination_metadata(secret_path)
        assert_redacted(ctx, "destination metadata")

        with patch(
            "validate_context_proposal.os.lstat",
            side_effect=OSError(f"path metadata denied for {token}"),
        ), self.assertRaises(StagingError) as ctx:
            _symlink_component(secret_path)
        assert_redacted(ctx, "path component")

        with patch(
            "validate_context_proposal._WIN",
            True,
        ), patch(
            "validate_context_proposal._symlink_component",
            return_value=False,
        ), patch(
            "validate_context_proposal._windows_file_attributes",
            return_value=(_INVALID_FILE_ATTRIBUTES, 5),
        ), self.assertRaises(StagingError) as ctx:
            _reparse_component(secret_path)
        assert_redacted(ctx, "reparse protection")

    def test_differing_invalid_utf8_files_fail_closed_instead_of_empty_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        (self.baseline / "invalid.bin").write_bytes(b"\xff\x00A")
        (self.candidate / "invalid.bin").write_bytes(b"\xfe\x00B")
        from validate_context_proposal import render_diff

        with self.assertRaises(ValueError) as ctx:
            render_diff(self.baseline, self.candidate)
        self.assertIn("invalid.bin", str(ctx.exception))

        report_path = self.staging / "binary-report.json"
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, _ = self._run_main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(report_path),
            ])
        self.assertEqual(code, 2)
        self.assertIn("invalid.bin", stderr.getvalue())
        self.assertFalse(report_path.exists())

    # -- refusal of protected writes ---------------------------------------

    def test_is_protected_output_rejects_policy_and_rules(self) -> None:
        from validate_context_proposal import is_protected_output

        self.assertTrue(is_protected_output(self.staging / "SOUL.md"))
        self.assertTrue(is_protected_output(self.staging / "AGENTS.md"))
        self.assertTrue(is_protected_output(self.staging / "CLAUDE.md"))
        self.assertTrue(is_protected_output(self.staging / "rules" / "nested" / "gpu.md"))
        self.assertFalse(is_protected_output(self.staging / "report.md"))

    def test_main_refuses_to_write_policy_file(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        from validate_context_proposal import main

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(self.staging / "AGENTS.md"),
            ])
        self.assertEqual(code, 2)
        self.assertFalse((self.staging / "AGENTS.md").exists())

    def test_main_rejects_live_repository_output_even_with_replace(self) -> None:
        target = REPO_ROOT / "docs" / "agent" / "REFLECTION.md"
        before = target.read_bytes() if target.exists() else None
        try:
            repo_temp = tempfile.TemporaryDirectory(dir=REPO_ROOT)
        except OSError as exc:
            self.skipTest(f"repository-local fixture is unavailable: {exc}")
        with repo_temp:
            fixture = Path(repo_temp.name)
            self.baseline = fixture / "baseline"
            self.candidate = fixture / "candidate"
            self.staging = fixture / "staging"
            self.baseline.mkdir()
            self.candidate.mkdir()
            self.staging.mkdir()
            _make_context(self.baseline, GOOD_TASKS)
            _make_context(self.candidate, GOOD_TASKS)
            stderr = io.StringIO()
            with patch(
                "validate_context_proposal._lint_issues",
                side_effect=AssertionError("linter must not run"),
            ), patch(
                "validate_context_proposal.render_diff",
                side_effect=AssertionError("diff must not run"),
            ), redirect_stderr_to(stderr):
                code, _ = self._run_main([
                    "--output-report", str(target),
                    "--replace",
                ])
        self.assertEqual(code, 2)
        self.assertIn("staging", stderr.getvalue().lower())
        if before is None:
            self.assertFalse(target.exists())
        else:
            self.assertEqual(target.read_bytes(), before)

    def test_main_allows_approved_staging_under_repository_evidence_root(self) -> None:
        from validate_context_proposal import ProposalReport, main

        target = REPO_ROOT / ".skillopt-sleep" / "staging" / (
            f"proposal-test-{uuid.uuid4().hex}.json"
        )
        report = ProposalReport(True, True, (), (), "")
        with patch(
            "validate_context_proposal.validate_proposal",
            return_value=report,
        ), patch("validate_context_proposal._stage") as stage:
            code = main([
                "--baseline", str(REPO_ROOT / ".proposal-test-baseline"),
                "--candidate", str(REPO_ROOT / ".proposal-test-candidate"),
                "--root", str(REPO_ROOT),
                "--output-report", str(target),
            ])
        self.assertEqual(code, 0)
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[0], target)
        self.assertFalse(target.exists())

    def test_refuses_hard_link_output_even_with_replace(self) -> None:
        owner = self.staging / "owner.txt"
        owner.write_text("original\n", encoding="utf-8")
        alias = self.staging / "owner-alias.txt"
        try:
            os.link(owner, alias)
        except OSError as exc:
            self.skipTest(f"hard-link creation unavailable: {exc}")
        from validate_context_proposal import StagingError, _check_staging

        with self.assertRaises(StagingError):
            _check_staging(alias, replace=True)
        self.assertEqual(owner.read_text(encoding="utf-8"), "original\n")

    def test_allows_new_nested_output_under_missing_parent_on_posix(self) -> None:
        from validate_context_proposal import _check_staging

        target = self.staging / "new-parent" / "nested" / "report.md"
        with patch("validate_context_proposal._WIN", False):
            _check_staging(target, replace=False)
        self.assertFalse(target.exists())

    def test_refuses_existing_non_regular_output_even_with_replace(self) -> None:
        target = self.staging / "report-directory"
        target.mkdir()
        from validate_context_proposal import StagingError, _check_staging

        with self.assertRaises(StagingError):
            _check_staging(target, replace=True)
        self.assertTrue(target.is_dir())

    def test_refuses_indeterminate_output_metadata(self) -> None:
        from validate_context_proposal import StagingError, _check_staging

        target = self.staging / "report.json"
        with patch(
            "validate_context_proposal.os.stat",
            side_effect=PermissionError("simulated metadata denial"),
        ):
            with self.assertRaises(StagingError):
                _check_staging(target, replace=False)
        self.assertFalse(target.exists())

    def test_refuses_indeterminate_output_resolution(self) -> None:
        from validate_context_proposal import StagingError, _check_staging

        target = self.staging / "report.json"
        with patch(
            "validate_context_proposal.os.path.realpath",
            side_effect=OSError("simulated resolution denial"),
        ):
            with self.assertRaises(StagingError):
                _check_staging(target, replace=False)
        self.assertFalse(target.exists())

    def test_refuses_windows_alternate_stream_output(self) -> None:
        if sys.platform != "win32":
            self.skipTest("alternate-stream coverage is Windows-only")
        owner = self.staging / "owner.txt"
        owner.write_text("original\n", encoding="utf-8")
        stream = Path(str(owner) + ":proposal")
        from validate_context_proposal import StagingError, _check_staging

        with self.assertRaises(StagingError):
            _check_staging(stream, replace=True)
        self.assertEqual(owner.read_text(encoding="utf-8"), "original\n")

    def test_refuses_windows_protected_lexical_alias(self) -> None:
        if sys.platform != "win32":
            self.skipTest("Windows lexical aliases are Windows-only")
        target = self.staging / "rules." / "report.json"
        from validate_context_proposal import StagingError, _check_staging

        with self.assertRaises(StagingError):
            _check_staging(target, replace=True)
        self.assertFalse(target.exists())

    def test_refuses_windows_dangling_junction_component(self) -> None:
        if sys.platform != "win32":
            self.skipTest("junction coverage is Windows-only")
        import subprocess

        junction = self.staging / "dangling"
        target = junction / "report.json"
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(self.staging / "missing")],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("junctions are unavailable in this environment")
        from validate_context_proposal import StagingError, _check_staging

        with self.assertRaises(StagingError):
            _check_staging(target, replace=True)
        self.assertFalse(target.exists())


    def test_preflights_both_outputs_before_writing_either(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        report_path = self.staging / "proposal-report.json"
        owner = self.staging / "diff-owner.txt"
        owner.write_text("original\n", encoding="utf-8")
        diff_path = self.staging / "diff-alias.txt"
        try:
            os.link(owner, diff_path)
        except OSError as exc:
            self.skipTest(f"hard-link creation unavailable: {exc}")

        with redirect_stderr_to(io.StringIO()):
            code, _ = self._run_main([
                "--output-report", str(report_path),
                "--output-diff", str(diff_path),
                "--replace",
            ])
        self.assertEqual(code, 2)
        self.assertFalse(report_path.exists())
        self.assertEqual(owner.read_text(encoding="utf-8"), "original\n")

    def test_refuses_report_and_diff_at_same_destination(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        target = self.staging / "proposal-output.txt"

        with redirect_stderr_to(io.StringIO()):
            code, _ = self._run_main([
                "--output-report", str(target),
                "--output-diff", str(target),
                "--replace",
            ])
        self.assertEqual(code, 2)
        self.assertFalse(target.exists())

    # -- CLI behavior -------------------------------------------------------

    def _run_main(self, extra: list[str]) -> tuple[int, str]:
        from validate_context_proposal import main

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            argv = ["--baseline", str(self.baseline),
                    "--candidate", str(self.candidate),
                    "--root", str(self.baseline)] + extra
            code = main(argv)
        return code, stdout.getvalue()

    def test_dry_run_writes_nothing(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        before = sorted(str(p) for p in self.staging.rglob("*"))
        code, out = self._run_main([])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(str(p) for p in self.staging.rglob("*")), before)
        self.assertIn("candidate_ok", out)
        self.assertIn("staged", out.lower())

    def test_dry_run_preserves_complete_json_report_data(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        long_line = "x" * 2000
        (self.candidate / "long.txt").write_text(
            long_line + "\n", encoding="utf-8"
        )

        code, out = self._run_main([])
        self.assertEqual(code, 0)
        start = out.find("{")
        self.assertGreaterEqual(start, 0)
        payload, end = json.JSONDecoder().raw_decode(out[start:])
        self.assertIn(long_line, payload["diff"])
        self.assertIn("staged: no file written", out[end:])


    def test_main_writes_only_staged_report_and_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        report_path = self.staging / "proposal-report.json"
        diff_path = self.staging / "proposal.diff"
        code, out = self._run_main([
            "--output-report", str(report_path),
            "--output-diff", str(diff_path),
        ])
        self.assertEqual(code, 0)
        written = sorted(p.relative_to(self.staging).as_posix()
                         for p in self.staging.rglob("*") if p.is_file())
        self.assertEqual(written, ["proposal-report.json", "proposal.diff"])
        self.assertTrue(report_path.exists())
        self.assertTrue(diff_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("candidate_ok", payload)
        self.assertFalse(payload["candidate_ok"])
        self.assertIn("staged", out.lower())

    def test_main_writes_diff_when_requested(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        protected = (
            self.baseline / "AGENTS.md",
            self.baseline / "CLAUDE.md",
            self.baseline / "SOUL.md",
            self.baseline / "rules" / "fixture.md",
        )
        before = {path: path.read_bytes() for path in protected}
        report_path = self.staging / "proposal-report.json"
        diff_path = self.staging / "proposal.diff"
        code, _ = self._run_main([
            "--output-report", str(report_path),
            "--output-diff", str(diff_path),
        ])
        self.assertEqual(code, 0)
        from validate_context_proposal import render_diff
        expected = render_diff(self.baseline, self.candidate)
        if expected:
            self.assertTrue(diff_path.exists())
        # no live policy was touched
        after = {path: path.read_bytes() for path in protected}
        self.assertEqual(after, before)

    def test_surrogate_report_diagnostic_is_rejected_before_output_write(self) -> None:
        from validate_context_proposal import ProposalReport

        report_path = self.staging / "proposal-report.json"
        report_path.write_text("ORIGINAL\n", encoding="utf-8")
        report = ProposalReport(
            True,
            False,
            ("diagnostic contains an invalid surrogate: \ud800",),
            (),
            "",
        )
        stderr = io.StringIO()
        with patch(
            "validate_context_proposal.validate_proposal",
            return_value=report,
        ), redirect_stderr_to(stderr):
            code, _ = self._run_main([
                "--output-report", str(report_path),
                "--replace",
            ])

        self.assertEqual(code, 2)
        self.assertEqual(report_path.read_text(encoding="utf-8"), "ORIGINAL\n")
        self.assertIn("utf-8", stderr.getvalue().lower())
        self.assertNotIn("\ud800", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_surrogate_report_diagnostic_is_rejected_in_dry_run(self) -> None:
        from validate_context_proposal import ProposalReport

        report = ProposalReport(
            True,
            False,
            ("diagnostic contains an invalid surrogate: \ud800",),
            (),
            "",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "validate_context_proposal.validate_proposal",
            return_value=report,
        ), redirect_stdout(stdout), redirect_stderr_to(stderr):
            code, _ = self._run_main([])

        self.assertEqual(code, 2)
        self.assertIn("utf-8", stderr.getvalue().lower())
        self.assertNotIn("\ud800", stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_dry_run_rejects_live_secret_in_report_payload(self) -> None:
        from validate_context_proposal import ProposalReport

        token = "sk-" + ("P" * 24)
        report = ProposalReport(True, True, (), (), f"payload {token}\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "validate_context_proposal.validate_proposal",
            return_value=report,
        ), redirect_stdout(stdout), redirect_stderr_to(stderr):
            code, _ = self._run_main([])

        self.assertEqual(code, 2)
        self.assertNotIn(token, stdout.getvalue())
        self.assertNotIn(token, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_error_unicode_is_stream_safe(self) -> None:
        from validate_context_proposal import StagingError, main

        class AsciiStream(io.StringIO):
            encoding = "ascii"

            def write(self, value):
                value.encode("ascii")
                return super().write(value)

        stdout = AsciiStream()
        stderr = AsciiStream()
        with patch(
            "validate_context_proposal._preflight_staging",
            return_value=None,
        ), patch(
            "validate_context_proposal.validate_proposal",
            side_effect=StagingError("café"),
        ), redirect_stdout(stdout), redirect_stderr_to(stderr):
            code = main(
                [
                    "--baseline",
                    str(self.baseline),
                    "--candidate",
                    str(self.candidate),
                    "--root",
                    str(self.baseline),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn(r"caf\xe9", stderr.getvalue())
        self.assertNotIn("UnicodeEncodeError", stderr.getvalue())

    def test_dry_run_unicode_report_is_stream_safe_and_parseable(self) -> None:
        from validate_context_proposal import main

        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        (self.candidate / "README.md").write_text("# café\n", encoding="utf-8")

        class AsciiStream(io.StringIO):
            encoding = "ascii"

            def write(self, value):
                value.encode("ascii")
                return super().write(value)

        stdout = AsciiStream()
        stderr = AsciiStream()
        with redirect_stdout(stdout), redirect_stderr_to(stderr):
            code = main(
                [
                    "--baseline",
                    str(self.baseline),
                    "--candidate",
                    str(self.candidate),
                    "--root",
                    str(self.baseline),
                ]
            )
        output = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn("UnicodeEncodeError", output + stderr.getvalue())
        start = output.find("{")
        self.assertGreaterEqual(start, 0)
        payload, end = json.JSONDecoder().raw_decode(output[start:])
        self.assertIn("café", payload["diff"])
        self.assertIn("staged: no file written", output[end:])

    def test_main_composes_error_and_success_lines_with_shared_bound(self) -> None:
        from validate_context_proposal import ProposalReport, StagingError

        report = ProposalReport(True, True, (), (), "")
        stderr = io.StringIO()
        with patch(
            "validate_context_proposal.validate_proposal",
            side_effect=StagingError("failure " + ("e" * 600)),
        ), redirect_stderr_to(stderr):
            code, _ = self._run_main([])
        self.assertEqual(code, 2)
        self.assertLessEqual(len(stderr.getvalue().rstrip("\n")), 512)

        long_output = self.staging / ("r" * 180) / ("e" * 180) / ("p" * 180)
        for option in ("--output-report", "--output-diff"):
            with self.subTest(option=option), patch(
                "validate_context_proposal._preflight_staging",
                return_value=None,
            ), patch(
                "validate_context_proposal.validate_proposal",
                return_value=report,
            ), patch("validate_context_proposal._stage", return_value=None):
                code, output = self._run_main([option, str(long_output)])
            self.assertEqual(code, 0)
            self.assertLessEqual(len(output.rstrip("\n")), 512)

    def test_unknown_argument_diagnostics_redact_secrets_and_controls(self) -> None:
        from validate_context_proposal import main

        token = "sk-" + ("M" * 24)
        unknown = f"--unknown={token} " + ("x" * 1000) + "\x1f"
        stderr = io.StringIO()
        with redirect_stderr_to(stderr), self.assertRaises(SystemExit) as ctx:
            main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                unknown,
            ])
        message = stderr.getvalue()
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("unrecognized", message)
        self.assertNotIn(token, message)
        self.assertNotIn("\x1f", message)
        self.assertLessEqual(len(message), 512)

    def test_help_states_no_adoption(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            from validate_context_proposal import main
            try:
                code = main(["--help"])
            except SystemExit as exc:  # argparse exits after printing help
                code = exc.code
        self.assertEqual(code, 0)
        text = stdout.getvalue()
        self.assertIn("cannot adopt", text.lower())
        self.assertIn("skillopt-sleep", text.lower())

    def test_help_parser_rejects_adoption_flags(self) -> None:
        from validate_context_proposal import _parse_args

        for flag in ("--adopt", "--apply", "--install"):
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    _parse_args([flag])

    # -- no repository mutation --------------------------------------------

    def test_repository_baseline_files_untouched(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        protected = (
            self.baseline / "AGENTS.md",
            self.baseline / "CLAUDE.md",
            self.baseline / "SOUL.md",
            self.baseline / "rules" / "fixture.md",
        )
        before = {path: path.read_bytes() for path in protected}

        from validate_context_proposal import main
        with redirect_stdout(io.StringIO()):
            main(["--baseline", str(self.baseline),
                  "--candidate", str(self.candidate),
                  "--root", str(self.baseline)])

        after = {path: path.read_bytes() for path in protected}
        self.assertEqual(after, before)

    def test_validate_proposal_rejects_bad_input(self) -> None:
        from validate_context_proposal import validate_proposal

        with self.assertRaises(ValueError):
            validate_proposal(
                baseline=self.staging / "missing-base",
                candidate=self.candidate,
                root=self.baseline,
            )

    # -- review-fix regressions ---------------------------------------------

    def test_untrusted_tree_symlink_is_rejected_before_linter_and_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        secret = Path(self._tmp.name) / "outside-secret.txt"
        secret.write_text("LEAKED_SECRET_VALUE\n", encoding="utf-8")

        for tree in (self.baseline, self.candidate):
            with self.subTest(tree=tree.name):
                link = tree / "linked-secret.txt"
                try:
                    link.symlink_to(secret)
                except OSError as exc:
                    self.skipTest(f"symlink creation unavailable: {exc}")
                with patch(
                    "validate_context_proposal._lint_issues",
                    side_effect=AssertionError("linter must not run"),
                ), patch(
                    "validate_context_proposal.render_diff",
                    side_effect=AssertionError("diff must not run"),
                ):
                    with self.assertRaises(ValueError) as ctx:
                        self._report()
                self.assertNotIn("LEAKED_SECRET_VALUE", str(ctx.exception))
                link.unlink()

    def test_untrusted_tree_hard_link_is_rejected_before_linter_and_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        owner = Path(self._tmp.name) / "owner.txt"
        owner.write_text("linked input\n", encoding="utf-8")
        linked = self.candidate / "linked-input.txt"
        try:
            os.link(owner, linked)
        except OSError as exc:
            self.skipTest(f"hard-link creation unavailable: {exc}")

        with patch(
            "validate_context_proposal._lint_issues",
            side_effect=AssertionError("linter must not run"),
        ), patch(
            "validate_context_proposal.render_diff",
            side_effect=AssertionError("diff must not run"),
        ):
            with self.assertRaises(ValueError) as ctx:
                self._report()
        self.assertIn("hard", str(ctx.exception).lower())

    def test_untrusted_root_symlink_is_rejected_before_linter_and_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        root_link = Path(self._tmp.name) / "root-link"
        try:
            root_link.symlink_to(self.baseline, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with patch(
            "validate_context_proposal._lint_issues",
            side_effect=AssertionError("linter must not run"),
        ), patch(
            "validate_context_proposal.render_diff",
            side_effect=AssertionError("diff must not run"),
        ):
            with self.assertRaises(ValueError):
                self._report(root=root_link)

    def test_staging_outputs_under_input_trees_are_rejected_even_with_replace(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        targets = (
            self.baseline / "proposal-report.json",
            self.candidate / "proposal-diff.txt",
            self.baseline / "nested" / "proposal-report.json",
        )
        for target in targets:
            with self.subTest(target=target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("KEEP_ME\n", encoding="utf-8")
                stderr = io.StringIO()
                with redirect_stderr_to(stderr):
                    code, _ = self._run_main([
                        "--output-report", str(target), "--replace",
                    ])
                self.assertEqual(code, 2)
                self.assertEqual(target.read_text(encoding="utf-8"), "KEEP_ME\n")

    def test_windows_short_alias_cannot_bypass_input_root_output_ban(self) -> None:
        if sys.platform != "win32":
            self.skipTest("8.3 path aliases are Windows-only")
        import ctypes

        long_root = self.staging / "context-tree-with-a-long-directory-name"
        long_root.mkdir()
        _make_context(long_root, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)

        try:
            get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
            get_short_path_name.argtypes = [
                ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32,
            ]
            get_short_path_name.restype = ctypes.c_uint32
            size = get_short_path_name(str(long_root), None, 0)
            buffer = ctypes.create_unicode_buffer(size) if size else None
            written = (
                get_short_path_name(str(long_root), buffer, size)
                if buffer is not None
                else 0
            )
        except (AttributeError, OSError) as exc:
            self.skipTest(f"Windows 8.3 short-path API is unavailable: {exc}")
        if not written or buffer is None or not buffer.value:
            self.skipTest("Windows 8.3 short paths are unavailable")
        short_root = Path(buffer.value)
        try:
            same_directory = os.path.samefile(long_root, short_root)
        except OSError as exc:
            self.skipTest(f"could not verify the Windows short alias: {exc}")
        if (
            not same_directory
            or os.path.normcase(short_root.name) == os.path.normcase(long_root.name)
        ):
            self.skipTest("Windows did not create a distinct 8.3 alias")

        output = short_root / "proposal-report.json"
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, _ = self._run_main([
                "--baseline", str(long_root),
                "--candidate", str(self.candidate),
                "--root", str(long_root),
                "--output-report", str(output),
            ])
        self.assertEqual(code, 2, stderr.getvalue())
        self.assertIn("input or evidence root", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_invalid_output_is_rejected_before_linter_and_diff(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        target = self.staging / "report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("KEEP_ME\n", encoding="utf-8")
        with patch(
            "validate_context_proposal._lint_issues",
            side_effect=AssertionError("linter must not run"),
        ), patch(
            "validate_context_proposal.render_diff",
            side_effect=AssertionError("diff must not run"),
        ):
            with redirect_stderr_to(io.StringIO()):
                code, _ = self._run_main([
                    "--output-report", str(target),
                ])
        self.assertEqual(code, 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "KEEP_ME\n")

    def test_malformed_status_types_are_reported_without_traceback(self) -> None:
        malformed_tasks = (
            "tasks:\n"
            "  - id: mock/lesson-smoke\n"
            "    status: [bad]\n"
            "    evidence: []\n"
            "experiments:\n"
            "  - id: mock/lesson-experiment\n"
            "    status: running\n"
            "    verdict: {bad: true}\n"
        )
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, malformed_tasks)
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, output = self._run_main([])
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("invalid task status", output)
        self.assertIn("invalid experiment verdict", output)

    def test_deep_yaml_recursion_is_reported_without_traceback(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        nested = "tasks: " + "[" * 2000 + "0" + "]" * 2000 + "\n"
        (self.candidate / "state" / "tasks.yaml").write_text(
            nested, encoding="utf-8"
        )
        stderr = io.StringIO()
        with redirect_stderr_to(stderr):
            code, output = self._run_main([])

        self.assertEqual(code, 0)
        self.assertIn("cannot parse state/tasks.yaml", output)
        self.assertNotIn("RecursionError", output)
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_mixed_regression_and_improvement(self) -> None:
        broken_readme = "# Fixture\nbroken_markdown_link: [x](missing-file.md)\n"
        _make_context(self.baseline, GOOD_TASKS, readme_text=broken_readme)
        _make_context(self.candidate, BAD_TASKS, readme_text="# Fixture\n")
        report = self._report()
        self.assertTrue(report.regressions)
        self.assertTrue(report.improvements)
        self.assertIn("invalid task status", "\n".join(report.regressions))
        self.assertIn("broken markdown link", "\n".join(report.improvements))
        self.assertFalse(report.candidate_ok)

    def test_policy_regression_in_candidate_detected(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        (Path(self.candidate) / "AGENTS.md").write_text(
            "run with python scripts/train.py\n", encoding="utf-8"
        )
        report = self._report()
        self.assertIn("bare Python command", "\n".join(report.regressions))
        self.assertFalse(report.candidate_ok)

    def test_root_is_evidence_resolution_root(self) -> None:
        tasks_with_evidence = (
            "tasks:\n"
            "  - id: mock/lesson-smoke\n"
            "    status: done\n"
            "    evidence: [scratch/note.txt]\n"
            "experiments: []\n"
        )
        # The candidate ADDS an evidence reference; the baseline has none, so
        # any missing-evidence issue must come from the proposal.
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, tasks_with_evidence)
        (Path(self.candidate) / "scratch").mkdir()
        (Path(self.candidate) / "scratch" / "note.txt").write_text("x\n", encoding="utf-8")
        evidence_root = Path(self._tmp.name) / "evidence_root"
        evidence_root.mkdir()
        # Evidence missing from the repo root: regression.
        report = self._report(root=evidence_root)
        self.assertFalse(report.candidate_ok)
        self.assertIn(
            "missing evidence path: scratch/note.txt",
            "\n".join(report.regressions),
        )
        # Same candidate with the evidence present at the root: passes.
        (evidence_root / "scratch").mkdir()
        (evidence_root / "scratch" / "note.txt").write_text("x\n", encoding="utf-8")
        report = self._report(root=evidence_root)
        self.assertEqual(report.regressions, ())
        self.assertTrue(report.candidate_ok)

    def test_junction_escape_to_victim_file_is_refused(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        escaped = False
        victim_dir = Path(self._tmp.name) / "victim"
        victim_dir.mkdir()
        victim = victim_dir / "SOUL.md"
        victim.write_text("original\n", encoding="utf-8")
        junction = self.staging / "escape"
        try:
            import subprocess

            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(victim_dir)],
                check=True, capture_output=True,
            )
            escaped = True
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("junctions/symlinks are unavailable in this environment")
        self.assertTrue(escaped)
        from validate_context_proposal import main

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr_to(io.StringIO()):
            code = main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(junction / "SOUL.md"),
            ])
        self.assertEqual(code, 2)
        self.assertEqual(victim.read_text(encoding="utf-8"), "original\n")
        # Containment also refuses non-protected names through the junction:
        # the resolved location leaves the staged directory, so the proposal
        # tool must refuse even tasks.yaml (no protected name involved).
        victim_tasks = victim_dir / "tasks.yaml"
        victim_tasks.write_text("immutable\n", encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr_to(io.StringIO()):
            code = main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(junction / "tasks.yaml"),
            ])
        self.assertEqual(code, 2)
        self.assertEqual(victim_tasks.read_text(encoding="utf-8"), "immutable\n")

    def test_posix_symlink_output_is_refused(self) -> None:
        if sys.platform == "win32":
            self.skipTest("the Windows junction test covers reparse-point staging")
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        victim_dir = Path(self._tmp.name) / "victim"
        victim_dir.mkdir()
        victim = victim_dir / "SOUL.md"
        victim.write_text("original\n", encoding="utf-8")
        link = self.staging / "escape"
        try:
            link.symlink_to(victim_dir, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        from validate_context_proposal import main

        with redirect_stdout(io.StringIO()), redirect_stderr_to(io.StringIO()):
            code = main([
                "--baseline", str(self.baseline),
                "--candidate", str(self.candidate),
                "--root", str(self.baseline),
                "--output-report", str(link / "SOUL.md"),
            ])
        self.assertEqual(code, 2)
        self.assertEqual(victim.read_text(encoding="utf-8"), "original\n")

    def test_staging_uses_checked_normalized_path_after_dot_segment(self) -> None:
        if sys.platform == "win32":
            self.skipTest("POSIX symlink resolution regression")
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        victim_dir = Path(self._tmp.name) / "victim"
        victim_dir.mkdir()
        outside = Path(self._tmp.name) / "outside.txt"
        outside.write_text("unchanged\n", encoding="utf-8")
        link = self.staging / "link"
        try:
            link.symlink_to(victim_dir, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        from validate_context_proposal import _stage

        requested = link / ".." / "outside.txt"
        _stage(requested, "staged\n", replace=False)
        self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged\n")
        self.assertEqual(
            (self.staging / "outside.txt").read_text(encoding="utf-8"),
            "staged\n",
        )

    def test_output_diff_alone_is_honored(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, BAD_TASKS)
        diff_path = self.staging / "proposal.diff"
        code, _ = self._run_main(["--output-diff", str(diff_path)])
        self.assertEqual(code, 0)
        self.assertTrue(diff_path.exists())
        report_files = [p for p in self.staging.rglob("*")
                        if p.is_file() and p.name != "proposal.diff"]
        self.assertEqual(report_files, [])

    def test_refuses_overwrite_without_replace(self) -> None:
        _make_context(self.baseline, GOOD_TASKS)
        _make_context(self.candidate, GOOD_TASKS)
        report_path = self.staging / "proposal-report.json"
        code, _ = self._run_main(["--output-report", str(report_path)])
        self.assertEqual(code, 0)
        with redirect_stderr_to(io.StringIO()):
            code, _ = self._run_main(["--output-report", str(report_path)])
        self.assertEqual(code, 2)
        code, _ = self._run_main([
            "--output-report", str(report_path), "--replace",
        ])
        self.assertEqual(code, 0)


def redirect_stderr_to(buffer):
    from contextlib import redirect_stderr

    return redirect_stderr(buffer)


if __name__ == "__main__":
    unittest.main()
