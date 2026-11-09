import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.closeout_reflection import (
    MAX_INPUT_BYTES,
    ProtectedPathError,
    ValidationError,
    _safe_diagnostic,
    _safe_exception_detail,
    _same_destination,
    _validate_output_path,
    build_reflection,
    lesson_candidate_id,
    main,
    stage_lesson_candidate,
    write_reflection,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "reflection_sample.json"


def base_payload():
    return {
        "requested_outcome": "Add task-closeout reflection support",
        "actual_outcome": "Reflection tool and docs added",
        "changed_files": ["scripts/closeout_reflection.py"],
        "validation_performed": ["unittest tests.agent_context.test_closeout_reflection -v"],
        "failures_near_misses": [],
        "root_causes": [],
        "fixes": [],
        "evidence": ["tests/agent_context/test_closeout_reflection.py"],
        "reusable_lesson": "Validate inputs before writing generated state.",
        "rule_candidate": "closeout: validate inputs before any write",
        "open_questions": [],
        "confidence_limitations": "Offline dry-run only.",
        "scope": "core",
        "source_reflection": "docs/reflections/example.md",
    }


def seed_state():
    return {
        "schema_version": 1,
        "lessons": [
            {
                "id": "docs/lesson-0001",
                "scope": "docs",
                "status": "candidate",
                "lesson": "pre-existing unrelated lesson",
                "source_reflection": "docs/reflections/old.md",
                "evidence": ["old/evidence.txt"],
                "promoted_to": [],
            }
        ],
    }


EXTENDED_SECRET_SAMPLES = {
    "openai": "sk-" + "A" * 24,
    "gitlab": "glpat-" + "B" * 24,
    "google": "AIza" + "C" * 35,
    "huggingface": "hf_" + "D" * 34,
    "slack": "xoxb-" + "E" * 24,
    "aws_asia": "ASIA" + "F" * 16,
    "github_fine_grained": "github_pat_" + "G" * 30,
    "authorization_bearer": "authorization: Bearer " + "H" * 30,
    "client_secret_assignment": "client_secret=" + "I" * 30,
    "private_key_assignment": "private_key=" + "J" * 30,
    "standalone_jwt": (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        + "S" * 32
    ),
    "explicit_bearer": "Bearer " + "H" * 30,
    "explicit_bearer_colon": "Bearer:" + "H" * 30,
    "explicit_bearer_equals": "Bearer=" + "H" * 30,
    "short_jwt": (
        "eyJhbGciOiJIUzI1NiJ9.e30."
        + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    ),
    "unsigned_jwt": "eyJhbGciOiJub25lIn0.e30.",
    "complete_private_key": (
        "-----BEGIN PRIVATE KEY-----\n"
        + ("M" * 64)
        + "\n-----END PRIVATE KEY-----"
    ),
}


class BuildReflectionTests(unittest.TestCase):
    def test_render_is_deterministic_and_complete(self):
        first = build_reflection(base_payload())
        second = build_reflection(base_payload())
        self.assertEqual(first, second)
        for heading in (
            "Requested outcome",
            "Actual outcome",
            "Changed files",
            "Validation performed",
            "Failures and near-misses",
            "Root causes",
            "Fixes",
            "Evidence",
            "Reusable lesson",
            "Rule candidate",
            "Open questions",
            "Confidence and limitations",
            "Lesson candidate",
        ):
            self.assertIn(heading, first)

    def test_omitted_scope_uses_core_in_reflection_and_lesson_candidate(self):
        payload = base_payload()
        del payload["scope"]
        reflection = build_reflection(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "lessons_pending.json"
            state_path.write_text(json.dumps(seed_state()), encoding="utf-8")
            candidate = stage_lesson_candidate(payload, state_path)
        self.assertIn("Scope: core", reflection)
        self.assertIn(candidate["id"], reflection)
        self.assertEqual(candidate["scope"], "core")

    def test_evidence_must_be_existing_repo_relative_artifact_paths(self):
        invalid_paths = (
            r"tests\agent_context\test_closeout_reflection.py",
            "../outside.txt",
            "https://example.invalid/result.json",
            "tests/agent_context/does_not_exist.txt",
        )
        for evidence_path in invalid_paths:
            with self.subTest(evidence=evidence_path):
                payload = base_payload()
                payload["evidence"] = [evidence_path]
                with self.assertRaises(ValidationError) as ctx:
                    build_reflection(payload)
                self.assertIn("evidence", str(ctx.exception).lower())

    def test_evidence_rejects_colon_in_any_component(self):
        payload = base_payload()
        payload["evidence"] = ["tests/agent_context/test_closeout_reflection.py:stream"]
        with patch("scripts.closeout_reflection.os.stat", return_value=object()):
            with self.assertRaises(ValidationError) as ctx:
                build_reflection(payload)
        self.assertIn("evidence", str(ctx.exception).lower())

    def test_evidence_rejects_windows_trailing_dot_and_space_aliases(self):
        if sys.platform != "win32":
            self.skipTest("Windows lexical evidence aliases are Windows-only")
        base = "tests/agent_context/test_closeout_reflection.py"
        for evidence_path in (base + ".", base + " "):
            with self.subTest(evidence=evidence_path):
                payload = base_payload()
                payload["evidence"] = [evidence_path]
                with self.assertRaises(ValidationError) as ctx:
                    build_reflection(payload)
                self.assertIn("evidence", str(ctx.exception).lower())

    def test_build_reflection_redacts_evidence_path_resolution_error(self):
        token = "sk-" + "Y" * 24
        with patch(
            "scripts.closeout_reflection.os.path.abspath",
            side_effect=OSError(f"raw evidence path {token}\t"),
        ):
            with self.assertRaises((ValidationError, OSError)) as ctx:
                build_reflection(base_payload())
        self.assertIsInstance(ctx.exception, ValidationError)
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\t", message)
        self.assertIn("filesystem operation failed", message)

    def test_validate_reflection_redacts_evidence_inspection_error(self):
        token = "sk-" + "N" * 24
        with patch(
            "scripts.closeout_reflection._validate_evidence_paths",
            side_effect=OSError(f"raw evidence inspection {token}\t"),
        ):
            with self.assertRaises(ValidationError) as ctx:
                build_reflection(base_payload())
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\t", message)
        self.assertIn("filesystem operation failed", message)

    def test_evidence_directory_is_valid_but_symlink_and_hard_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            artifact_root = workspace / "repo"
            artifact_root.mkdir()
            (artifact_root / "runs" / "u1_full").mkdir(parents=True)
            secret = workspace / "secret.txt"
            secret.write_text("SECRET_TARGET_CONTENT\n", encoding="utf-8")
            link = artifact_root / "linked-evidence"
            try:
                link.symlink_to(secret)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            payload = base_payload()
            payload["evidence"] = ["runs/u1_full"]
            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", artifact_root):
                self.assertIn("runs/u1_full", build_reflection(payload))
            payload["evidence"] = ["linked-evidence"]
            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", artifact_root):
                with self.assertRaises(ValidationError) as ctx:
                    build_reflection(payload)
            self.assertNotIn("SECRET_TARGET_CONTENT", str(ctx.exception))

    def test_evidence_rejects_symlink_ancestor_of_repository_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            real_parent = workspace / "real-root"
            real_root = real_parent / "repo"
            real_root.mkdir(parents=True)
            (real_root / "artifact.txt").write_text("safe artifact\n", encoding="utf-8")
            linked_parent = workspace / "linked-root"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            payload = base_payload()
            payload["evidence"] = ["artifact.txt"]
            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", linked_parent / "repo"):
                with self.assertRaises(ValidationError) as ctx:
                    build_reflection(payload)
            self.assertNotIn("safe artifact", str(ctx.exception))

    def test_evidence_rejects_reparse_ancestor_of_repository_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            repository_root = workspace / "repo"
            repository_root.mkdir()
            (repository_root / "artifact.txt").write_text(
                "safe artifact\n", encoding="utf-8"
            )
            ancestor = repository_root.parent
            expected_ancestor = os.path.normcase(os.path.normpath(str(ancestor)))

            def attributes_for(path):
                normalized = os.path.normcase(os.path.normpath(str(path)))
                if normalized == expected_ancestor:
                    return (0x400, 0)
                return (0x10, 0)

            payload = base_payload()
            payload["evidence"] = ["artifact.txt"]
            with patch("scripts.closeout_reflection._WIN", True), patch(
                "scripts.closeout_reflection._windows_file_attributes",
                side_effect=attributes_for,
            ):
                with self.assertRaises(ValidationError) as ctx:
                    with patch(
                        "scripts.closeout_reflection.REPOSITORY_ROOT", repository_root
                    ):
                        build_reflection(payload)
            self.assertNotIn("safe artifact", str(ctx.exception))

    def test_evidence_hard_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            artifact_root = workspace / "repo"
            artifact_root.mkdir()
            owner = workspace / "owner.txt"
            owner.write_text("SECRET_TARGET_CONTENT\n", encoding="utf-8")
            linked = artifact_root / "hard-linked-evidence.txt"
            try:
                os.link(owner, linked)
            except OSError as exc:
                self.skipTest(f"hard-link creation unavailable: {exc}")
            payload = base_payload()
            payload["evidence"] = ["hard-linked-evidence.txt"]
            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", artifact_root):
                with self.assertRaises(ValidationError) as ctx:
                    build_reflection(payload)
            self.assertNotIn("SECRET_TARGET_CONTENT", str(ctx.exception))

    def test_missing_required_field_is_rejected(self):
        payload = base_payload()
        del payload["root_causes"]
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("root_causes", str(ctx.exception))

    def test_wrong_type_field_is_rejected(self):
        payload = base_payload()
        payload["changed_files"] = "scripts/closeout_reflection.py"
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("changed_files", str(ctx.exception))

    def test_embedded_control_character_is_rejected(self):
        payload = base_payload()
        payload["actual_outcome"] = "before\x00after"
        with self.assertRaises(ValidationError) as ctx:
            build_reflection(payload)
        self.assertIn("control", str(ctx.exception).lower())

    def test_embedded_c1_control_character_is_rejected(self):
        payload = base_payload()
        payload["actual_outcome"] = "before\x85after"
        with self.assertRaises(ValidationError) as ctx:
            build_reflection(payload)
        self.assertIn("control", str(ctx.exception).lower())

    def test_empty_string_field_is_rejected(self):
        payload = base_payload()
        payload["actual_outcome"] = "   "
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("actual_outcome", str(ctx.exception))

    def test_non_string_list_item_is_rejected(self):
        payload = base_payload()
        payload["open_questions"] = [1]
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("open_questions", str(ctx.exception))

    def test_oversized_list_is_rejected(self):
        payload = base_payload()
        payload["changed_files"] = [f"file_{i}.py" for i in range(101)]
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("changed_files", str(ctx.exception))

    def test_payload_is_not_mutated(self):
        payload = base_payload()
        original = json.dumps(payload, sort_keys=True)
        build_reflection(payload)
        self.assertEqual(json.dumps(payload, sort_keys=True), original)


class DiagnosticSafetyTests(unittest.TestCase):
    def test_diagnostic_helpers_bound_preserve_and_redact_diagnostics(self):
        safe_path = r"C:\ordinary\reflection.md"
        self.assertEqual(_safe_diagnostic(safe_path), safe_path)
        self.assertEqual(_safe_diagnostic("value", limit=0), "")
        self.assertEqual(_safe_diagnostic("value", limit=1), "v")
        bounded = _safe_diagnostic("x" * 241)
        self.assertLessEqual(len(bounded), 240)
        self.assertIn("…", bounded)

        for name, token in EXTENDED_SECRET_SAMPLES.items():
            with self.subTest(pattern=name):
                self.assertEqual(_safe_diagnostic(token), "<redacted diagnostic>")
        for value in (
            "\ud800",
            "before\t" + "after",
            "before\n" + "after",
            "before\r" + "after",
            "before" + chr(0xE0000) + "after",
            "before" + chr(0xE0020) + "after",
            "before" + chr(0xE007F) + "after",
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(_safe_diagnostic(value), "<redacted diagnostic>")

    def test_output_path_resolution_suppresses_raw_exception_cause(self):
        token = "github_pat_" + "Z" * 30
        with patch(
            "scripts.closeout_reflection.os.path.realpath",
            side_effect=OSError(f"raw resolution {token}\rC:/private/output.md"),
        ):
            with self.assertRaises(ProtectedPathError) as ctx:
                _validate_output_path(Path("docs/reflections/cause.md"))
        self.assertIsNone(ctx.exception.__cause__)
        rendered = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(token, rendered)
        self.assertNotIn("C:/private/output.md", rendered)

    def test_exception_diagnostics_are_bounded_and_sanitized(self):
        self.assertEqual(
            _safe_exception_detail(ValueError("ordinary safe detail")),
            "ordinary safe detail",
        )
        for value in (
            OSError("raw \t filesystem detail"),
            ValueError("sk-" + "T" * 24),
            ValueError("before" + chr(0xE0020) + "after"),
            ValueError("\ud800"),
            ValueError("x" * 241),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(
                    _safe_exception_detail(value),
                    "filesystem operation failed",
                )


class SecretDetectionTests(unittest.TestCase):
    def test_assignment_pattern_is_rejected(self):
        payload = base_payload()
        payload["requested_outcome"] = "connect with password=hunter2 please"
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("secret", str(ctx.exception).lower())
        self.assertNotIn("hunter2", str(ctx.exception))

    def test_extended_token_patterns_are_rejected_in_payload(self):
        for name, token in EXTENDED_SECRET_SAMPLES.items():
            with self.subTest(pattern=name):
                payload = base_payload()
                payload["requested_outcome"] = token
                with self.assertRaises(ValueError) as ctx:
                    build_reflection(payload)
                self.assertNotIn(token, str(ctx.exception))

    def test_extended_token_patterns_are_rejected_in_persisted_state(self):
        for name, token in EXTENDED_SECRET_SAMPLES.items():
            with self.subTest(pattern=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    state_path = Path(temp_dir) / "lessons_pending.json"
                    state = seed_state()
                    state["lessons"][0]["lesson"] = token
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                    before = state_path.read_bytes()
                    with self.assertRaises(ValueError) as ctx:
                        stage_lesson_candidate(base_payload(), state_path)
                    self.assertNotIn(token, str(ctx.exception))
                    self.assertEqual(state_path.read_bytes(), before)

    def test_unicode_tag_boundaries_and_interior_are_rejected_in_state(self):
        for codepoint in (0xE0000, 0xE0020, 0xE007F):
            with self.subTest(codepoint=codepoint):
                with tempfile.TemporaryDirectory() as temp_dir:
                    state_path = Path(temp_dir) / "lessons_pending.json"
                    state = seed_state()
                    state["lessons"][0]["lesson"] = (
                        f"before{chr(codepoint)}after"
                    )
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                    before = state_path.read_bytes()
                    with self.assertRaises(ValueError) as ctx:
                        stage_lesson_candidate(base_payload(), state_path)
                    message = str(ctx.exception)
                    self.assertNotIn(chr(codepoint), message)
                    self.assertNotIn("before", message)
                    self.assertEqual(state_path.read_bytes(), before)

    def test_direct_write_reflection_rejects_secret_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reflection.md"
            token = "github_pat_" + "K" * 30
            with self.assertRaises(ValueError):
                write_reflection(output, token, replace=False)
            self.assertFalse(output.exists())

    def test_direct_write_reflection_rejects_secret_with_replace_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reflection.md"
            original = b"KEEP EXISTING CONTENT\n"
            output.write_bytes(original)
            token = "github_pat_" + "K" * 30
            with self.assertRaises(ValueError):
                write_reflection(output, token, replace=True)
            self.assertEqual(output.read_bytes(), original)

    def test_direct_write_reflection_rejects_each_extended_token_with_replace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (name, token) in enumerate(EXTENDED_SECRET_SAMPLES.items()):
                with self.subTest(pattern=name):
                    output = Path(temp_dir) / f"reflection-{index}.md"
                    original = f"KEEP {index}\n".encode("utf-8")
                    output.write_bytes(original)
                    with self.assertRaises(ValueError):
                        write_reflection(output, token, replace=True)
                    self.assertEqual(output.read_bytes(), original)

    def test_unicode_format_controls_are_rejected(self):
        for codepoint in (
            0x00AD, 0x061C, 0x0600, 0x180E, 0x200B, 0x2028, 0x2029,
            0x202E, 0x2066, 0xFFF9, 0xE0000, 0xE0001, 0xE007F,
        ):
            with self.subTest(codepoint=codepoint):
                payload = base_payload()
                payload["actual_outcome"] = f"before{chr(codepoint)}after"
                with self.assertRaises(ValidationError):
                    build_reflection(payload)

    def test_private_key_pattern_is_rejected(self):
        payload = base_payload()
        payload["root_causes"] = ["pasted -----BEGIN PRIVATE KEY----- into notes"]
        with self.assertRaises(ValueError):
            build_reflection(payload)

    def test_complete_private_key_block_is_rejected(self):
        payload = base_payload()
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("M" * 64)
            + "\n-----END PRIVATE KEY-----"
        )
        payload["root_causes"] = [private_key]
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertNotIn("M" * 64, str(ctx.exception))

    def test_optional_string_field_secret_is_rejected_without_echo(self):
        payload = base_payload()
        payload["scope"] = "token=supersecretvalue"
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertIn("secret", str(ctx.exception).lower())
        self.assertNotIn("supersecretvalue", str(ctx.exception))

    def test_optional_list_item_secret_is_rejected_without_echo(self):
        payload = base_payload()
        payload["open_questions"] = ["why is api_key=abcdef stored in configs?"]
        with self.assertRaises(ValueError) as ctx:
            build_reflection(payload)
        self.assertNotIn("abcdef", str(ctx.exception))

    def test_cli_reports_secret_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bad_input = temp / "bad.json"
            bad_input.write_text(
                json.dumps({**base_payload(), "actual_outcome": "token=abc123"}),
                encoding="utf-8",
            )
            state_path = temp / "state.json"
            state_path.write_text(json.dumps(seed_state()), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(bad_input),
                        "--write",
                        "--output",
                        str(temp / "out.md"),
                        "--stage-lesson",
                        "--state-path",
                        str(state_path),
                    ]
                )
            self.assertNotEqual(code, 0)
            self.assertFalse((temp / "out.md").exists())
            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), seed_state())
            self.assertNotIn("abc123", stderr.getvalue())


class WriteReflectionTests(unittest.TestCase):
    def test_writes_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "reflection.md"
            write_reflection(target, "# Reflection\n", replace=False)
            self.assertEqual(target.read_text(encoding="utf-8"), "# Reflection\n")

    def test_rejects_unencodable_content_before_touching_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            target.write_text("ORIGINAL\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                write_reflection(target, "\ud800", replace=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "ORIGINAL\n")

    def test_rejects_embedded_control_content_before_touching_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            target.write_text("ORIGINAL\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                write_reflection(target, "replacement\x00", replace=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "ORIGINAL\n")

    def test_refuses_overwrite_without_replace_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            write_reflection(target, "original", replace=False)
            with self.assertRaises(ValueError):
                write_reflection(target, "replacement", replace=False)
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            write_reflection(target, "replacement", replace=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "replacement")

    def test_refuses_protected_policy_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            for name in ("SOUL.md", "AGENTS.md", "CLAUDE.md"):
                target = temp / name
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "x", replace=False)
                self.assertFalse(target.exists())
            rules_dir = temp / "rules" / "components"
            for name in ("gpu.md", "data.md", "data.txt"):
                target = rules_dir / name
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "x", replace=False)
                self.assertFalse(target.exists())
            # --replace must not bypass the protected-path refusal either.
            existing = temp / "rules" / "evidence.md"
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_text("policy\n", encoding="utf-8")
            with self.assertRaises(ProtectedPathError):
                write_reflection(existing, "overwritten", replace=True)
            self.assertEqual(existing.read_text(encoding="utf-8"), "policy\n")

    def test_refuses_windows_protected_lexical_aliases(self):
        if sys.platform != "win32":
            self.skipTest("Windows lexical path coverage is Windows-only")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            targets = (
                temp / "rules." / "reflection.md",
                temp / "rules.." / "reflection.md",
                temp / "rules..." / "reflection.md",
                temp / "rules " / "reflection.md",
                temp / "rules. " / "reflection.md",
                temp / "SOUL.md.",
                temp / "CLAUDE.md ",
            )
            for target in targets:
                with self.subTest(path=target):
                    with self.assertRaises(ProtectedPathError):
                        write_reflection(target, "SHOULD NOT WRITE", replace=True)
                    self.assertFalse(target.exists())
            self.assertFalse((temp / "rules").exists())

    def test_refuses_symlink_escape_to_protected_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            victim_dir = temp / "victim"
            victim_dir.mkdir()
            victim = victim_dir / "SOUL.md"
            victim.write_text("ORIGINAL\n", encoding="utf-8")
            link = temp / "reflection.md"
            try:
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["cmd", "/c", "mklink", str(link), str(victim)],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        self.skipTest(f"file symlinks unavailable: {result.stderr}")
                else:
                    link.symlink_to(victim)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            if not link.is_symlink():
                self.skipTest("symlink creation did not produce a link")
            with self.subTest(operation="reflection output"):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(link, "OVERWRITTEN\n", replace=True)
            with self.subTest(operation="lesson state"):
                with self.assertRaises(ProtectedPathError):
                    stage_lesson_candidate(base_payload(), link)
            with self.subTest(operation="cli validates before writing"):
                output = temp / "cli-reflection.md"
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output),
                        "--stage-lesson",
                        "--state-path",
                        str(link),
                    ]
                )
                self.assertNotEqual(code, 0)
                self.assertFalse(output.exists())
            self.assertEqual(victim.read_text(encoding="utf-8"), "ORIGINAL\n")

    def test_refuses_junction_components_including_dangling_target(self):
        if sys.platform != "win32":
            self.skipTest("junction coverage is Windows-only")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            live_target = temp / "live-target"
            live_target.mkdir()
            live_junction = temp / "live-junction"
            dangling_target = temp / "dangling-target"
            dangling_junction = temp / "dangling-junction"
            for junction, target in (
                (live_junction, live_target),
                (dangling_junction, dangling_target),
            ):
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    self.skipTest(f"junctions unavailable: {result.stderr}")
                if not os.path.isjunction(junction):
                    self.skipTest("mklink did not produce a junction")
                output = junction / "reflection.md"
                with self.subTest(junction=target.name):
                    with self.assertRaises(ProtectedPathError):
                        write_reflection(output, "OVERWRITTEN\n", replace=True)
                    self.assertFalse(output.exists())

    def test_refuses_indeterminate_reparse_query_on_existing_ancestor(self):
        if sys.platform != "win32":
            self.skipTest("reparse query error coverage is Windows-only")
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_parent = Path(temp_dir) / "existing"
            existing_parent.mkdir()
            target = existing_parent / "reflection.md"
            expected_parent = os.path.normcase(os.path.normpath(str(existing_parent)))

            def attributes_for(path):
                normalized = os.path.normcase(os.path.normpath(str(path)))
                if normalized == expected_parent:
                    return 0xFFFFFFFF
                return 0x10

            with patch(
                "ctypes.windll.kernel32.GetFileAttributesW",
                side_effect=attributes_for,
            ), patch(
                "ctypes.windll.kernel32.GetLastError", return_value=5
            ):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "SHOULD NOT WRITE", replace=True)
            self.assertFalse(target.exists())

    def test_refuses_indeterminate_posix_symlink_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch("scripts.closeout_reflection._WIN", False), patch(
                "scripts.closeout_reflection.os.lstat",
                side_effect=OSError("simulated lstat denial"),
            ):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "SHOULD NOT WRITE", replace=True)
            self.assertFalse(target.exists())

    def test_refuses_ntfs_alternate_data_stream_paths(self):
        if sys.platform != "win32":
            self.skipTest("alternate-stream coverage is Windows-only")
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir) / "owner.txt"
            owner.write_text("ORIGINAL\n", encoding="utf-8")
            stream = Path(str(owner) + ":reflection")
            with self.assertRaises(ProtectedPathError):
                write_reflection(stream, "STREAM", replace=True)
            with self.assertRaises(ProtectedPathError):
                stage_lesson_candidate(base_payload(), stream)
            self.assertEqual(owner.read_text(encoding="utf-8"), "ORIGINAL\n")

    def test_refuses_when_output_metadata_cannot_be_inspected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection.os.stat",
                side_effect=PermissionError("simulated metadata denial"),
            ):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "SHOULD NOT WRITE", replace=True)
            self.assertFalse(target.exists())

    def test_refuses_when_target_resolution_is_indeterminate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection.os.path.realpath",
                side_effect=OSError("simulated target-resolution denial"),
            ):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "SHOULD NOT WRITE", replace=True)
            self.assertFalse(target.exists())

    def test_direct_write_redacts_path_resolution_error_detail(self):
        token = "sk-" + "W" * 24
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection.os.path.abspath",
                side_effect=OSError(f"raw path resolution {token}\t"),
            ):
                with self.assertRaises((ProtectedPathError, OSError)) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        self.assertIsInstance(ctx.exception, ProtectedPathError)
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\t", message)
        self.assertIn(str(target), message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_write_redacts_unexpected_preflight_error(self):
        token = "glpat-" + "R" * 24
        raw_path = "C:/private/customer/preflight.txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection._validate_output_path",
                side_effect=RuntimeError(
                    f"raw preflight failure {token} {raw_path}\n"
                ),
            ):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn(raw_path, message)
        self.assertNotIn("\n", message)
        self.assertIn("filesystem operation failed", message)
        self.assertIn(str(target), message)

    def test_direct_write_redacts_unexpected_write_error(self):
        token = "glpat-" + "S" * 24
        raw_path = "C:/private/customer/write.txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"

            def failing_write(path, data, **kwargs):
                raise RuntimeError(f"raw write failure {token} {raw_path}\r")

            with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn(raw_path, message)
        self.assertNotIn("\r", message)
        self.assertIn("filesystem operation failed", message)
        self.assertIn(str(target), message)

    def test_refuses_when_output_existence_check_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            real_stat = os.stat
            calls = 0

            def stat_with_existence_denial(path, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise PermissionError("simulated existence denial")
                return real_stat(path, *args, **kwargs)

            with patch(
                "scripts.closeout_reflection.os.stat",
                side_effect=stat_with_existence_denial,
            ):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(target, "SHOULD NOT WRITE", replace=True)
            self.assertEqual(calls, 2)
            self.assertFalse(target.exists())

    def test_rejects_embedded_nul_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection\0.md"
            with self.assertRaises(ValidationError):
                write_reflection(target, "SHOULD NOT WRITE", replace=True)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(target),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("path", stderr.getvalue().lower())
            self.assertFalse((Path(temp_dir) / "reflection.md").exists())

    def test_invalid_destination_path_redacts_secret_and_control_characters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            token = "sk-" + "Z" * 24
            target = Path(temp_dir) / f"{token}\x00.md"
            with self.assertRaises(ValidationError) as ctx:
                write_reflection(target, "SHOULD NOT WRITE", replace=True)
            message = str(ctx.exception)
            self.assertNotIn(token, message)
            self.assertNotIn("\x00", message)
            self.assertNotIn("\\x00", message)
            self.assertIn("<redacted diagnostic>", message)
            self.assertFalse((Path(temp_dir) / f"{token}.md").exists())

    def test_direct_write_redacts_unsafe_destination_filename(self):
        tag = chr(0xE0020)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / f"reflection-{tag}.md"
            with patch(
                "scripts.closeout_reflection._path_exists",
                side_effect=OSError("raw existence failure"),
            ):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertNotIn(tag, message)
        self.assertIn("<redacted diagnostic>", message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_write_redacts_destination_preflight_filesystem_error(self):
        token = "sk-" + "O" * 24
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection._validate_output_path",
                side_effect=OSError(f"raw destination preflight {token}\n"),
            ):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertIn(str(target), message)
        self.assertIn("filesystem operation failed", message)
        self.assertNotIn(token, message)
        self.assertNotIn("\n", message)

    def test_direct_write_redacts_existence_check_filesystem_error(self):
        token = "sk-" + "P" * 24
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"
            with patch(
                "scripts.closeout_reflection._path_exists",
                side_effect=OSError(f"raw existence check {token}\r"),
            ):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertIn(str(target), message)
        self.assertIn("filesystem operation failed", message)
        self.assertNotIn(token, message)
        self.assertNotIn("\r", message)

    def test_direct_write_redacts_filesystem_error_detail(self):
        token = "sk-" + "T" * 24
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection.md"

            def failing_write(path, data, **kwargs):
                raise OSError(f"raw write failure {token}\n")

            with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
                with self.assertRaises((ProtectedPathError, OSError)) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
            self.assertIsInstance(ctx.exception, ProtectedPathError)

            message = str(ctx.exception)
            self.assertNotIn(token, message)
            self.assertNotIn("\n", message)
            self.assertIn(str(target), message)
            self.assertIn("filesystem operation failed", message)

    def test_direct_write_does_not_echo_safe_exception_path_detail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "private.txt"

            def failing_write(path, data, **kwargs):
                raise OSError(f"Permission denied: {path}")

            with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
                with self.assertRaises(ProtectedPathError) as ctx:
                    write_reflection(target, "SAFE REFLECTION\n")
        message = str(ctx.exception)
        self.assertIn(str(target), message)
        self.assertNotIn("Permission denied:", message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_write_redacts_unknown_validation_path_detail(self):
        raw_path = "C:/private/customer/secret.txt"
        token = "github_pat_" + "Q" * 30
        cases = (
            ("path-only", f"failed path {raw_path}", (raw_path,)),
            ("secret-and-control", f"failed path {raw_path}\r{token}", (raw_path, token, "\r")),
        )
        for error_type in (ValidationError, ProtectedPathError):
            for case_name, detail, forbidden in cases:
                with self.subTest(error_type=error_type.__name__, case=case_name):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        target = Path(temp_dir) / "requested.md"
                        with patch(
                            "scripts.closeout_reflection._validate_output_path",
                            side_effect=error_type(detail),
                        ):
                            with self.assertRaises(error_type) as ctx:
                                write_reflection(target, "SAFE REFLECTION\n")
                    message = str(ctx.exception)
                    rendered = "".join(traceback.format_exception(ctx.exception))
                    for value in forbidden:
                        self.assertNotIn(value, message)
                        self.assertNotIn(value, rendered)
                    self.assertIsNone(ctx.exception.__cause__)
                    self.assertIn(str(target), message)
                    self.assertIn("filesystem operation failed", message)

    def test_refuses_existing_multi_link_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            owner = temp / "owner.txt"
            original = json.dumps(seed_state(), indent=2) + "\n"
            owner.write_text(original, encoding="utf-8")
            alias = temp / "owner-alias.txt"
            try:
                os.link(owner, alias)
            except OSError as exc:
                self.skipTest(f"hard-link creation unavailable: {exc}")
            with self.subTest(operation="reflection output"):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(alias, "REPLACED", replace=True)
            with self.subTest(operation="lesson state"):
                with self.assertRaises(ProtectedPathError):
                    stage_lesson_candidate(base_payload(), alias)
            self.assertEqual(owner.read_text(encoding="utf-8"), original)


    def test_refuses_existing_non_regular_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reflection-directory"
            target.mkdir()
            with self.assertRaises(ProtectedPathError):
                write_reflection(target, "REPLACED", replace=True)
            self.assertTrue(target.is_dir())


    def test_posix_case_distinct_paths_are_not_treated_as_collisions(self):
        with patch("scripts.closeout_reflection._WIN", False), patch(
            "scripts.closeout_reflection.os.path.normcase",
            side_effect=lambda value: value,
        ), patch(
            "scripts.closeout_reflection.os.path.samefile",
            side_effect=OSError("not available"),
        ):
            self.assertFalse(
                _same_destination(Path("Reflection.md"), Path("reflection.md"))
            )


class StageLessonCandidateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "lessons_pending.json"
        self.state_path.write_text(json.dumps(seed_state(), indent=2), encoding="utf-8")
        self.policy_files = {}
        for rel in ("SOUL.md", "AGENTS.md", "CLAUDE.md"):
            path = Path(self._tmp.name) / rel
            path.write_text("policy content\n", encoding="utf-8")
            self.policy_files[path] = "policy content\n"
        rules_dir = Path(self._tmp.name) / "rules"
        rules_dir.mkdir()
        rule = rules_dir / "gpu.md"
        rule.write_text("rule content\n", encoding="utf-8")
        self.policy_files[rule] = "rule content\n"

    def test_appends_candidate_with_design_fields(self):
        candidate = stage_lesson_candidate(base_payload(), self.state_path)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 2)
        self.assertEqual(state["lessons"][-1], candidate)
        for field in (
            "id",
            "scope",
            "status",
            "lesson",
            "source_reflection",
            "evidence",
            "promoted_to",
        ):
            self.assertIn(field, candidate)
        self.assertEqual(candidate["status"], "candidate")
        self.assertTrue(candidate["lesson"])
        self.assertEqual(candidate["promoted_to"], [])
        self.assertIn(self.state_path.name, "lessons_pending.json")

    def test_direct_staging_redacts_preflight_filesystem_error(self):
        token = "sk-" + "Q" * 24
        with patch(
            "scripts.closeout_reflection._prepare_lesson_candidate",
            side_effect=OSError(f"raw state preflight {token}\t"),
        ):
            with self.assertRaises(ProtectedPathError) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        message = str(ctx.exception)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)
        self.assertNotIn(token, message)
        self.assertNotIn("\t", message)

    def test_direct_staging_redacts_unexpected_preflight_error(self):
        token = "github_pat_" + "T" * 30
        raw_path = "C:/private/customer/state.json"
        with patch(
            "scripts.closeout_reflection._prepare_lesson_candidate",
            side_effect=RuntimeError(
                f"raw state preflight {token} {raw_path}\n"
            ),
        ):
            with self.assertRaises(ProtectedPathError) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn(raw_path, message)
        self.assertNotIn("\n", message)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_staging_redacts_unexpected_write_error(self):
        token = "github_pat_" + "U" * 30
        raw_path = "C:/private/customer/state.json"

        def failing_write(path, data, **kwargs):
            raise RuntimeError(f"raw state write failure {token} {raw_path}\r")

        with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
            with self.assertRaises(ProtectedPathError) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn(raw_path, message)
        self.assertNotIn("\r", message)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_staging_redacts_filesystem_error_detail(self):
        token = "sk-" + "U" * 24

        def failing_write(path, data, **kwargs):
            raise OSError(f"raw state write failure {token}\n")

        with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
            with self.assertRaises((ProtectedPathError, OSError)) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        self.assertIsInstance(ctx.exception, ProtectedPathError)

        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\n", message)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_staging_redacts_state_read_error_detail(self):
        token = "sk-" + "V" * 24

        def failing_read(path, **kwargs):
            raise OSError(f"raw state read failure {token}\n")

        with patch.object(Path, "read_text", autospec=True, side_effect=failing_read):
            with self.assertRaises((ValidationError, OSError)) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        self.assertIsInstance(ctx.exception, ValidationError)

        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\n", message)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)

    def test_direct_staging_redacts_destination_identity_error_detail(self):
        token = "sk-" + "I" * 24
        with patch(
            "scripts.closeout_reflection._same_destination",
            side_effect=OSError(f"raw identity check {token}\t"),
        ):
            with self.assertRaises((ProtectedPathError, ValidationError, OSError)) as ctx:
                stage_lesson_candidate(base_payload(), self.state_path)
        self.assertIsInstance(ctx.exception, (ProtectedPathError, ValidationError))
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("\t", message)
        self.assertIn(str(self.state_path), message)
        self.assertIn("filesystem operation failed", message)

    def test_staging_is_idempotent(self):
        first = stage_lesson_candidate(base_payload(), self.state_path)
        before = self.state_path.read_text(encoding="utf-8")
        second = stage_lesson_candidate(base_payload(), self.state_path)
        self.assertEqual(first, second)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 2)
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)

    def test_direct_staging_rejects_state_as_evidence_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = Path(temp_dir) / "repo"
            state_path = repository / "state" / "lessons_pending.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(json.dumps(seed_state(), indent=2), encoding="utf-8")
            before = state_path.read_bytes()
            payload = base_payload()
            payload["evidence"] = ["state/lessons_pending.json"]
            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repository):
                with self.assertRaises(ValidationError):
                    stage_lesson_candidate(payload, state_path)
            self.assertEqual(state_path.read_bytes(), before)

    def test_legacy_three_field_id_is_reused(self):
        payload = base_payload()
        fields = {
            "scope": "core",
            "source_reflection": "docs/reflections/example.md",
            "reusable_lesson": payload["reusable_lesson"],
        }
        legacy_digest = hashlib.sha256(
            json.dumps(
                [fields["scope"], fields["source_reflection"], fields["reusable_lesson"]],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        legacy = {
            "id": f"lesson-core-{legacy_digest}",
            "scope": "core",
            "status": "candidate",
            "lesson": fields["reusable_lesson"],
            "source_reflection": fields["source_reflection"],
            "evidence": list(payload["evidence"]),
            "promoted_to": [],
        }
        state = {"schema_version": 1, "lessons": [legacy]}
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_text(encoding="utf-8")
        candidate = stage_lesson_candidate(payload, self.state_path)
        self.assertEqual(candidate["id"], legacy["id"])
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)

    def test_legacy_id_reuses_when_all_persisted_facts_match(self):
        payload = base_payload()
        legacy_digest = hashlib.sha256(
            json.dumps(
                [payload["scope"], payload["source_reflection"], payload["reusable_lesson"]],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        legacy = {
            "id": f"lesson-core-{legacy_digest}",
            "scope": payload["scope"],
            "status": "candidate",
            "lesson": payload["reusable_lesson"],
            "source_reflection": payload["source_reflection"],
            "rule_candidate": payload["rule_candidate"],
            "evidence": list(payload["evidence"]),
            "promoted_to": [],
        }
        state = {"schema_version": 1, "lessons": [legacy]}
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_text(encoding="utf-8")
        candidate = stage_lesson_candidate(payload, self.state_path)
        self.assertEqual(candidate["id"], legacy["id"])
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)

    def test_legacy_id_with_changed_rule_candidate_creates_new_entry(self):
        payload = base_payload()
        legacy_digest = hashlib.sha256(
            json.dumps(
                [payload["scope"], payload["source_reflection"], payload["reusable_lesson"]],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        legacy = {
            "id": f"lesson-core-{legacy_digest}",
            "scope": payload["scope"],
            "status": "candidate",
            "lesson": payload["reusable_lesson"],
            "source_reflection": payload["source_reflection"],
            "rule_candidate": "historical rule candidate",
            "evidence": list(payload["evidence"]),
            "promoted_to": [],
        }
        self.state_path.write_text(
            json.dumps({"schema_version": 1, "lessons": [legacy]}), encoding="utf-8"
        )
        candidate = stage_lesson_candidate(payload, self.state_path)
        self.assertNotEqual(candidate["id"], legacy["id"])
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 2)
        self.assertEqual(state["lessons"][0], legacy)
        self.assertEqual(state["lessons"][1], candidate)

    def test_legacy_id_with_changed_evidence_creates_new_entry(self):
        payload = base_payload()
        legacy_digest = hashlib.sha256(
            json.dumps(
                [payload["scope"], payload["source_reflection"], payload["reusable_lesson"]],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        legacy = {
            "id": f"lesson-core-{legacy_digest}",
            "scope": payload["scope"],
            "status": "candidate",
            "lesson": payload["reusable_lesson"],
            "source_reflection": payload["source_reflection"],
            "rule_candidate": payload["rule_candidate"],
            "evidence": ["tests/agent_context/test_context_proposal.py"],
            "promoted_to": [],
        }
        self.state_path.write_text(
            json.dumps({"schema_version": 1, "lessons": [legacy]}), encoding="utf-8"
        )
        changed = base_payload()
        changed["evidence"] = ["tests/agent_context/test_check_agent_context.py"]
        candidate = stage_lesson_candidate(changed, self.state_path)
        self.assertNotEqual(candidate["id"], legacy["id"])
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 2)
        self.assertEqual(state["lessons"][1]["evidence"], changed["evidence"])

    def test_state_schema_and_nan_are_rejected_before_write(self):
        cases = (
            {"schema_version": True},
            {"schema_version": 1.0},
            {"schema_version": "1"},
            {"schema_version": 2},
        )
        for override in cases:
            with self.subTest(override=override):
                state = seed_state()
                state.update(override)
                self.state_path.write_text(json.dumps(state), encoding="utf-8")
                before = self.state_path.read_bytes()
                with self.assertRaises(ValidationError):
                    stage_lesson_candidate(base_payload(), self.state_path)
                self.assertEqual(self.state_path.read_bytes(), before)

        malformed_fields = (
            ("id", 1),
            ("scope", []),
            ("status", "unknown"),
            ("lesson", {}),
            ("source_reflection", 1),
            ("evidence", "old/evidence.txt"),
            ("promoted_to", "rules/gpu.md"),
        )
        for field, value in malformed_fields:
            with self.subTest(field=field):
                state = seed_state()
                state["lessons"][0][field] = value
                self.state_path.write_text(json.dumps(state), encoding="utf-8")
                before = self.state_path.read_bytes()
                with self.assertRaises(ValidationError):
                    stage_lesson_candidate(base_payload(), self.state_path)
                self.assertEqual(self.state_path.read_bytes(), before)

        self.state_path.write_text(
            '{"schema_version": 1, "lessons": [], "extra": NaN}\n',
            encoding="utf-8",
        )
        before = self.state_path.read_bytes()
        with self.assertRaises(ValidationError):
            stage_lesson_candidate(base_payload(), self.state_path)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_duplicate_persisted_identity_with_unknown_legacy_rule_is_rejected(self):
        payload = base_payload()
        legacy_digest = hashlib.sha256(
            json.dumps(
                [payload["scope"], payload["source_reflection"], payload["reusable_lesson"]],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        legacy = {
            "id": f"lesson-core-{legacy_digest}",
            "scope": payload["scope"],
            "status": "candidate",
            "lesson": payload["reusable_lesson"],
            "source_reflection": payload["source_reflection"],
            "evidence": list(payload["evidence"]),
            "promoted_to": [],
        }
        current = dict(legacy)
        current["id"] = "lesson-core-current"
        current["rule_candidate"] = payload["rule_candidate"]
        self.state_path.write_text(
            json.dumps({"schema_version": 1, "lessons": [legacy, current]}),
            encoding="utf-8",
        )
        before = self.state_path.read_bytes()
        with self.assertRaises(ValidationError):
            stage_lesson_candidate(payload, self.state_path)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_duplicate_persisted_id_diagnostic_is_bounded(self):
        state = seed_state()
        long_id = "id-" + ("x" * 300)
        first = state["lessons"][0]
        second = dict(first)
        first["id"] = long_id
        second["id"] = long_id
        state["lessons"] = [first, second]
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()

        with self.assertRaises(ValidationError) as ctx:
            stage_lesson_candidate(base_payload(), self.state_path)

        message = str(ctx.exception)
        self.assertNotIn(long_id, message)
        self.assertIn("id-", message)
        self.assertIn("…", message)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_duplicate_persisted_identity_diagnostic_is_bounded(self):
        state = seed_state()
        first_id = "id-" + ("x" * 300)
        second_id = "id-" + ("y" * 300)
        first = state["lessons"][0]
        second = dict(first)
        first["id"] = first_id
        second["id"] = second_id
        state["lessons"] = [first, second]
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()

        with self.assertRaises(ValidationError) as ctx:
            stage_lesson_candidate(base_payload(), self.state_path)

        message = str(ctx.exception)
        self.assertLessEqual(len(message), 240)
        self.assertNotIn(first_id, message)
        self.assertNotIn(second_id, message)
        self.assertIn("id-", message)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_old_evidence_paths_need_not_exist(self):
        state = seed_state()
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        candidate = stage_lesson_candidate(base_payload(), self.state_path)
        self.assertEqual(candidate["status"], "candidate")

    def test_invalid_persisted_evidence_fails_before_state_rewrite(self):
        state = seed_state()
        state["lessons"][0]["evidence"] = ["../../outside.txt"]
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()

        with self.assertRaises(ValidationError):
            stage_lesson_candidate(base_payload(), self.state_path)

        self.assertEqual(self.state_path.read_bytes(), before)

    def test_persisted_control_character_fails_before_state_rewrite(self):
        state = seed_state()
        state["lessons"][0]["lesson"] = "unsafe\x00value"
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()

        with self.assertRaises(ValidationError) as ctx:
            stage_lesson_candidate(base_payload(), self.state_path)

        self.assertIn("control", str(ctx.exception).lower())
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_persisted_unknown_secret_fails_without_echo_or_write(self):
        state = seed_state()
        state["lessons"][0]["extra"] = {"notes": ["api_key=supersecretvalue"]}
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()

        with self.assertRaises(ValueError) as ctx:
            stage_lesson_candidate(base_payload(), self.state_path)

        self.assertIn("secret", str(ctx.exception).lower())
        self.assertNotIn("supersecretvalue", str(ctx.exception))
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_changed_rule_candidate_gets_distinct_id_and_entry(self):
        first = stage_lesson_candidate(base_payload(), self.state_path)
        changed_payload = base_payload()
        changed_payload["rule_candidate"] = "closeout: reject unsafe paths"
        second = stage_lesson_candidate(changed_payload, self.state_path)

        self.assertNotEqual(first["id"], second["id"])
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 3)
        self.assertEqual(state["lessons"][1], first)
        self.assertEqual(state["lessons"][2], second)

    def test_preserves_existing_entries_and_comment(self):
        raw = self.state_path.read_text(encoding="utf-8")
        raw = raw.replace("{", '{\n  "_comment": "Staging area for lesson candidates.",', 1)
        self.state_path.write_text(raw, encoding="utf-8")
        stage_lesson_candidate(base_payload(), self.state_path)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["lessons"][0], seed_state()["lessons"][0])
        self.assertIn("Staging area", state["_comment"])

    def test_empty_lesson_is_rejected(self):
        payload = base_payload()
        payload["reusable_lesson"] = ""
        with self.assertRaises(ValueError) as ctx:
            stage_lesson_candidate(payload, self.state_path)
        self.assertIn("lesson", str(ctx.exception).lower())

    def test_no_policy_or_persona_file_is_touched(self):
        stage_lesson_candidate(base_payload(), self.state_path)
        for path, content in self.policy_files.items():
            self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_refuses_protected_state_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            for rel in (
                "SOUL.md",
                "AGENTS.md",
                "CLAUDE.md",
                Path("rules") / "gpu.md",
                Path("rules") / "data.txt",
            ):
                target = temp / rel
                with self.assertRaises(ProtectedPathError):
                    stage_lesson_candidate(base_payload(), target)
                self.assertFalse(target.exists())

    def test_never_promotes_candidate(self):
        candidate = stage_lesson_candidate(base_payload(), self.state_path)
        self.assertEqual(candidate["status"], "candidate")
        self.assertEqual(candidate["promoted_to"], [])
        rules_dir = Path(self._tmp.name) / "rules"
        self.assertEqual((rules_dir / "gpu.md").read_text(encoding="utf-8"), "rule content\n")


class MainCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.temp = Path(self._tmp.name)
        self.state_path = self.temp / "lessons_pending.json"
        self.state_path.write_text(json.dumps(seed_state(), indent=2), encoding="utf-8")

    def test_dry_run_prints_draft_and_writes_nothing(self):
        before = sorted(p.name for p in self.temp.iterdir())
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = main(["--input", str(SAMPLE)])
        self.assertEqual(code, 0)
        draft = captured.getvalue()
        self.assertIn("Requested outcome", draft)
        after = sorted(p.name for p in self.temp.iterdir())
        self.assertEqual(after, before)
        self.assertEqual(
            json.loads(self.state_path.read_text(encoding="utf-8")), seed_state()
        )

    def test_dry_run_redacts_unicode_tag_from_invalid_payload(self):
        tag = chr(0xE0020)
        payload = base_payload()
        payload["actual_outcome"] = f"before{tag}after"
        input_path = self.temp / "tagged-input.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--input", str(input_path)])
        self.assertEqual(code, 2)
        self.assertNotIn(tag, stdout.getvalue())
        self.assertNotIn(tag, stderr.getvalue())
        self.assertNotIn("before", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_complete_private_key_from_exception_detail(self):
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("M" * 64)
            + "\n-----END PRIVATE KEY-----"
        )
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.build_reflection",
            side_effect=ValidationError(private_key),
        ):
            with redirect_stderr(stderr):
                code = main(["--input", str(SAMPLE)])
        self.assertEqual(code, 2)
        self.assertNotIn(private_key, stderr.getvalue())
        self.assertNotIn("BEGIN PRIVATE KEY", stderr.getvalue())
        self.assertNotIn("M" * 64, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_secret_bearing_destination_filename(self):
        token = "sk-" + "Z" * 24
        output = self.temp / f"{token}.md"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--input", str(SAMPLE), "--write", "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertNotIn(token, stdout.getvalue())
        self.assertNotIn(token, stderr.getvalue())
        self.assertIn("<redacted diagnostic>", stdout.getvalue())
        self.assertTrue(output.exists())

    def test_write_without_output_is_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--input", str(SAMPLE), "--write"])
        self.assertNotEqual(code, 0)
        self.assertIn("--output", stderr.getvalue())

    def test_output_without_write_writes_nothing(self):
        out = self.temp / "reflection.md"
        code = main(["--input", str(SAMPLE), "--output", str(out)])
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())

    def test_write_creates_file_and_replace_controls_overwrite(self):
        out = self.temp / "reflection.md"
        code = main(["--input", str(SAMPLE), "--write", "--output", str(out)])
        self.assertEqual(code, 0)
        first_content = out.read_text(encoding="utf-8")
        self.assertIn("Requested outcome", out.read_text(encoding="utf-8"))
        code = main(["--input", str(SAMPLE), "--write", "--output", str(out)])
        self.assertNotEqual(code, 0)
        self.assertEqual(out.read_text(encoding="utf-8"), first_content)
        code = main(
            ["--input", str(SAMPLE), "--write", "--output", str(out), "--replace"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.read_text(encoding="utf-8"), first_content)

    def test_cli_refuses_live_repository_output_before_overwrite(self):
        repo = self.temp / "repo"
        (repo / "docs" / "agent").mkdir(parents=True)
        (repo / "evidence.txt").write_text("safe evidence\n", encoding="utf-8")
        live = repo / "docs" / "agent" / "REFLECTION.md"
        live.write_text("LIVE CONTEXT\n", encoding="utf-8")
        input_path = self.temp / "input.json"
        payload = base_payload()
        payload["evidence"] = ["evidence.txt"]
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        before = live.read_bytes()
        stderr = io.StringIO()

        with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repo):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(input_path),
                        "--write",
                        "--replace",
                        "--output",
                        str(live),
                    ]
                )

        self.assertNotEqual(code, 0)
        self.assertIn("docs/reflections", stderr.getvalue())
        self.assertEqual(live.read_bytes(), before)

    def test_repository_boundary_canonicalizes_root_alias(self):
        repo = self.temp / "repo"
        output = repo / "docs" / "reflections" / "report.md"
        output.parent.mkdir(parents=True)
        alias = self.temp / "repo-alias"
        try:
            alias.symlink_to(repo, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        with patch("scripts.closeout_reflection.REPOSITORY_ROOT", alias):
            write_reflection(output, "SAFE REFLECTION\n")

        self.assertEqual(output.read_text(encoding="utf-8"), "SAFE REFLECTION\n")

    def test_repository_boundary_canonicalizes_windows_short_root_alias(self):
        if sys.platform != "win32":
            self.skipTest("Windows 8.3 aliases are Windows-only")

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / (
                "repository-root-with-a-deliberately-long-short-alias-name"
            )
            (repo / "docs" / "agent").mkdir(parents=True)
            live = repo / "docs" / "agent" / "REFLECTION.md"
            live.write_text("LIVE CONTEXT\n", encoding="utf-8")
            before = live.read_bytes()

            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                get_short_path = kernel32.GetShortPathNameW
                get_short_path.argtypes = [
                    wintypes.LPCWSTR,
                    wintypes.LPWSTR,
                    wintypes.DWORD,
                ]
                get_short_path.restype = wintypes.DWORD
                buffer = ctypes.create_unicode_buffer(4096)
                length = get_short_path(str(repo), buffer, len(buffer))
            except (AttributeError, OSError) as exc:
                self.skipTest(f"Windows short-path API unavailable: {exc}")

            short_root = Path(buffer.value) if length else None
            if (
                short_root is None
                or length >= len(buffer)
                or str(short_root).casefold() == str(repo).casefold()
            ):
                self.skipTest("Windows 8.3 aliases are unavailable for this path")

            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", short_root):
                with self.assertRaises(ProtectedPathError):
                    write_reflection(live, "OVERWRITTEN\n", replace=True)
            self.assertEqual(live.read_bytes(), before)

    def test_cli_refuses_noncanonical_repository_state_paths(self):
        repo = self.temp / "repo"
        (repo / "state").mkdir(parents=True)
        (repo / "evidence.txt").write_text("safe evidence\n", encoding="utf-8")
        input_path = self.temp / "state-input.json"
        payload = base_payload()
        payload["evidence"] = ["evidence.txt"]
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        for index, relative in enumerate(("state/current.json", "state/tasks.yaml")):
            target = repo / relative
            target.write_text(json.dumps(seed_state()), encoding="utf-8")
            before = target.read_bytes()
            out = self.temp / f"state-boundary-reflection-{index}.md"
            stderr = io.StringIO()

            with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repo):
                with redirect_stderr(stderr):
                    code = main(
                        [
                            "--input",
                            str(input_path),
                            "--write",
                            "--output",
                            str(out),
                            "--stage-lesson",
                            "--state-path",
                            str(target),
                        ]
                    )

            self.assertNotEqual(code, 0, relative)
            self.assertIn("state/lessons_pending.json", stderr.getvalue())
            self.assertEqual(target.read_bytes(), before, relative)
            self.assertFalse(out.exists(), relative)

    def test_write_without_stage_lesson_leaves_state_untouched(self):
        out = self.temp / "reflection.md"
        code = main(
            ["--input", str(SAMPLE), "--write", "--output", str(out)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(self.state_path.read_text(encoding="utf-8")), seed_state()
        )

    def test_stage_lesson_requires_write(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = main(["--input", str(SAMPLE), "--stage-lesson"])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(self.state_path.read_text(encoding="utf-8")), seed_state()
        )
        self.assertIn("--write", captured.getvalue())

    def test_stage_lesson_with_write_stages_candidate(self):
        out = self.temp / "reflection.md"
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(out),
                    "--stage-lesson",
                    "--state-path",
                    str(self.state_path),
                ]
            )
        self.assertEqual(code, 0)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["lessons"]), 2)
        self.assertEqual(state["lessons"][-1]["status"], "candidate")

    def test_default_stage_path_is_repository_canonical_from_external_cwd(self):
        repo = self.temp / "repo"
        (repo / "state").mkdir(parents=True)
        (repo / "evidence.txt").write_text("safe evidence\n", encoding="utf-8")
        input_path = self.temp / "default-state-input.json"
        payload = base_payload()
        payload["evidence"] = ["evidence.txt"]
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        output = self.temp / "default-state-reflection.md"
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.temp)
        with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repo):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main([
                    "--input",
                    str(input_path),
                    "--write",
                    "--output",
                    str(output),
                    "--stage-lesson",
                ])
        self.assertEqual(code, 0)
        self.assertTrue((repo / "state" / "lessons_pending.json").exists())
        self.assertFalse((self.temp / "state" / "lessons_pending.json").exists())

    def test_output_cannot_overwrite_payload_evidence(self):
        repo = self.temp / "repo"
        (repo / "docs" / "reflections").mkdir(parents=True)
        evidence = repo / "docs" / "reflections" / "source.md"
        evidence.write_text("SOURCE EVIDENCE\n", encoding="utf-8")
        input_path = self.temp / "evidence-collision-input.json"
        payload = base_payload()
        payload["evidence"] = ["docs/reflections/source.md"]
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repo):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main([
                    "--input",
                    str(input_path),
                    "--write",
                    "--replace",
                    "--output",
                    str(evidence),
                ])
        self.assertNotEqual(code, 0)
        self.assertEqual(evidence.read_text(encoding="utf-8"), "SOURCE EVIDENCE\n")
        self.assertIn("evidence", stderr.getvalue().lower())
        self.assertNotIn("SOURCE EVIDENCE", stderr.getvalue())

    def test_output_diagnostics_redact_secret_filename(self):
        token = "github_pat_" + "L" * 30
        output = self.temp / f"{token}.md"
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            code = main(["--input", str(SAMPLE), "--write", "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertNotIn(token, captured.getvalue())
        self.assertTrue(output.exists())

    def test_state_cannot_overwrite_payload_evidence(self):
        repo = self.temp / "repo"
        (repo / "state").mkdir(parents=True)
        (repo / "docs" / "reflections").mkdir(parents=True)
        state = repo / "state" / "lessons_pending.json"
        state.write_text(json.dumps(seed_state()), encoding="utf-8")
        before = state.read_bytes()
        input_path = self.temp / "state-evidence-input.json"
        payload = base_payload()
        payload["evidence"] = ["state/lessons_pending.json"]
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        output = repo / "docs" / "reflections" / "safe.md"
        with patch("scripts.closeout_reflection.REPOSITORY_ROOT", repo):
            with redirect_stderr(io.StringIO()):
                code = main([
                    "--input",
                    str(input_path),
                    "--write",
                    "--output",
                    str(output),
                    "--stage-lesson",
                    "--state-path",
                    str(state),
                ])
        self.assertNotEqual(code, 0)
        self.assertEqual(state.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_cli_refuses_protected_output_paths(self):
        stderr = io.StringIO()
        targets = (
            self.temp / "SOUL.md",
            self.temp / "AGENTS.md",
            self.temp / "CLAUDE.md",
            self.temp / "rules" / "gpu.md",
        )
        for target in targets:
            with redirect_stderr(stderr):
                code = main(
                    ["--input", str(SAMPLE), "--write", "--output", str(target)]
                )
            self.assertNotEqual(code, 0, str(target))
            self.assertFalse(target.exists(), str(target))
            self.assertIn("protected", stderr.getvalue().lower(), str(target))

    def test_cli_refuses_protected_state_path_for_staging(self):
        out = self.temp / "reflection.md"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(out),
                    "--stage-lesson",
                    "--state-path",
                    str(self.temp / "rules" / "data.md"),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertFalse((self.temp / "rules" / "data.md").exists())
        self.assertIn("protected", stderr.getvalue().lower())

    def test_cli_validates_state_contents_before_writing_reflection(self):
        out = self.temp / "reflection.md"
        self.state_path.write_text("not valid JSON", encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(out),
                    "--stage-lesson",
                    "--state-path",
                    str(self.state_path),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertFalse(out.exists())

    def test_cli_rejects_surrogate_input_before_output_write(self):
        input_path = self.temp / "surrogate-input.json"
        payload = base_payload()
        payload["reusable_lesson"] = "\ud800"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        output = self.temp / "reflection.md"
        output.write_text("ORIGINAL\n", encoding="utf-8")
        before = output.read_bytes()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(input_path),
                    "--write",
                    "--output",
                    str(output),
                    "--replace",
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(output.read_bytes(), before)
        self.assertIn("utf-8", stderr.getvalue().lower())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_surrogate_state_before_output_write(self):
        state = seed_state()
        state["_comment"] = "\ud800"
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()
        output = self.temp / "reflection.md"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(output),
                    "--stage-lesson",
                    "--state-path",
                    str(self.state_path),
                ]
            )
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertIn("state", stderr.getvalue().lower())
        self.assertIn("utf-8", stderr.getvalue().lower())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_deep_state_before_output_write(self):
        state = seed_state()
        nested = "leaf"
        for _ in range(995):
            nested = {"next": nested}
        state["_deep"] = nested
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        before = self.state_path.read_bytes()
        output = self.temp / "reflection.md"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(output),
                    "--stage-lesson",
                    "--state-path",
                    str(self.state_path),
                ]
            )
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertIn("state", stderr.getvalue().lower())
        self.assertIn("depth", stderr.getvalue().lower())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_invalid_utf8_state_before_writing_output(self):
        out = self.temp / "reflection.md"
        self.state_path.write_bytes(b"\xff\xfe")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(out),
                    "--stage-lesson",
                    "--state-path",
                    str(self.state_path),
                ]
            )
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())
        self.assertIn("state", stderr.getvalue().lower())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_shared_output_and_state_destination(self):
        shared = self.temp / "shared.json"
        aliased = self.temp / "unused" / ".." / "shared.json"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(shared),
                    "--stage-lesson",
                    "--state-path",
                    str(aliased),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertFalse(shared.exists())
        self.assertIn("distinct", stderr.getvalue().lower())

        shared.write_text(json.dumps(seed_state()), encoding="utf-8")
        before = shared.read_text(encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            code = main(
                [
                    "--input",
                    str(SAMPLE),
                    "--write",
                    "--output",
                    str(shared),
                    "--replace",
                    "--stage-lesson",
                    "--state-path",
                    str(aliased),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertEqual(shared.read_text(encoding="utf-8"), before)

    def test_cli_reports_staging_filesystem_failure_without_traceback(self):
        output = self.temp / "reflection.md"
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.stage_lesson_candidate",
            side_effect=OSError("simulated state write failure"),
        ):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output),
                        "--stage-lesson",
                        "--state-path",
                        str(self.state_path),
                    ]
                )
        self.assertNotEqual(code, 0)
        self.assertIn("state", stderr.getvalue().lower())
        self.assertNotIn("simulated state write failure", stderr.getvalue())
        self.assertTrue(output.exists())

    def test_cli_parser_redacts_unsafe_unknown_arguments(self):
        token = "sk-" + "X" * 24
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                main(["--input", str(SAMPLE), f"{token}\x00"])
        self.assertEqual(ctx.exception.code, 2)
        output = stderr.getvalue()
        self.assertNotIn(token, output)
        self.assertNotIn("\x00", output)
        self.assertIn("<redacted diagnostic>", output)
        self.assertNotIn("Traceback", output)

        safe_stderr = io.StringIO()
        with redirect_stderr(safe_stderr):
            with self.assertRaises(SystemExit) as safe_ctx:
                main(["--input", str(SAMPLE), "ordinary-extra"])
        self.assertEqual(safe_ctx.exception.code, 2)
        self.assertIn("ordinary-extra", safe_stderr.getvalue())

    def test_cli_diagnostics_are_encodable_on_ascii_stream(self):
        for unknown_argument in ("a" * 300, "é" * 300):
            with self.subTest(unknown_argument=unknown_argument[0]):
                raw = io.BytesIO()
                stream = io.TextIOWrapper(raw, encoding="ascii")
                try:
                    with redirect_stderr(stream):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["--input", "x", unknown_argument])
                    stream.flush()
                    output = raw.getvalue().decode("ascii")
                finally:
                    stream.detach()
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("closeout_reflection: error:", output)
                self.assertNotIn("Traceback", output)

    def test_cli_redacts_input_path_resolution_error_detail(self):
        detail = "Permission denied: /private/input.json"
        with patch(
            "scripts.closeout_reflection.os.path.abspath",
            side_effect=OSError(detail),
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["--input", "x"])
        self.assertEqual(code, 2)
        self.assertIn("filesystem operation failed", stderr.getvalue())
        self.assertNotIn(detail, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_unsafe_staged_candidate_id(self):
        output = self.temp / "reflection.md"
        token = "sk-" + "J" * 24
        captured = io.StringIO()
        with patch(
            "scripts.closeout_reflection.stage_lesson_candidate",
            return_value={"id": f"{token}\n"},
        ):
            with redirect_stdout(captured):
                code = main(
                    [
                        "--input", str(SAMPLE),
                        "--write",
                        "--output", str(output),
                        "--stage-lesson",
                        "--state-path", str(self.state_path),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertNotIn(token, captured.getvalue())
        self.assertNotIn("\n", captured.getvalue().split("staged lesson candidate:", 1)[1])
        self.assertIn("<redacted diagnostic>", captured.getvalue())

    def test_cli_redacts_secret_bearing_preflight_filesystem_failure(self):
        output = self.temp / "reflection.md"
        token = "github_pat_" + "P" * 30
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection._validate_output_path",
            side_effect=OSError(f"simulated preflight failure for {token}"),
        ):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("filesystem operation failed", stderr.getvalue())
        self.assertNotIn(token, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_cli_redacts_secret_bearing_validation_exception(self):
        token = "sk-" + "Q" * 24
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.build_reflection",
            side_effect=ValidationError(f"unsafe destination {token}\x00"),
        ):
            with redirect_stderr(stderr):
                code = main(["--input", str(SAMPLE)])
        self.assertEqual(code, 2)
        self.assertNotIn(token, stderr.getvalue())
        self.assertNotIn("\x00", stderr.getvalue())
        self.assertNotIn("\\x00", stderr.getvalue())
        self.assertIn("filesystem operation failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_control_characters_from_exception_detail(self):
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.build_reflection",
            side_effect=ValidationError("ordinary\nunsafe"),
        ):
            with redirect_stderr(stderr):
                code = main(["--input", str(SAMPLE)])
        self.assertEqual(code, 2)
        self.assertNotIn("ordinary", stderr.getvalue())
        self.assertIn("filesystem operation failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_unexpected_input_exception_detail(self):
        token = "github_pat_" + "N" * 30
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.build_reflection",
            side_effect=TypeError(f"unexpected build failure {token}\n"),
        ):
            with redirect_stderr(stderr):
                code = main(["--input", str(SAMPLE)])
        output = stderr.getvalue()
        diagnostic = output.rstrip("\n")
        self.assertEqual(code, 2)
        self.assertNotIn(token, output)
        self.assertNotIn("\n", diagnostic)
        self.assertNotIn("\r", output)
        self.assertIn("filesystem operation failed", output)
        self.assertNotIn("Traceback", output)

    def test_cli_redacts_unexpected_input_path_detail(self):
        detail = "failed path C:/private/customer/input.json"
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.build_reflection",
            side_effect=RuntimeError(detail),
        ):
            with redirect_stderr(stderr):
                code = main(["--input", str(SAMPLE)])
        output = stderr.getvalue()
        diagnostic = output.rstrip("\n")
        self.assertEqual(code, 2)
        self.assertNotIn(detail, output)
        self.assertNotIn("\n", diagnostic)
        self.assertIn("filesystem operation failed", output)
        self.assertNotIn("Traceback", output)

    def test_cli_redacts_unexpected_write_exception_detail(self):
        token = "sk-" + "O" * 24
        raw_path = "C:/private/customer/write.txt"
        output_path = self.temp / "unexpected-write.md"
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection.write_reflection",
            side_effect=TypeError(
                f"unexpected write failure {token} {raw_path}\r"
            ),
        ):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output_path),
                    ]
                )
        output = stderr.getvalue()
        self.assertEqual(code, 2)
        self.assertNotIn(token, output)
        self.assertNotIn(raw_path, output)
        self.assertNotIn("\r", output)
        self.assertIn("filesystem operation failed", output)
        self.assertNotIn("Traceback", output)
        self.assertFalse(output_path.exists())

    def test_cli_redacts_unexpected_preflight_path_detail(self):
        detail = "failed path C:/private/customer/output.md"
        output_path = self.temp / "unexpected-preflight.md"
        stderr = io.StringIO()
        with patch(
            "scripts.closeout_reflection._validate_output_path",
            side_effect=RuntimeError(detail),
        ):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output_path),
                    ]
                )
        output = stderr.getvalue()
        diagnostic = output.rstrip("\n")
        self.assertEqual(code, 2)
        self.assertNotIn(detail, output)
        self.assertNotIn("\n", diagnostic)
        self.assertIn("filesystem operation failed", output)
        self.assertNotIn("Traceback", output)
        self.assertFalse(output_path.exists())

    def test_cli_redacts_unexpected_staged_id_lookup_error(self):
        token = "github_pat_" + "V" * 30
        raw_path = "C:/private/candidate/staged.json"
        output = self.temp / "staged-lookup.md"

        class BrokenStaged:
            def get(self, key):
                raise RuntimeError(f"staged lookup {token} {raw_path}\n")

        stdout = io.StringIO()
        with patch(
            "scripts.closeout_reflection.stage_lesson_candidate",
            return_value=BrokenStaged(),
        ):
            with redirect_stdout(stdout):
                code = main(
                    [
                        "--input", str(SAMPLE),
                        "--write",
                        "--output", str(output),
                        "--stage-lesson",
                        "--state-path", str(self.state_path),
                    ]
                )
        output_text = stdout.getvalue()
        staged_line = output_text.split("staged lesson candidate:", 1)[1]
        self.assertEqual(code, 0)
        self.assertNotIn(token, output_text)
        self.assertNotIn(raw_path, output_text)
        self.assertNotIn("\n", staged_line)
        self.assertIn("<redacted diagnostic>", output_text)
        self.assertNotIn("Traceback", output_text)

    def test_cli_preflight_and_write_validation_errors_redact_detail(self):
        token = "sk-" + "R" * 24
        for phase in ("preflight", "write"):
            with self.subTest(phase=phase):
                output = self.temp / f"{phase}-reflection.md"
                stderr = io.StringIO()
                if phase == "preflight":
                    exception_patch = patch(
                        "scripts.closeout_reflection._validate_output_path",
                        side_effect=ValidationError(f"unsafe destination {token}\x00"),
                    )
                else:
                    exception_patch = patch(
                        "scripts.closeout_reflection.write_reflection",
                        side_effect=ValidationError(f"unsafe destination {token}\x00"),
                    )
                with exception_patch, redirect_stderr(stderr):
                    code = main(
                        [
                            "--input", str(SAMPLE),
                            "--write",
                            "--output", str(output),
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertNotIn(token, stderr.getvalue())
                self.assertNotIn("\x00", stderr.getvalue())
                self.assertNotIn("\\x00", stderr.getvalue())
                self.assertIn("filesystem operation failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_safe_looking_validation_path_detail(self):
        detail = "failed path C:/private/customer/secret.txt"
        for phase in ("preflight", "write"):
            with self.subTest(phase=phase):
                output = self.temp / f"safe-looking-{phase}.md"
                stderr = io.StringIO()
                if phase == "preflight":
                    exception_patch = patch(
                        "scripts.closeout_reflection._validate_output_path",
                        side_effect=ValidationError(detail),
                    )
                else:
                    exception_patch = patch(
                        "scripts.closeout_reflection.write_reflection",
                        side_effect=ValidationError(detail),
                    )
                with exception_patch, redirect_stderr(stderr):
                    code = main(
                        [
                            "--input", str(SAMPLE),
                            "--write",
                            "--output", str(output),
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertNotIn(detail, stderr.getvalue())
                self.assertIn("filesystem operation failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_reports_partial_reflection_write_failure_without_traceback(self):
        output = self.temp / "reflection.md"
        stderr = io.StringIO()

        def partial_write(path, data, **kwargs):
            path.write_bytes(b"PARTIAL")
            raise OSError("simulated reflection write failure")

        with patch.object(Path, "write_text", autospec=True, side_effect=partial_write):
            with redirect_stderr(stderr):
                code = main(
                    [
                        "--input",
                        str(SAMPLE),
                        "--write",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "PARTIAL")
        self.assertIn("reflection", stderr.getvalue().lower())
        self.assertNotIn("simulated reflection write failure", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_redacts_secret_bearing_write_failure(self):
        output = self.temp / "reflection.md"
        token = "github_pat_" + "M" * 30
        stderr = io.StringIO()

        def failing_write(path, data, **kwargs):
            path.write_bytes(b"PARTIAL")
            raise OSError(f"simulated failure for {token}")

        with patch.object(Path, "write_text", autospec=True, side_effect=failing_write):
            with redirect_stderr(stderr):
                code = main([
                    "--input", str(SAMPLE),
                    "--write",
                    "--output", str(output),
                ])
        self.assertEqual(code, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "PARTIAL")
        self.assertNotIn(token, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("reflection", stderr.getvalue().lower())

    def test_cli_rejects_input_state_destination_collision(self):
        input_path = self.temp / "input.json"
        input_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        before = input_path.read_bytes()
        output = self.temp / "reflection.md"
        with redirect_stderr(io.StringIO()):
            code = main(
                [
                    "--input",
                    str(input_path),
                    "--write",
                    "--output",
                    str(output),
                    "--stage-lesson",
                    "--state-path",
                    str(input_path),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertFalse(output.exists())
        self.assertEqual(input_path.read_bytes(), before)

    def test_cli_rejects_input_output_destination_collision(self):
        input_path = self.temp / "input.json"
        input_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        before = input_path.read_bytes()
        state = self.temp / "lessons.json"
        state.write_text(json.dumps(seed_state()), encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            code = main(
                [
                    "--input",
                    str(input_path),
                    "--write",
                    "--output",
                    str(input_path),
                    "--replace",
                    "--stage-lesson",
                    "--state-path",
                    str(state),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertEqual(input_path.read_bytes(), before)
        self.assertEqual(json.loads(state.read_text(encoding="utf-8")), seed_state())

    def test_nonfinite_input_json_is_rejected_without_traceback(self):
        for index, raw in enumerate(('{"value": NaN}', '{"value": 1e999}')):
            with self.subTest(raw=raw):
                input_path = self.temp / f"nonfinite-{index}.json"
                input_path.write_text(raw, encoding="utf-8")
                output = self.temp / f"nonfinite-output-{index}.md"
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    code = main(
                        ["--input", str(input_path), "--write", "--output", str(output)]
                    )
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())
                self.assertIn("json", stderr.getvalue().lower())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_symlink_input_before_reading_or_output(self):
        target = self.temp / "input-target.json"
        target.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        link = self.temp / "linked-input.json"
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--input", str(link)])

        self.assertEqual(code, 2)
        self.assertIn("symlink", stderr.getvalue().lower())
        self.assertNotIn("Requested outcome", stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_oversized_input_before_json_or_output(self):
        oversized = self.temp / "oversized-input.json"
        oversized.write_bytes(b"x" * (MAX_INPUT_BYTES + 1))
        output = self.temp / "reflection.md"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(oversized),
                    "--write",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(code, 2)
        self.assertIn("limit", stderr.getvalue().lower())
        self.assertNotIn("valid json", stderr.getvalue().lower())
        self.assertNotIn("Requested outcome", stdout.getvalue())
        self.assertFalse(output.exists())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_invalid_input_json_is_rejected(self):
        bad = self.temp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--input", str(bad)])
        self.assertNotEqual(code, 0)

    def test_invalid_utf8_input_is_rejected_without_traceback(self):
        bad = self.temp / "bad.json"
        bad.write_bytes(b"\xff\xfe")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--input", str(bad)])
        self.assertEqual(code, 2)
        self.assertIn("input", stderr.getvalue().lower())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_input_file_is_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--input", str(self.temp / "nope.json")])
        self.assertNotEqual(code, 0)

    def test_required_field_validation_via_cli(self):
        bad = self.temp / "missing.json"
        payload = base_payload()
        payload.pop("validation_performed")
        bad.write_text(json.dumps(payload), encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--input", str(bad)])
        self.assertNotEqual(code, 0)
        self.assertIn("validation_performed", stderr.getvalue())


class MarkdownLinkTests(unittest.TestCase):
    """Task 5 docs must link only to targets that exist in the repository."""

    TASK5_DOCS = (
        Path("docs") / "agent" / "REFLECTION.md",
        Path("docs") / "reflections" / "README.md",
        Path("docs") / "reflections" / "INDEX.md",
        Path("docs") / "reflections" / "_template.md",
    )

    _LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")

    def test_every_relative_markdown_link_resolves(self):
        repo_root = Path(__file__).resolve().parents[2]
        broken = []
        for doc in self.TASK5_DOCS:
            text = (repo_root / doc).read_text(encoding="utf-8")
            for target in self._LINK_RE.findall(text):
                if "://" in target or target.startswith("#"):
                    continue
                if not (repo_root / doc.parent / target).is_file():
                    broken.append(f"{doc}: {target}")
        self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()
