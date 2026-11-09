"""Redaction: a secret never reaches a manifest, a log, or a report.

Redaction is by VALUE, not only by key. A token pasted into a "notes" field, or
an API key echoed into an error message, is the leak that key matching misses.
So the registry records the live values of the environment variables V2 treats as
sensitive, and every string that passes through here has those values replaced
with a marker before anything is written.
"""
from __future__ import annotations

import os
import re

__all__ = [
    "REDACTED",
    "SENSITIVE_ENV_NAMES",
    "SENSITIVE_KEY_PATTERN",
    "redact",
    "redact_mapping",
    "sensitive_environment",
    "value_is_sensitive",
]

REDACTED = "[REDACTED]"

#: Environment variables whose VALUES are secrets. The name is public; the value
#: is not, and the value is what gets replaced wherever it appears.
SENSITIVE_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GITHUB_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
)

#: Key names whose values are redacted even when the key is not in the list.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"credential|authorization|bearer|session[_-]?id)",
    re.IGNORECASE,
)

_MINIMUM_SECRET_LENGTH = 6


def sensitive_environment(environ=None) -> tuple:
    """The live values of the sensitive variables, longest first.

    Longest first matters: redacting a short value that is a substring of a
    longer one would leave a readable tail of the longer secret.
    """
    source = os.environ if environ is None else environ
    values = set()
    for name in SENSITIVE_ENV_NAMES:
        value = source.get(name)
        if value and len(value) >= _MINIMUM_SECRET_LENGTH:
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def value_is_sensitive(key: str) -> bool:
    """True when the KEY alone says the value is a secret."""
    return bool(SENSITIVE_KEY_PATTERN.search(str(key)))


def redact(value, environ=None):
    """Replace every known secret value in a string, however it got there."""
    secrets = sensitive_environment(environ)
    if not isinstance(value, str):
        return value
    for secret in secrets:
        if secret in value:
            value = value.replace(secret, REDACTED)
    return value


def redact_mapping(document, environ=None):
    """A deep copy with sensitive keys masked and secret values replaced.

    The input is never mutated: a redacted manifest must not be the same object
    as the live one, or a later write could publish the unredacted form.
    """
    secrets = sensitive_environment(environ)
    if isinstance(document, dict):
        result = {}
        for key, item in document.items():
            # A secret can hide in a KEY as well as a value: a dict of tokens keyed
            # by token is a real shape, and redacting only the values would publish
            # every one of them in the key.
            safe_key = key
            if isinstance(safe_key, str):
                for secret in secrets:
                    safe_key = safe_key.replace(secret, REDACTED)
            if value_is_sensitive(key):
                result[safe_key] = REDACTED if item is not None else None
            else:
                result[safe_key] = redact_mapping(item, environ)
        return result
    if isinstance(document, list):
        return [redact_mapping(item, environ) for item in document]
    if isinstance(document, tuple):
        return tuple(redact_mapping(item, environ) for item in document)
    if isinstance(document, str):
        for secret in secrets:
            document = document.replace(secret, REDACTED)
    return document
