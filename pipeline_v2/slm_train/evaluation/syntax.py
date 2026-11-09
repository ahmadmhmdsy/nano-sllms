"""Syntax checking for decoded samples.

The check is a pure function of the text: the caller supplies a parser name and
the callable that parses, so this module never imports a language runtime,
never shells out, and never guesses. An unparseable sample raises; a sample
that the parser accepts contributes zero to the error rate.
"""
from __future__ import annotations

__all__ = ["syntax_error_rate", "syntax_metrics"]


def syntax_error_rate(samples, parser) -> tuple:
    """(rate, failures). A parser that raises on a sample counts as a failure."""
    if not isinstance(samples, (list, tuple)):
        raise ValueError("syntax samples must be a sequence")
    if not samples:
        raise ValueError("syntax checking needs at least one sample")
    if not callable(parser):
        raise ValueError("a syntax parser must be callable")
    failures = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, str):
            failures.append("sample " + str(index) + " is not text")
            continue
        try:
            parser(sample)
        except Exception as error:
            failures.append("sample " + str(index) + ": " + type(error).__name__)
    return len(failures) / len(samples), failures


def syntax_metrics(samples, parser) -> dict:
    """The exact metric set the syntax suite declares."""
    rate, failures = syntax_error_rate(samples, parser)
    return {"syntax_error_rate": rate, "failures": failures}


def python_parser() -> callable:
    """Parse with the standard library only. No subprocess, no shell."""
    import ast

    def parse(text: str) -> None:
        ast.parse(text)

    return parse
