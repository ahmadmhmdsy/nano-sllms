"""Deterministic normalization, document IDs, and (dedupe-before-split order).

Document IDs are a pure function of the pinned source id, the source row
index, and the normalized text; no time, randomness, or set ordering enters.
"""
from typing import Callable

from slm_train.contracts.hash import canonical_sha256


def apply_normalization(value: object, mode: str) -> tuple[str, str]:
    """Apply one declared normalization mode; return (text, applied_mode).

    The DECLARED mode must be the APPLIED mode (A7): a manifest certifying
    unicode_nfkc over data that was only whitespace-collapsed is the exact
    class of false certification this module exists to make impossible. Only
    the modes named here exist; anything else is refused by the caller.
    """
    mode = str(mode or "ws-collapse")
    if mode == "unicode_nfkc":
        import unicodedata

        if not isinstance(value, str):
            return "", "unicode_nfkc"
        return " ".join(
            unicodedata.normalize("NFKC", value).split()
        ), "unicode_nfkc"
    if mode == "ui_task_nl":
        import unicodedata

        if not isinstance(value, str):
            return "", "ui_task_nl"
        # The ui chain contract (ui/scripts/prepare_u1.py) separates TASK and
        # chain spec with exactly one newline; unicode_nfkc flattens ALL
        # whitespace and would erase the loss-mask boundary. This mode keeps
        # one literal newline and collapses whitespace only within each line.
        text = unicodedata.normalize("NFKC", value)
        lines = text.split("\n")
        return "\n".join(" ".join(line.split()) for line in lines), "ui_task_nl"
    if mode == "ws-collapse":
        if not isinstance(value, str):
            return "", "ws-collapse"
        return " ".join(value.split()), "ws-collapse"
    raise ValueError("unsupported data.normalization: " + mode)


def _probe_normalization(mode: str) -> str:
    """Refuse an undeclared normalization mode before any row is read."""
    if str(mode) not in ("unicode_nfkc", "ui_task_nl", "ws-collapse"):
        raise ValueError("unsupported data.normalization: " + str(mode))
    return str(mode)


def normalized_text(value: object) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def text_key(text: str) -> str:
    """Dedupe key: normalized text lowercased (deterministic)."""
    return text.lower()


def document_id(source_id: str, source_row_index: int, text: str) -> str:
    if not source_id or source_row_index < 0 or not text:
        raise ValueError("document id requires source id, row index, and nonempty text")
    value = canonical_sha256({
        "row_index": int(source_row_index),
        "source_id": str(source_id),
        "text": str(text),
    })
    return value


def dedupe_before_split(
    documents: "list[dict]",
    key_fn: Callable = text_key,
) -> "tuple[list[dict], int]":
    """Keep the first occurrence per dedupe key; drop later duplicates.

    Deterministic: iteration follows the caller's stable ordering (the
    preparer supplies documents sorted by (source_id, row_index)); no set
    iteration order can leak into the result.
    """
    seen: "set[str]" = set()
    kept: "list[dict]" = []
    duplicates = 0
    for document in documents:
        marker = key_fn(document["text"])
        if marker in seen:
            duplicates += 1
            continue
        seen.add(marker)
        kept.append(document)
    return kept, duplicates
