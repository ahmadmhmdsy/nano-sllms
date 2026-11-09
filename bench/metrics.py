"""
Multi-metric evaluation module.

Each metric is a callable that takes predictions and references and returns
a score. Metrics are composable and can be added without changing the runner.

DER calculation matches the gate probe (eval_der.py):
- Position-wise word alignment
- Mark-based comparison (diacritic marks only, not base letters)
- Multi-reference support: a word is correct if its marks match ANY reference
- Case-ending marks handled separately
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any
from collections.abc import Callable

__all__ = ["metric_for", "list_metrics", "pass_at_k", "der", "der_nocase", "wer", "preserve", "accuracy", "f1"]


# Code LLM Metrics
def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric.
    
    n: total number of samples
    c: number of correct samples
    k: k value for pass@k
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - prod((n-c-i)/(n-i) for i in range(k))
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


# Diacritizer Metrics (matching eval_der.py)
MARK_LO, MARK_HI = 0x064B, 0x0652  # fathatan .. sukun (the 8 label marks)


def _is_mark(ch: str) -> bool:
    """Check if a character is an Arabic diacritic mark."""
    return MARK_LO <= ord(ch) <= MARK_HI


def _base_of(word: str) -> str:
    """Get base letters (without diacritic marks)."""
    return "".join(c for c in word if not _is_mark(c))


def _marks_of(word: str) -> str:
    """Get diacritic marks only."""
    return "".join(c for c in word if _is_mark(c))


def _strip_case_marks(word: str) -> str:
    """Remove the ENTIRE trailing mark run (final letter's full label)."""
    end = len(word)
    while end > 0 and _is_mark(word[end - 1]):
        end -= 1
    return word[:end]


def _pairs_ok(pred_word: str | None, ref_word: str | None, nocase: bool) -> bool:
    """Check if pred and ref words match (mark-based comparison)."""
    if pred_word is None or ref_word is None:
        return False
    if nocase:
        return _strip_case_marks(pred_word) == _strip_case_marks(ref_word)
    return _marks_of(pred_word) == _marks_of(ref_word)


def _evaluate(pred_text: str, refs: list[str], nocase: bool = False) -> dict[str, float]:
    """Evaluate prediction against references (multi-reference supported).
    
    Returns dict with DER, preservation, WER, and word count.
    """
    pw = pred_text.split()
    rows = []
    max_len = max([len(pw)] + [len(r.split()) for r in refs])
    for i in range(max_len):
        p = pw[i] if i < len(pw) else None
        cand = [(r.split()[i] if i < len(r.split()) else None) for r in refs]
        der_ok = any(_pairs_ok(p, c, nocase) for c in cand)
        g = cand[0]
        base_ok = (p is not None and g is not None and _base_of(p) == _base_of(g))
        wer_ok = (p is not None and g is not None and p == g)
        rows.append((int(der_ok), int(base_ok), int(wer_ok)))
    n = max(len(rows), 1)
    der = 1.0 - sum(r[0] for r in rows) / n
    preservation = sum(r[1] for r in rows) / n
    wer = 1.0 - sum(r[2] for r in rows) / n
    return {"der": der, "preserve": preservation, "wer": wer, "words": n}


def _parse_multi_ref(ref_text: str) -> list[str]:
    """Parse reference text with multi-reference format.
    
    WikiNews format: words with "/" separators indicate alternatives.
    Each line may have multiple references separated by tabs.
    """
    # Split by tab first (multiple references)
    refs = ref_text.split("\t")
    if len(refs) > 1:
        return [r.strip() for r in refs if r.strip()]
    
    # Single reference with "/" separators in words
    # We need to expand the alternatives into separate reference strings
    words = ref_text.split()
    if not words:
        return [ref_text]
    
    # Check if any word has "/" separator
    has_multi = any("/" in w for w in words)
    if not has_multi:
        return [ref_text]
    
    # Expand multi-reference words into separate reference strings
    # For simplicity, take the first alternative of each word
    # (the gate probe uses all alternatives via tab separation)
    ref_words = []
    for w in words:
        if "/" in w:
            ref_words.append(w.split("/")[0])
        else:
            ref_words.append(w)
    return [" ".join(ref_words)]


def der(predictions: list[str], references: list[str]) -> float:
    """Diacritic Error Rate: fraction of words with incorrect diacritics.
    
    Matches gate probe semantics:
    - Position-wise word alignment
    - Mark-based comparison (diacritic marks only)
    - Multi-reference support: correct if marks match ANY reference
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    total_words = 0
    total_der = 0.0
    
    for pred, ref in zip(predictions, references):
        refs = _parse_multi_ref(ref)
        result = _evaluate(pred, refs, nocase=False)
        total_der += result["der"] * result["words"]
        total_words += result["words"]
    
    return total_der / total_words if total_words > 0 else 0.0


def der_nocase(predictions: list[str], references: list[str]) -> float:
    """Diacritic Error Rate (case-insensitive): ignores case endings."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    total_words = 0
    total_der = 0.0
    
    for pred, ref in zip(predictions, references):
        refs = _parse_multi_ref(ref)
        result = _evaluate(pred, refs, nocase=True)
        total_der += result["der"] * result["words"]
        total_words += result["words"]
    
    return total_der / total_words if total_words > 0 else 0.0


def wer(predictions: list[str], references: list[str]) -> float:
    """Word Error Rate: fraction of incorrect words (exact match)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    total_words = 0
    total_wer = 0.0
    
    for pred, ref in zip(predictions, references):
        refs = _parse_multi_ref(ref)
        result = _evaluate(pred, refs, nocase=False)
        total_wer += result["wer"] * result["words"]
        total_words += result["words"]
    
    return total_wer / total_words if total_words > 0 else 0.0


def preserve(predictions: list[str], references: list[str]) -> float:
    """Text preservation: fraction of words with identical base letters."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    total_words = 0
    total_preserve = 0.0
    
    for pred, ref in zip(predictions, references):
        refs = _parse_multi_ref(ref)
        result = _evaluate(pred, refs, nocase=False)
        total_preserve += result["preserve"] * result["words"]
        total_words += result["words"]
    
    return total_preserve / total_words if total_words > 0 else 0.0


def accuracy(predictions: list[Any], references: list[Any]) -> float:
    """Simple accuracy: fraction of exact matches."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    if not predictions:
        return 0.0
    
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def f1(predictions: list[Any], references: list[Any]) -> float:
    """F1 score for binary classification."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    
    tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
    fp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 0)
    fn = sum(1 for p, r in zip(predictions, references) if p == 0 and r == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Metric registry
METRICS: dict[str, Callable] = {
    "pass@1": lambda preds, refs: pass_at_k(len(preds), sum(1 for p, r in zip(preds, refs) if p == r), 1),
    "pass@10": lambda preds, refs: pass_at_k(len(preds), sum(1 for p, r in zip(preds, refs) if p == r), 10),
    "pass@100": lambda preds, refs: pass_at_k(len(preds), sum(1 for p, r in zip(preds, refs) if p == r), 100),
    "der": der,
    "der_nocase": der_nocase,
    "wer": wer,
    "preserve": preserve,
    "accuracy": accuracy,
    "f1": f1,
}


def metric_for(name: str) -> Callable:
    """Get a metric function by name."""
    if name not in METRICS:
        available = ", ".join(sorted(METRICS))
        raise KeyError(f"unknown metric: {name}; available: {available}")
    return METRICS[name]


def list_metrics() -> list[str]:
    """List all available metric names."""
    return sorted(METRICS)
