"""Seeded generators for the three synthetic μ0 tasks.

Line conventions (char-level, self-distinguishing formats):
  arith:     prompt "a+b=|" then the answer, then newline
  structure: bracket string, newline, then "ok"/"bad", then newline
  strops:    "rev:abc|cba", "sort:zab|abz", "copy:qrs|qrs"
Every line ends with \n; all split at line level and stay disjoint.
"""
from __future__ import annotations

import random

SEED_DEFAULT = 42


def _split(rng: random.Random, lines: list[str], n_val: int, n_test: int):
    rng.shuffle(lines)
    assert len(lines) > n_val + n_test
    return {"train": lines[: len(lines) - n_val - n_test],
            "val": lines[len(lines) - n_val - n_test: len(lines) - n_test],
            "test": lines[len(lines) - n_test:]}


def arith(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
          n_train: int = 30000, max_op: int = 999) -> dict[str, list[str]]:
    rng = random.Random(f"mex-arith-{seed}")
    n = max_op + 1
    lines = []
    # Exact sampling over the full operand grid: exactly n_train addition lines
    # and n_train subtraction lines (the old Bernoulli filter kept ~97% of
    # additions plus every subtraction line, ~1.97M instead of ~2*n_train).
    # Index sampling is equivalent to sampling the generated line lists, since
    # each grid index maps deterministically to one line in the same order.
    for i in rng.sample(range(n * n), n_train):
        a, b = divmod(i, n)
        lines.append(f"{a}+{b}=|{a + b}\n")
    for i in rng.sample(range(n * n), n_train):
        a, b = divmod(i, n)
        s = max(a, b); d = min(a, b)
        lines.append(f"{s}-{d}=|{s - d}\n")
    return _split(rng, lines, n_val, n_test)


def structure(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
              maxlen: int = 24, n_train: int = 30000) -> dict[str, list[str]]:
    # E-24 hardening: the old generator drew raw random strings, so ~99% of
    # lines were 'bad' (trivial baseline 0.988) and badness was trivially
    # detectable garbage. Now 'ok' lines are balanced by construction; 'bad'
    # lines are half subtle (ONE bracket of a balanced string flipped to its
    # pair partner — locally plausible, badness needs the whole sequence)
    # and half legacy full flips. Label semantics IDENTICAL: 'ok' iff
    # _balanced says so; the checker itself is untouched. Same rng seeding
    # scheme and (seed, n_val, n_test, n_train) signature as before.
    rng = random.Random(f"mex-brk-{seed}")
    pairs = {"(": ")", "[": "]", "{": "}"}
    lines = []
    while len(lines) < n_train + n_val + n_test:
        want_ok = rng.random() < 0.5
        while True:
            if want_ok:
                seq = _balanced_seq(rng, 2 * rng.randrange(1, maxlen // 2 + 1), pairs)
            elif rng.random() < 0.5:  # subtle: one-pair flip of a balanced seq
                seq = _flip_one(rng, _balanced_seq(rng, 2 * rng.randrange(1, maxlen // 2 + 1), pairs), pairs)
            else:  # legacy full structural corruption
                seq = "".join(rng.choice("()[]{}") for _ in range(rng.randrange(2, maxlen + 1)))
            if _balanced(seq, pairs) == want_ok:
                break
        lines.append(f"{seq}\n{'ok' if _balanced(seq, pairs) else 'bad'}\n")
    return _split(rng, lines, n_val, n_test)


def _balanced_seq(rng: random.Random, n: int, pairs: dict[str, str]) -> str:
    """Random balanced bracket string of even length n (recursive splice)."""
    if n == 0:
        return ""
    o = rng.choice(list(pairs))
    inner = rng.randrange(0, n - 1, 2)  # even inner length; tail n-2-inner even
    return (o + _balanced_seq(rng, inner, pairs) + pairs[o]
            + _balanced_seq(rng, n - 2 - inner, pairs))


def _flip_one(rng: random.Random, seq: str, pairs: dict[str, str]) -> str:
    """Subtle corruption: flip ONE bracket position to its pair partner.

    Any single substitution of a balanced string unbalances exactly one
    pair's open/close counts, so the result is always 'bad' by _balanced —
    yet locally plausible: a flipped close bracket leaves every prefix
    valid, so badness surfaces only at the end of the line.
    """
    partner = {**pairs, **{v: k for k, v in pairs.items()}}
    i = rng.randrange(len(seq))
    out = seq[:i] + partner[seq[i]] + seq[i + 1:]
    assert not _balanced(out, pairs)
    return out


def _balanced(seq: str, pairs: dict[str, str]) -> bool:
    stack = []
    for ch in seq:
        if ch in pairs:
            stack.append(pairs[ch])
        elif not stack or stack.pop() != ch:
            return False
    return not stack


def strops(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
           maxlen: int = 16, n_train: int = 30000) -> dict[str, list[str]]:
    rng = random.Random(f"mex-str-{seed}")
    lines = []
    for i in range(n_train + n_val + n_test):
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randrange(3, maxlen)))
        mode = ("rev", "sort", "copy")[i % 3]
        out = s[::-1] if mode == "rev" else ("".join(sorted(s)) if mode == "sort" else s)
        lines.append(f"{mode}:{s}|{out}\n")
    return _split(rng, lines, n_val, n_test)
