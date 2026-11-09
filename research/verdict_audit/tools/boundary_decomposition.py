#!/usr/bin/env python
r"""E-092 amendment: decompose the sliding-window vote's gain by distance to the
nearest hard chunk boundary.

POST-HOC ANALYSIS OF EXISTING E-092 ARTIFACTS - no model runs, CPU only.

bench.predict_bares builds the "single" arm with chunks_of(bare, ctx), i.e.
NON-OVERLAPPING 256-char pieces (diacritizer/scripts/bench.py:52-53), so every
chunk boundary is a self-inflicted context cut. The vote arm decodes with
overlapping windows. This tool buckets every word by its character distance to
the nearest multiple of 256 and reports both arms per bucket, which separates
"the vote repairs chunk boundaries" from "the vote mixes contexts".

Usage:
  & .\.venv\Scripts\python.exe research\verdict_audit\tools\boundary_decomposition.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
import bench as B          # noqa: E402
import eval_metrics as em  # noqa: E402

BASE = REPO / "runs" / "diac" / "e92_stride" / "128"
LIMIT = 400
CTX = 256
ORDER = ["0-8", "9-16", "17-32", "33-64", "65+"]


def is_mark(ch):
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in B.MARK_RANGES)


def bucket(off):
    d = off % CTX
    d = min(d, CTX - d)
    for hi, name in ((8, "0-8"), (16, "9-16"), (32, "17-32"), (64, "33-64")):
        if d <= hi:
            return name
    return "65+"


def load_pairs():
    pairs = []
    for bare, refs in B.load_pairs("abdou_test", CTX):
        pairs.append((bare, refs))
        if len(pairs) >= LIMIT:
            break
    return pairs


def recover(path, bares):
    """Recover the per-pair prediction strings from a joined output file.

    The file is "\n".join(preds) + "\n" and 197 of the 400 abdou bares contain
    embedded newlines, so a naive split is wrong. Walk the file and cut at each
    newline where the non-mark character count has reached the pair's bare
    length; that condition is true only at the separator, because an internal
    newline is itself a non-mark character and so leaves the count short.
    """
    raw = Path(path).read_text(encoding="utf-8")
    out, i, cnt, start, pos = [], 0, 0, 0, 0
    while i < len(bares) and pos < len(raw):
        ch = raw[pos]
        if ch == "\n" and cnt == len(bares[i]):
            out.append(raw[start:pos])
            i += 1
            cnt = 0
            start = pos + 1
            pos += 1
            continue
        if not is_mark(ch):
            cnt += 1
        pos += 1
    if i != len(bares):
        raise SystemExit("recovered %d pairs, expected %d" % (i, len(bares)))
    for j, p in enumerate(out):
        if B.strip_marks(p) != bares[j]:
            raise SystemExit("pair %d does not round-trip to its bare text" % j)
    return out


def word_spans(bare):
    spans, i, n = [], 0, len(bare)
    while i < n:
        while i < n and bare[i].isspace():
            i += 1
        if i >= n:
            break
        s = i
        while i < n and not bare[i].isspace():
            i += 1
        spans.append((s, i))
    return spans


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    pairs = load_pairs()
    bares = [b for b, _ in pairs]
    arms = {}
    for arm in ("single", "vote"):
        preds = recover(BASE / ("abdou_test.%s.txt" % arm), bares)
        agg = {k: [0, 0] for k in ORDER}
        for (bare, refs), pred in zip(pairs, preds):
            bits = em.word_outcomes(pred, "\t".join(refs))
            for (s, _e), ok in zip(word_spans(bare), bits):
                agg[bucket(s)][0] += 1
                agg[bucket(s)][1] += 1 if ok else 0
        arms[arm] = agg

    print("E-092 AMENDMENT - error rate by distance to nearest %d-char chunk boundary" % CTX)
    print("(abdou_test first %d pairs; POST-HOC on existing artifacts; no model runs)\n" % LIMIT)
    print("%-8s %-8s %-12s %-12s %-10s" % ("dist", "words", "single_err", "vote_err", "reduction"))
    for k in ORDER:
        n, gs = arms["single"][k]
        _, gv = arms["vote"][k]
        es, ev = 1 - gs / n, 1 - gv / n
        print("%-8s %-8d %-12.4f %-12.4f %-10.1f" % (k, n, es, ev, (es - ev) * n))

    tot_n = sum(arms["single"][k][0] for k in ORDER)
    tot_s = sum(arms["single"][k][0] - arms["single"][k][1] for k in ORDER)
    tot_v = sum(arms["vote"][k][0] - arms["vote"][k][1] for k in ORDER)
    total_red = tot_s - tot_v
    print("\nALL words=%d  single_err=%.4f  vote_err=%.4f  reduction=%.1f errors"
          % (tot_n, tot_s / tot_n, tot_v / tot_n, total_red))
    print("share of the vote's TOTAL error reduction, by bucket:")
    for k in ORDER:
        red = (arms["single"][k][0] - arms["single"][k][1]) - (arms["vote"][k][0] - arms["vote"][k][1])
        print("  %-8s %6.1f pct" % (k, 100.0 * red / total_red))
    n0 = arms["single"]["0-8"][0]
    print("\n0-8 bucket: %.1f pct of words, %.1f pct relative excess vs 65+ (single),"
          % (100.0 * n0 / tot_n,
             100.0 * ((1 - arms["single"]["0-8"][1] / n0) / (1 - arms["single"]["65+"][1] / arms["single"]["65+"][0]) - 1)))
    print("repaired %.1f pct relative by the vote; residual %.2f pct DER"
          % (100.0 * (1 - (1 - arms["vote"]["0-8"][1] / n0) / (1 - arms["single"]["0-8"][1] / n0)),
             100.0 * ((1 - arms["vote"]["0-8"][1] / n0) - (1 - arms["vote"]["65+"][1] / arms["vote"]["65+"][0])) * n0 / tot_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
