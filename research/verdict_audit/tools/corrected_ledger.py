#!/usr/bin/env python
"""Corrected D-line ledger: der_canonical is the order-insensitive metric.

E-090 (2026-10-11) established that every legacy registered-gate number was
inflated: the decoder emitted shadda-first mark runs while bench.load_pairs
NFC-normalizes every reference to vowel-first, so the two sides were compared in
different byte orders on every geminate, and der_registered counts WORDS.

der_canonical is computed order-insensitively (canonicalize_word rewrites mark
runs through the label map), so an arm's RECORDED der_canonical IS its corrected
der_registered. E-090 verified this directly: re-running the gate with the fixed
decoder reproduced the artifact der_canonical within 0.06 pct on four of five
corpora and was slightly better on the other two.

Consequence: the corrected ledger can be recovered from artifacts that already
exist, with NO model re-runs.

SIGNIFICANCE DISCIPLINE. A sign change is not a finding. The pre-registered bar
on this line is 2 pct relative, and C0-R measured an init-draw variance of
0.82 pct. Arms whose corrected gap is inside 0.82 pct are reported as
INDISTINGUISHABLE, not as a new ordering -- the legacy metric did not merely
inflate the numbers, it manufactured orderings among near-ties.

CPU only. Writes runs/diac/_screen_out/corrected_ledger.json
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCAN = ROOT / "runs" / "diac"
OUT = SCAN / "_screen_out" / "corrected_ledger.json"
GATE = ("fadel_test", "sadeed25", "wikinews2014")
SIG_PCT = 2.0      # pre-registered significance bar (plan section 5)
NOISE_PCT = 0.82   # measured C0-R init-draw variance
# The deployed single-pass baseline this line selects against.
BASELINE = ("runs/diac/e75_vote/verdict.json", "single")


def gate_mean(rows, key):
    vals = [rows[c][key] for c in GATE if isinstance(rows.get(c), dict) and key in rows[c]]
    return sum(vals) / len(vals) if len(vals) == len(GATE) else None


def main():
    entries = []
    for path in sorted(SCAN.rglob("*.json")):
        if path == OUT or "_screen_out" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for arm, rows in data.items():
            if not isinstance(rows, dict) or arm == "gate_mean3":
                continue
            old = gate_mean(rows, "der_registered")
            new = gate_mean(rows, "der_canonical")
            if old is None or new is None:
                continue
            entries.append({
                "arm": arm, "source": rel,
                "decode": "nfc" if "e90_nfc_fix" in rel else "legacy",
                "gate_old": old, "gate_corrected": new,
                "rel_delta_pct": (new - old) / old * 100.0,
                "per_corpus_corrected": {c: rows[c]["der_canonical"] for c in GATE},
            })

    base = next((e for e in entries
                 if e["source"] == BASELINE[0] and e["arm"] == BASELINE[1]), None)
    b = base["gate_corrected"]
    for e in entries:
        e["delta_vs_baseline_pct"] = (e["gate_corrected"] - b) / b * 100.0
        d = abs(e["delta_vs_baseline_pct"])
        e["vs_baseline"] = "SIG" if d >= SIG_PCT else ("noise" if d < NOISE_PCT else "inconclusive")

    # Rank within each decode convention, corrected condition.
    for conv in ("legacy", "nfc"):
        order = sorted([e for e in entries if e["decode"] == conv],
                       key=lambda e: e["gate_corrected"])
        for i, e in enumerate(order, 1):
            e["rank_corrected_in_convention"] = i
    # Rank on the legacy (inflated) metric, legacy arms only, for flip detection.
    legacy = [e for e in entries if e["decode"] == "legacy"]
    for i, e in enumerate(sorted(legacy, key=lambda e: e["gate_old"]), 1):
        e["rank_old_legacy"] = i

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "gate_external": list(GATE), "baseline": BASELINE,
        "baseline_gate_corrected": b,
        "sig_pct": SIG_PCT, "noise_pct": NOISE_PCT, "arms": entries,
    }, indent=2), encoding="utf-8")

    legacy.sort(key=lambda e: e["gate_corrected"])
    print(f"baseline = {BASELINE[0]}::{BASELINE[1]} corrected {b:.6f}")
    print(f"bars: SIG >= {SIG_PCT}%  |  NOISE < {NOISE_PCT}%\n")
    print(f"{'source::arm':<46} {'corrected':>10} {'vs base':>9} {'flag':>13} {'old':>9} {'rank flip':>9}")
    print("-" * 104)
    for e in legacy:
        label = f"{e['source'].replace('runs/diac/','')}::{e['arm']}"
        flip = e["rank_old_legacy"] - e["rank_corrected_in_convention"]
        print(f"{label[:45]:<46} {e['gate_corrected']:>10.6f} "
              f"{e['delta_vs_baseline_pct']:>8.2f}% {e['vs_baseline']:>13} "
              f"{e['gate_old']:>9.6f} {flip:>+9d}")
    print(f"\nlegacy arms = {len(legacy)}")
    sig = [e for e in legacy if e["vs_baseline"] == "SIG"]
    print(f"arms SIGNIFICANTLY better than the single baseline: {len(sig)}")
    for e in sig:
        print(f"  {e['source']}::{e['arm']}  {e['delta_vs_baseline_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
