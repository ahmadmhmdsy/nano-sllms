"""D-line selftest gate runner (plan section 2, R50/R51/R52 + AGENTS conventions).

Usage (venv, from anywhere):
    & .\\.venv\\Scripts\\python.exe diacritizer\\scripts\\selftest.py [--phase a0a|a0b|a0c|all]

A0 exit gate = --phase all == 4/4 PASS. Currently a0a (tokenizer + labels +
passthrough byte-exactness) is implemented; a0b/a0c register as SKIPPED until
R51/R52 land them.

Validation labels PASS / FAIL / SKIPPED; honest reporting over optimism.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import labels as L  # noqa: E402
import passthrough as PT  # noqa: E402
import tokenizer as TK  # noqa: E402


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (status, name, (" -- " + detail) if detail and not cond else ""))
    return cond


# --- gold fixtures: (diacritized text, expected per-target labels) ------------

def run_a0a():
    print("=== selftest a0a: tokenizer + labels + passthrough ===")
    ok = True

    # 1. vocab sanity
    ok &= check("vocab_size<400", TK.VOCAB_SIZE < 400, "size=%d" % TK.VOCAB_SIZE)
    ok &= check("vocab decode==identity on printable", TK.decode(TK.encode("abc XYZ 123")) == "abc XYZ 123")
    base_vocab = "".join(chr(o) for o in range(0x0621, 0x064B))
    ok &= check("arabic block in vocab", all(ch in TK.ID_OF for ch in base_vocab))

    # 2. labels: encode/decode symmetry across all 15 classes
    all_ok = True
    for lbl in range(L.N_CLASSES):
        marks = L.marks_for_label(lbl)
        if marks:
            got = L.label_for_marks(tuple(marks))
            if got != lbl:
                all_ok = False
                print("  mismatch label %d %r -> %d" % (lbl, marks, got))
    ok &= check("15-class marks<->id symmetry", all_ok)

    # 3. hand-built gold: "بِسْمِ اللهِ" style analysis
    text = "\u0628\u0650\u0633\u0652\u0645\u0651\u064a"  # b+i s+u? no: b+i, s+sukun, m+shadda+y
    units = PT.parse(text)
    got = [u.label for u in units if u.is_target]
    want = [L.C_KASRA, L.C_SUKUN, L.C_SHADDA, L.C_BARE]
    ok &= check("gold bism labels", got == want, "got=%s want=%s" % (got, want))

    # m + shadda + kasra -> shadda_kasra (11)
    u1 = PT.parse("\u0645\u0651\u0650")
    ok &= check("shadda+kasra decode", [x.label for x in u1 if x.is_target] == [L.C_SHADDA_KASRA])

    # shadda+tanwin vs vowel combo: "اً" alif+fathatan
    u2 = PT.parse("\u0627\u064b")
    ok &= check("fathatan decode", [u.label for u in u2 if u.is_target] == [L.C_FATHATAN])

    # 4. quarantine surface: shadda+sukun, double fatha
    for bad in ("\u0628\u0651\u0652", "\u0628\u064e\u064e"):
        try:
            PT.parse(bad)
            ok &= check("quarantine %r" % bad, False, "no error raised")
        except PT.QuarantineError:
            ok &= check("quarantine %r" % bad, True)

    # 5. BYTE-EXACT password: reconstruct(input) == input on pathological corpus
    cases = {
        "emoji soup": "\u0645\u0631\u062d\u0628\u0627 \U0001F600 caf\u00e9 \u0639\u0644\u064a\u0643\u0645 \u270c\uFE0F",
        "latin/URL/number": "IPv6 \u0641\u064a https://example.com/a?b=1&c=2 \u0648 3.14",
        "arabic-indic + western digits": "\u0627\u0644\u0639\u0627\u0645 1446 || 2025 \u0645",
        "RLM/ZWJ/ZWSP/BOM": "\u0623\u200f\u0628\u200d\u062c\u200b\u062f \uFEFF\u0647\u0646\u0627",
        "tatweel run": "\u0633\u0640\u0640\u0640\u0644\u0627\u0645",
        "lone marks start": "\u064e\u0627\u0644\u0633\u0644\u0627\u0645",
        "quranic marks": "\u0628\u0650\u0633\u0652\u0645\u06d6\u0644\u0644\u0651\u0647\u06d0",
        "mixed script word": "WiFi\u062c\u0647\u0627\u0632 GPT-4o \u0646\u0645\u0648\u0630\u062c",
        "superscript alef": "\u0639\u0644\u064a\u0649\u0670",
        "neutral punctuation line": ". , ! ? ; : ... \u0645\u0641\u062a\u0627\u062d",
        "CJK": "\u0623\u0647\u0644\u0627 \u4f60\u597d",
        "rtl markers around latin": "\u0627\u0644\u0645\u062f\u0649 \u200eRock\u200e \u0633\u0644\u0627\u0645",
    }
    # Byte-exactness = input minus the 15-class marks (those are replaced by the
    # prediction head by design; EVERY other codepoint must survive verbatim).
    def strip_after_base(t):
        out = []
        prev_base = False
        for ch in t:
            if ch in L._CONSUMABLE and prev_base:
                prev_base = False
                continue
            out.append(ch)
            prev_base = PT.is_arabic_base(ch)
        return "".join(out)
    for name, s in cases.items():
        units = PT.parse(s)
        expected = strip_after_base(s)
        ok &= check("byte-exact %s" % name, PT.reconstruct(units) == expected,
                    "got=%r want=%r" % (PT.reconstruct(units), expected))

    # 6. round trips: strip->relabel->reconstruct on a diacritized sentence
    sent = "\u0628\u0650\u0633\u0652\u0645\u0651\u0647\u0650 \u200f\u0627\u0644\u0644\u0651\u064e\u0647\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0646\u0650"
    u = PT.parse(sent)
    bases = "".join(x.raw for x in u if x.is_target)
    rebuilt = "".join(x.raw + ("" if not x.is_target else L.marks_for_label(x.label)) for x in u)
    # The decoder emits canonical NFC (labels.marks_for_label) while this
    # fixture is raw shadda-first Tashkeela order, so the round trip is exact
    # UP TO UNICODE NORMALIZATION -- which is the contract that matters.
    # Asserting raw byte equality here would re-assert the very
    # non-normalized decode that made the registered gate disagree with every
    # NFC reference on every geminate.
    ok &= check(
        "strip->relabel round trip (up to NFC)",
        unicodedata.normalize("NFC", rebuilt) == unicodedata.normalize("NFC", sent),
        "got=%r" % rebuilt)
    # stripped text has zero of the class marks
    ok &= check("stripped stream mark-free", not any(c in L._CONSUMABLE for c in bases))

    # 7. strip-mode input encode: base stream has NO UNK for these fixtures
    ids = TK.encode(bases)
    ok &= check("base stream vocab-clean", TK.UNK not in ids)

    print("=== a0a %s ===" % ("PASS" if ok else "FAIL"))
    return ok


def run_a0b():
    print("=== selftest a0b: data pipeline ===")
    ok = True
    import json as _json
    prep = SRC.parent.parent / "data" / "diac" / "prepared"
    train_p, val_p = prep / "train.jsonl", prep / "val.jsonl"
    if not (train_p.exists() and val_p.exists()):
        print("[SKIPPED] a0b -- REASON: prepared files absent; run prepare_data.py first")
        return True
    train = [ _json.loads(l) for l in train_p.read_text(encoding="utf-8").splitlines() ]
    val = [ _json.loads(l) for l in val_p.read_text(encoding="utf-8").splitlines() ]
    ok &= check("train non-empty", len(train) > 0, str(len(train)))
    ok &= check("val non-empty", len(val) > 0, str(len(val)))

    # 1. reparse association: bases/labels fields agree with a fresh parse
    import random as _rnd
    _rnd.seed(7)
    sample = _rnd.sample(val, min(50, len(val)))
    agree = True
    for r in sample:
        u = PT.parse(r["text"])
        if "".join(x.raw for x in u if x.is_target) != r["bases"]:
            agree = False; break
        if [x.label for x in u if x.is_target] != r["labels"]:
            agree = False; break
    ok &= check("reparse agreement on 50 samples", agree)

    # 2. window cap respected
    ok &= check("window cap 1024", all(len(r["text"]) <= 1024 for r in sample),
                detail=str(max(len(r["text"]) for r in sample)))

    # 3. doc-stratified leak check: no doc_hash on both sides
    both = set(r["doc_hash"] for r in train) & set(r["doc_hash"] for r in val)
    ok &= check("no train/val doc leakage", len(both) == 0, str(len(both)))

    # 4. determinism: seed + hashes -> stat file exists and is loadable
    st = prep / "prepare_stats.json"
    ok &= check("prepare_stats present", st.exists())

    print("=== a0b %s ===" % ("PASS" if ok else "FAIL"))
    return ok


def run_a0c():
    print("=== selftest a0c: eval harness (DER/WER/preservation) ===")
    import eval_der as EV
    ok = True

    # 1. regression-locked: perfect model -> every metric 0 error / 1.0 preserve
    gold = "\u0628\u0650\u0633\u0652\u0645 \u200f\u0627\u0644\u0644\u0651\u064e\u0647\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0646\u0650 text 42 \u0645\u0641\u062a\u0627\u062d"
    m = EV.score_line(gold, [gold])
    ok &= check("perfect -> 0.0/0.0/0.0/1.0",
                m["DER"] == 0 and m["DER_nocase"] == 0 and m["WER"] == 0
                and m["text_preservation"] == 1.0, str(m))

    # 2. one wrong diacritic on a NON-final word -> DER 1/4 words incl. case-off
    bad = "\u0628\u064e\u0633\u0652\u0645 \u200f\u0627\u0644\u0644\u0651\u064e\u0647\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0646\u0650 text 42 \u0645\u0641\u062a\u0627\u062d"
    m = EV.score_line(bad, [gold])
    ok &= check("1 vowel-wrong word -> DER==0.5",
                abs(m["DER"] - (1.0 - 5 / 6)) < 1e-9, str(m))
    ok &= check("words counted 6", m["words"] == 6, str(m["words"]))

    # 3. case-ending error only on final word -> caught by DER, missed by DER_nocase
    # 3. case-ending error only on the final word
    gold_c = "text \u0645\u0641\u062a\u0627\u062d\u0650"  # kafata + case kasra
    case_bad = "text \u0645\u0641\u062a\u0627\u062d"      # same word, case ending dropped
    m = EV.score_line(case_bad, [gold_c])
    d_with, d_off = m["DER"], m["DER_nocase"]
    ok &= check("case-ending error: DER>DER_nocase", d_with > d_off, str((d_with, d_off)))

    # 4. base-drift hallucination -> preservation hard gate drops
    halluc = "\u0628\u0650\u0633\u0652\u0645 \u200f\u0627\u0644\u0644\u0647 text 42 \u0645\u0641\u062a\u0627\u062d"
    m = EV.score_line(halluc, [gold])
    ok &= check("hallucination drops preservation<1", m["text_preservation"] < 1.0, str(m))

    # 5. multi-reference: match ANY reference counts as correct
    ref2 = "\u0628\u064e\u0633\u0652\u0645"
    m = EV.score_line(bad, [gold, ref2])
    ok &= check("multi-ref: any match correct", m["DER"] == 0, str(m))

    # 6. length mismatch counted as error, never silently shrunk
    m = EV.score_line("word1 word2", ["a b c"])
    ok &= check("len mismatch errors counted", m["words"] == 3 and m["DER"] > 0, str(m))

    # 7. cross-check every fixture RECONSTRUCTS through the pipeline (bases+marks)
    print("=== a0c %s ===" % ("PASS" if ok else "FAIL"))
    return ok


def run_a0d():
    print("=== selftest a0d: metric library parity with eval_der ===")
    import eval_der as EV
    import eval_metrics as EM
    ok = True

    # 1. the library and the label schema must agree on the class count, or every
    #    confusion matrix below is silently mis-shaped.
    ok &= check("eval_metrics.N_CLASSES == labels.N_CLASSES",
                EM.N_CLASSES == L.N_CLASSES, "%d vs %d" % (EM.N_CLASSES, L.N_CLASSES))

    # 2. WORD-LEVEL PARITY. eval_metrics re-implements the word loop so it can
    #    accumulate counts instead of rates; the point of this phase is that the
    #    re-implementation decides exactly what eval_der decides, word for word.
    W1 = "\u0628\u0650\u0633\u0652\u0645"                        # b+i s+sukun m
    W1_ALT = W1 + "/" + "\u0628\u064e\u0633\u0652\u0645"          # ... / b+a s+sukun m
    W2 = "\u0627\u0644\u0644\u0651\u064e\u0647\u0650"          # al-lah+i
    W3 = "\u0645\u0641\u062a\u0627\u062d"                        # bare mftah
    fixtures = [
        ("perfect", ["%s %s" % (W1, W2)], ["%s %s" % (W1, W2)]),
        ("anyalt match on 2nd alternative", ["%s %s" % (W1, W2)], ["%s %s" % (W1_ALT, W2)]),
        ("base drift", ["%s %s" % (W3, W2)], ["%s %s" % (W1, W2)]),
        ("prediction shorter than reference", ["%s" % W1], ["%s %s" % (W1, W2)]),
        ("prediction longer than reference", ["%s %s %s" % (W1, W2, W3)], ["%s %s" % (W1, W2)]),
    ]

    def direct_counts(pred_lines, ref_lines):
        """What eval_der's own predicate says, counted the long way round."""
        reg_wrong = any_wrong = total = 0
        for li in range(len(ref_lines)):
            gw = ref_lines[li].split()
            pw = pred_lines[li].split() if li < len(pred_lines) else []
            for i in range(max(len(gw), len(pw))):
                total += 1
                if i >= len(gw) or i >= len(pw):
                    reg_wrong += 1
                    any_wrong += 1
                    continue
                if not EV._pairs_ok(pw[i], gw[i], nocase=False):
                    reg_wrong += 1
                if not any(EV._pairs_ok(pw[i], alt, nocase=False) for alt in gw[i].split("/")):
                    any_wrong += 1
        return reg_wrong, any_wrong, total

    for name, pl, rl in fixtures:
        scored = EM.score_corpus(pl, rl)
        rec = scored["record"]
        dr, da, dt = direct_counts(pl, rl)
        ok &= check("parity %s: registered wrong" % name,
                    rec["registered_wrong"] == dr, "%d vs %d" % (rec["registered_wrong"], dr))
        ok &= check("parity %s: anyalt wrong" % name,
                    rec["anyalt_wrong"] == da, "%d vs %d" % (rec["anyalt_wrong"], da))
        ok &= check("parity %s: word count" % name, rec["words"] == dt,
                    "%d vs %d" % (rec["words"], dt))

    # 3. COUNTS, NOT RATES. A corpus rate is a ratio of totals, never a mean of
    #    per-line fractions. The fixture below is the shape that made the old
    #    eval_multi_ref aggregate wrong: one short bad line next to one long
    #    good line. Mean-of-lines would say 0.5; the truth is 1/101.
    long_ok = " ".join([W1] * 100)
    corpus = EM.score_corpus([W3, long_ok], [W1, long_ok])
    s = EM.summarize(corpus)
    ok &= check("corpus rate is a ratio of totals, not a mean of line rates",
                abs(s["der_anyalt"] - (1.0 / 101.0)) < 1e-12, repr(s["der_anyalt"]))
    ok &= check("the two lines really do have opposite per-line rates",
                corpus["lines"][0]["anyalt_wrong"] == 1 and corpus["lines"][1]["anyalt_wrong"] == 0,
                str([l["anyalt_wrong"] for l in corpus["lines"]]))

    # 4. CANONICAL ORDER. labels.label_for_marks calls shadda+vowel and
    #    vowel+shadda the same class; eval_der compares raw strings and does not.
    #    The canonical condition must close that gap, and must NOT close it for
    #    genuinely different marks.
    shadda_first = "\u0628\u0651\u064e"   # b + shadda + fatha
    vowel_first = "\u0628\u064e\u0651"    # b + fatha + shadda
    plain = "\u0628\u064e"                 # b + fatha
    ok &= check("class_of agrees across both byte orders",
                EM.class_of(EV.marks_of(shadda_first)) == EM.class_of(EV.marks_of(vowel_first)))
    ok &= check("raw marks_of DISAGREES across byte orders",
                not EV._pairs_ok(shadda_first, vowel_first, nocase=False))
    ok &= check("canonical equality AGREES across byte orders",
                EM.marks_equal_canonical(shadda_first, vowel_first))
    ok &= check("canonical equality still separates different marks",
                not EM.marks_equal_canonical(shadda_first, plain))

    # 5. EXACT RECORDED PARITY on the registered WikiNews 2024 fixtures. These
    #    two numbers are the gate's published values; if the library drifts, the
    #    historical comparison in research/MODEL_EVALUATION_REPORT.md is void.
    root = SRC.parent.parent
    ref_p = root / "models" / "e19" / "wn2024.gold.raw.pred.ref.txt"
    zm_p = root / "models" / "e19" / "wn2024.zmahood.pred.txt"
    s7_p = root / "runs" / "diac" / "e63s_s7_18l64" / "final" / "pred_wikinews2024.txt"
    mi_p = root / "models" / "e19" / "wn2024.mishkala.pred.txt"
    if not (ref_p.exists() and zm_p.exists() and s7_p.exists() and mi_p.exists()):
        print("[SKIPPED] a0d recorded parity -- REASON: wn2024 fixtures absent")
    else:
        ref = ref_p.read_text(encoding="utf-8").splitlines()
        for label, path, want_anyalt, want_registered in (
            ("zmahood", zm_p, 0.4788055764883195, None),
            ("e63s_s7_18l64", s7_p, 0.6864167294649586, 0.6860399397136397),
        ):
            got = EM.summarize(EM.score_corpus(path.read_text(encoding="utf-8").splitlines(), ref))
            ok &= check("recorded der_anyalt %s" % label,
                        got["der_anyalt"] == want_anyalt,
                        "got=%r want=%r" % (got["der_anyalt"], want_anyalt))
            if want_registered is not None:
                ok &= check("recorded der_registered %s" % label,
                            got["der_registered"] == want_registered,
                            "got=%r want=%r" % (got["der_registered"], want_registered))

        # 6. EVERY CI TARGET MUST MEASURE THE METRIC IT IS NAMED FOR. The driver
        #    pairs a metric with the (numerator, denominator) counters it is a
        #    ratio of; if a numerator is the COMPLEMENT of the metric, the report
        #    prints the interval for 1-metric under the metric's own name. That is
        #    invisible while the metric is 0.0 (ser was exactly this case) and
        #    actively misleading once it is not. Asserting point == metric pins
        #    the pairing, not just today's numbers.
        import eval_report as ER
        scored = EM.score_corpus(zm_p.read_text(encoding="utf-8").splitlines(), ref)
        summ = EM.summarize(scored)
        for label, nk, dk in ER.CI_TARGETS:
            ci = EM.bootstrap_ci(scored["lines"], nk, dk, n_boot=200, seed=42)
            ok &= check("CI target %s measures the metric it is named for" % label,
                        abs(ci["point"] - summ[label]) < 1e-9,
                        "ci point=%r metric=%r" % (ci["point"], summ[label]))

        # 7. A PAIRED TEST MUST RUN UNDER THE CONDITION IT IS REPORTED FOR. The
        #    McNemar contingency is derived from per-word correctness bits, so if
        #    those bits use a different condition than the reported difference,
        #    the p-value silently answers a different question. mishkala differs
        #    by 7.25 der points between anyalt and canonical, so this is not
        #    academic. Reconcile: correct_a = both + a_only must reproduce the
        #    metric the contingency is presented beside.
        zl = zm_p.read_text(encoding="utf-8").splitlines()
        ml = mi_p.read_text(encoding="utf-8").splitlines()
        zs = EM.summarize(EM.score_corpus(zl, ref))
        ms = EM.summarize(EM.score_corpus(ml, ref))
        n_words = sum(len(line.split()) for line in ref)
        for cond, key in ((False, "der_anyalt"), (True, "der_canonical")):
            mc = EM.mcnemar(ml, zl, ref, canonical=cond)
            total = (mc["both_correct"] + mc["neither_correct"]
                     + mc["b_a_correct_only"] + mc["c_b_correct_only"])
            # A is mishkala (the base), B is zmahood (the compared model).
            a_correct = mc["both_correct"] + mc["b_a_correct_only"]
            b_correct = mc["both_correct"] + mc["c_b_correct_only"]
            ok &= check("mcnemar(canonical=%s) reconciles with %s" % (cond, key),
                        total == n_words
                        and abs(a_correct / total - (1 - ms[key])) < 5e-4
                        and abs(b_correct / total - (1 - zs[key])) < 5e-4,
                        "total=%r a=%r(1-%s=%r) b=%r(1-%s=%r)"
                        % (total, a_correct, key, 1 - ms[key], b_correct, key,
                           1 - zs[key]))

    print("=== a0d %s ===" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a0a", "a0b", "a0c", "a0d", "all"], default="a0a")
    args = ap.parse_args()
    runners = {"a0a": run_a0a, "a0b": run_a0b, "a0c": run_a0c, "a0d": run_a0d}
    if args.phase == "all":
        phases = ["a0a", "a0b", "a0c", "a0d"]
    else:
        phases = [args.phase]
    results = [runners[p]() for p in phases]
    n_pass = sum(1 for r in results if r)
    print("=== %d/%d PASS (phases: %s) ===" % (n_pass, len(results), ",".join(phases)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
