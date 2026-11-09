"""Reusable full-metric evaluation report for the D-line (CPU only).

One driver for every diacritization metric the line reports, so a benchmark
run is a command rather than a bespoke script. Reusable by construction:

  * the metric set lives in diacritizer/src/eval_metrics.py, not here;
  * the model set is declared in a manifest file or on the command line;
  * the artifact records the exact command, the reference hash, the
    declared conditions, and every metric definition, so a reader can tell
    what a number means without reading this source.

Usage (always through the repo venv):

  .venv\\Scripts\\python.exe diacritizer\\scripts\\eval_report.py ^
      --ref models\\e19\\wn2024.gold.raw.pred.ref.txt ^
      --manifest research\\wn2024_models.json ^
      --out-json research\\wn2024_full_metrics.json ^
      --out-md research\\wn2024_full_metrics.md

  # or without a manifest:
  ... --pred zmahood=models\\e19\\wn2024.zmahood.pred.txt --pred s7=runs\\...\\pred.txt

The manifest is a JSON list: [{"model": "...", "pred": "..."}, ...].
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as EM  # noqa: E402

#: Every metric this driver reports, with the definition that goes into the
#: artifact. A metric that is not declared here is not reported.
METRIC_DEFINITIONS = {
    "der_registered": "word error rate, whole '/' token, marks vs the single raw reference line (eval_der parity)",
    "der_nocase_registered": "as der_registered with the trailing case-ending run dropped",
    "wer_registered": "word error rate on exact token equality against the raw reference line",
    "preserve_registered": "share of words whose base letters equal the raw reference's",
    "der_anyalt": "word error rate, marks compared against ANY '/' alternative (registered gate semantics)",
    "der_anyalt_nocase": "as der_anyalt with the trailing case-ending run dropped",
    "der_canonical": "as der_anyalt but with each mark run rewritten to shadda-first order first, so shadda+vowel and vowel+shadda compare equal (the equivalence labels.py declares)",
    "der_canonical_primary": "as der_primary but with shadda/vowel byte order canonicalized",
    "der_primary": "word error rate of the model's FIRST alternative against the reference's FIRST alternative",
    "der_primary_nocase": "as der_primary with the trailing case-ending run dropped",
    "preserve_primary": "share of words whose first-alternative base letters equal the reference's first alternative",
    "word_accuracy_primary": "1 - der_primary",
    "ser": "sentence exact-match rate: share of lines where EVERY word is correct under der_anyalt",
    "der_ambiguous": "word error rate restricted to words carrying '/' alternatives",
    "der_unambiguous": "word error rate restricted to words with a single reference form",
    "case_accuracy": "share of words whose FINAL base letter's class matches the reference (i'rab/case ending)",
    "case_error_rate": "1 - case_accuracy",
    "case_contribution_to_der": "der_primary - der_primary_nocase, the case ending's share of word errors",
    "cder": "character-level diacritic error rate: (substitutions+deletions+insertions)/gold diacritic count",
    "cder_per_position": "mark edit operations divided by base-letter positions",
    "mark_precision": "correct marks / marks emitted",
    "mark_recall": "correct marks / marks in the reference",
    "mark_f1": "harmonic mean of mark_precision and mark_recall",
    "mark_substitutions": "diacritics emitted but wrong",
    "mark_deletions": "reference diacritics omitted",
    "mark_insertions": "diacritics emitted where the reference has none",
    "mark_correct": "diacritics emitted at the right position with the right class",
    "gold_marks": "classifiable diacritics in the reference (the CDER and recall denominator)",
    "pred_marks": "classifiable diacritics the model emitted (the precision and density denominator)",
    "density_pred": "marks emitted per base-letter position",
    "density_gold": "marks present in the reference per base-letter position",
    "density_ratio": "density_pred / density_gold; below 1 means systematic under-diacritization",
    "over_diacritization": "share of positions where the model marked a bare reference letter",
    "under_diacritization": "share of positions where the model left a marked reference letter bare",
    "bare_emission_rate": "share of positions the model left unmarked (a silent model scores 1.0 here)",
    "macro_f1": "unweighted mean F1 over the classes present in the reference",
    "weighted_f1": "support-weighted mean F1 over the classes present in the reference",
    "micro_f1": "equals position_accuracy for single-label multiclass",
    "position_accuracy": "share of aligned base positions whose 15-class label is correct",
    "cohen_kappa": "agreement corrected for chance agreement",
    "matthews_cc": "multiclass Matthews correlation coefficient",
    "mean_class_recall": "unweighted mean per-class recall",
    "worst_class_recall": "lowest per-class recall among classes present; catches rare-class collapse",
    "collapsed_classes": "classes present in the reference with recall exactly 0.0",
    "alignment_coverage": "share of words whose base letters aligned, i.e. the class metrics' denominator",
    "unclassified_units": "base letters carrying a non-classifiable mark (passthrough); excluded from the confusion matrix",
}

#: Bootstrap intervals are reported for these (numerator, denominator) pairs.
CI_TARGETS = (
    ("der_anyalt", "anyalt_wrong", "words"),
    ("der_canonical", "canonical_wrong", "words"),
    ("der_primary", "primary_wrong", "words"),
    ("der_registered", "registered_wrong", "words"),
    # exact_lines (the POSITIVE indicator), not its complement: a CI labelled
    # "ser" must be the interval for ser. Pairing it with a "not exact" counter
    # prints the interval for 1-ser under the name ser, which is invisible while
    # ser is 0.0 and actively misleading as soon as it is not.
    ("ser", "exact_lines", "line_unit"),
    ("cder", "mark_errors", "gold_marks"),
)

#: Metrics that can drive the ranking, baseline choice, and paired test, each
#: mapped to the (numerator, denominator) counters it is a ratio of.
RANK_TARGETS = {
    "der_anyalt": ("anyalt_wrong", "words"),
    "der_canonical": ("canonical_wrong", "words"),
    "der_primary": ("primary_wrong", "words"),
    "der_registered": ("registered_wrong", "words"),
}

HEADLINE = (
    "der_anyalt", "der_canonical", "der_primary", "der_primary_nocase", "ser", "cder",
    "mark_f1", "macro_f1", "position_accuracy", "cohen_kappa",
    "worst_class_recall", "density_ratio", "case_accuracy",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(args):
    models = []
    if args.manifest:
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for row in data:
            models.append((row["model"], row["pred"]))
    for item in args.pred or []:
        if "=" not in item:
            raise SystemExit("--pred needs NAME=PATH, got: " + item)
        name, path = item.split("=", 1)
        models.append((name, path))
    if not models:
        raise SystemExit("no models: pass --manifest or at least one --pred NAME=PATH")
    return models


def pct(v):
    return "n/a" if v is None else "%.2f%%" % (100.0 * v)


def fmt_p(p):
    """Exact McNemar p-values underflow to 0.0 for large corpora; say so."""
    if p is None:
        return "n/a"
    if p == 0.0:
        return "< 1e-308"
    return "%.3g" % p


def num(v, nd=4):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def build_markdown(artifact):
    L = []
    A = artifact
    L.append("# Diacritizer full-metric evaluation")
    L.append("")
    L.append("Generated %s by `%s`." % (A["generated"], A["generator"]))
    L.append("")
    L.append("Reference: `%s` (%d lines, %d words, sha256 `%s`)."
             % (A["ref"]["path"], A["ref"]["lines"], A["ref"]["words"], A["ref"]["sha256"][:16]))
    L.append("")
    L.append("Command: `%s`" % A["command"])
    L.append("")
    L.append("## Conditions")
    L.append("")
    L.append("The WikiNews 2024 reference carries `/` alternatives inside a word, so")
    L.append('"DER" is ambiguous unless the comparison condition is named. Every table')
    L.append("below states which one it uses.")
    L.append("")
    L.append("| condition | meaning |")
    L.append("|---|---|")
    L.append("| registered | whole `/` token, marks vs the single raw reference line (eval_der) |")
    L.append("| anyalt | whole `/` token, marks vs ANY alternative (registered gate / eval_multi_ref) |")
    L.append("| primary | the model's FIRST alternative vs the reference's FIRST alternative |")
    L.append("| canonical | as anyalt, but each mark run is rewritten to shadda-first order first |")
    L.append("")
    L.append("### Shadda/vowel byte order")
    L.append("")
    L.append("labels.label_for_marks accepts a shadda+vowel run in EITHER byte order and")
    L.append("maps both to a single class; its own docstring records that real corpora use")
    L.append("both. eval_der compares raw mark STRINGS, so a vowel-first prediction is")
    L.append("scored wrong against a shadda-first reference even though the schema calls")
    L.append("the two identical. The canonical condition canonicalizes the order first, so")
    L.append("the delta below is exactly the cost of that byte order - and it is large")
    L.append("enough to change which model ranks first.")
    L.append("")
    L.append("| model | der_anyalt | der_canonical | delta |")
    L.append("|---|---|---|---|")
    for m in A["models"]:
        s = m["metrics"]
        d, c = s.get("der_anyalt"), s.get("der_canonical")
        L.append("| %s | %s | %s | %s |"
                 % (m["model"], pct(d), pct(c),
                    "n/a" if d is None or c is None else pct(d - c)))
    L.append("")
    L.append("## Headline metrics")
    L.append("")
    cols = ["model"] + list(HEADLINE)
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for m in A["models"]:
        s = m["metrics"]
        cells = [m["model"]]
        for k in HEADLINE:
            cells.append(pct(s.get(k)) if k not in ("density_ratio", "cohen_kappa") else num(s.get(k), 3))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Confidence intervals (line bootstrap, %d resamples)" % A["bootstrap_n"])
    L.append("")
    L.append("Lines are the independent unit: a sentence is annotated as a whole, so the")
    L.append("bootstrap resamples lines and re-weights by each line's own word count.")
    L.append("")
    L.append("Ranking and the paired test below use %s." % A["rank_by"])
    L.append("")
    L.append("| model | metric | point | 95% CI |")
    L.append("|---|---|---|---|")
    for m in A["models"]:
        for name, ci in m["cis"].items():
            L.append("| %s | %s | %s | [%s, %s] |"
                     % (m["model"], name, num(ci["point"]),
                        num(ci["lo"]), num(ci["hi"])))
    L.append("")
    L.append("## Pairwise significance vs `%s` on `%s` (paired, per word)"
             % (A["baseline"], A["rank_by"]))
    L.append("")
    L.append("Exact two-sided McNemar on discordant words, plus a paired line bootstrap of")
    L.append("the difference. Both use the same sentences for both systems, which is what")
    L.append("makes them valid where comparing two independent intervals would not be.")
    L.append("")
    L.append("| model | diff %s | 95%% CI | excludes 0 | McNemar p | significant |" % A["rank_by"])
    L.append("|---|---|---|---|---|---|")
    for p in A["pairwise"]:
        L.append("| %s | %s | [%s, %s] | %s | %s | %s |"
                 % (p["model"], num(p["difference"]), num(p["paired_bootstrap"]["lo"]),
                    num(p["paired_bootstrap"]["hi"]),
                    "yes" if p["paired_bootstrap"]["excludes_zero"] else "no",
                    fmt_p(p["mcnemar"]["p_value_two_sided"]),
                    "yes" if p["mcnemar"]["significant_at_0.05"] else "no"))
    L.append("")
    L.append("## Per-class recall (primary condition)")
    L.append("")
    L.append("Support is the reference count of each class; a recall of 0.00% on a class")
    L.append("with support is a collapsed class that an accuracy figure would hide.")
    L.append("")
    L.append("Support is taken as the largest count any model aligned, i.e. the")
    L.append("reference count. It is NOT one model's denominator: class metrics are")
    L.append("computed over each model's ALIGNED subset and alignment coverage differs")
    L.append("(see alignment_coverage in the headline table), so a model that fails to")
    L.append("align a word also drops it from that class's denominator. Reading one")
    L.append("model's support beside another model's recall silently compares two")
    L.append("different denominators.")
    L.append("")
    names = [pc["class"] for pc in A["models"][0]["per_class"]]
    sup = {}
    for _mm in A["models"]:
        for _pc in _mm["per_class"]:
            sup[_pc["class"]] = max(sup.get(_pc["class"], 0), _pc["support"])
    L.append("| class | support | " + " | ".join(m["model"] for m in A["models"]) + " |")
    L.append("|" + "---|" * (len(A["models"]) + 2))
    for cname in names:
        row = ["`%s`" % cname, str(sup[cname])]
        for m in A["models"]:
            pc = next(p for p in m["per_class"] if p["class"] == cname)
            row.append(pct(pc["recall"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## Collapsed classes")
    L.append("")
    L.append("| model | classes present | collapsed (recall 0.00%) | worst class recall |")
    L.append("|---|---|---|---|")
    for m in A["models"]:
        s = m["metrics"]
        L.append("| %s | %d | %s | %s |"
                 % (m["model"], s["classes_present"],
                    ", ".join("`%s`" % c for c in s["collapsed_classes"]) or "none",
                    pct(s["worst_class_recall"])))
    L.append("")
    L.append("## Confusion matrix - `%s` (gold rows x predicted columns)" % A["baseline"])
    L.append("")
    base = next(m for m in A["models"] if m["model"] == A["baseline"])
    conf = base["confusion"]
    short = [c[:9] for c in names]
    L.append("| gold \\ pred | " + " | ".join(short) + " |")
    L.append("|" + "---|" * (len(short) + 1))
    for i, cname in enumerate(names):
        L.append("| `%s` | " % cname + " | ".join(str(conf[i][j]) for j in range(len(names))) + " |")
    L.append("")
    L.append("## Metric definitions")
    L.append("")
    L.append("| metric | definition |")
    L.append("|---|---|")
    for k, v in A["metric_definitions"].items():
        L.append("| `%s` | %s |" % (k, v))
    L.append("")
    L.append("## Families implemented but not computed here")
    L.append("")
    for k, v in A["not_computed"].items():
        L.append("- **%s** - %s" % (k, v))
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--pred", action="append")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md")
    ap.add_argument("--baseline")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--rank-by", default="der_anyalt", choices=sorted(RANK_TARGETS),
                    help="which condition drives the ranking and the paired test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.is_absolute():
        ref_path = REPO / ref_path
    ref_lines = ref_path.read_text(encoding="utf-8").splitlines()

    models = load_models(args)
    results = []
    t0 = time.time()
    for name, rel in models:
        path = Path(rel)
        if not path.is_absolute():
            path = REPO / path
        pred_lines = path.read_text(encoding="utf-8").splitlines()
        scored = EM.score_corpus(pred_lines, ref_lines)
        s = EM.summarize(scored)
        lines = scored["lines"]
        cis = {}
        for label, nk, dk in CI_TARGETS:
            cis[label] = EM.bootstrap_ci(lines, nk, dk, n_boot=args.bootstrap, seed=args.seed)
        results.append({
            "model": name,
            "pred": str(path.relative_to(REPO)).replace("\\", "/"),
            "pred_sha256": sha256_file(path),
            "metrics": {k: v for k, v in s.items() if k not in ("per_class", "confusion")},
            "per_class": s["per_class"],
            "confusion": s["confusion"],
            "cis": cis,
            "lines": lines,
        })
        print("%-22s der_anyalt=%.4f  der_canonical=%.4f  macro_f1=%.4f  ser=%.4f  cder=%.4f"
              % (name, s["der_anyalt"], s["der_canonical"], s["macro_f1"], s["ser"], s["cder"]))

    rank_nk, rank_dk = RANK_TARGETS[args.rank_by]
    results.sort(key=lambda m: m["metrics"][args.rank_by])
    baseline = args.baseline or results[0]["model"]
    base = next(m for m in results if m["model"] == baseline)

    pairwise = []
    for m in results:
        if m["model"] == baseline:
            continue
        pb = EM.paired_bootstrap(base["lines"], m["lines"], rank_nk, rank_dk,
                                 n_boot=args.bootstrap, seed=args.seed)
        base_pred = (REPO / base["pred"]).read_text(encoding="utf-8").splitlines()
        other_pred = (REPO / m["pred"]).read_text(encoding="utf-8").splitlines()
        # The paired test must run under the SAME condition as the difference
        # it is printed beside, or the p-value answers a different question
        # than the table header claims.
        mc = EM.mcnemar(base_pred, other_pred, ref_lines,
                        canonical=(rank_nk == "canonical_wrong"))
        pairwise.append({"model": m["model"], "difference": pb["point"],
                         "paired_bootstrap": pb, "mcnemar": mc})

    artifact = {
        "artifact": "Diacritizer full-metric evaluation (WikiNews 2024)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "diacritizer/scripts/eval_report.py",
        "command": " ".join(sys.argv),
        "elapsed_s": round(time.time() - t0, 1),
        "ref": {
            "path": str(ref_path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_file(ref_path),
            "lines": len(ref_lines),
            "words": sum(len(l.split()) for l in ref_lines),
        },
        "conditions": ["registered", "anyalt", "primary", "canonical", "nocase"],
        "rank_by": args.rank_by,
        "bootstrap_n": args.bootstrap,
        "baseline": baseline,
        "metric_definitions": METRIC_DEFINITIONS,
        "models": [{k: v for k, v in m.items() if k != "lines"} for m in results],
        "pairwise": pairwise,
        "not_computed": {
            # The blocker for these three is a MISSING DUMP PATH, not hardware:
            # eval.py is compare-only and bench.py discards logits after argmax.
            # A forward pass over these 356 sentences is seconds on CPU, so the
            # fix is a small logits dumper, not a GPU window. Note also that
            # mishkala/zmahood ship no local checkpoint, so these families can
            # never cover the two published baselines - only our own 8 models.
            "top_k_accuracy": "implemented as eval_metrics.topk_accuracy; needs a per-position logits dump that no script produces yet (CPU-sufficient)",
            "calibration_ece_nll_brier": "implemented as eval_metrics.expected_calibration_error / negative_log_likelihood / brier_score; needs the same missing logits dump (CPU-sufficient)",
            "auc_one_vs_rest": "implemented as eval_metrics.auc_ovr; needs the same missing logits dump (CPU-sufficient)",
            "latency_throughput": "needs the model loaded; not derivable from decoded text",
            "oov_conditioned_der": "needs the training vocabulary; requires tokenizer + train split",
            "noise_robustness": "needs re-inference on perturbed input",
            "idempotence": "needs re-inference on the model's own output",
            "human_rating": "out of scope for an automated harness",
        },
    }
    # full per-line records stay out of the JSON (they are large); the
    # intervals above are computed from them.
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = REPO / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote " + str(out_json))

    if args.out_md:
        out_md = Path(args.out_md)
        if not out_md.is_absolute():
            out_md = REPO / out_md
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(artifact), encoding="utf-8")
        print("wrote " + str(out_md))


if __name__ == "__main__":
    main()
