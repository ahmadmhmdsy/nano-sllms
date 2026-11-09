"""The diacritizer (D-line) evaluator suite, declared as static data.

This is a PARALLEL catalogue, deliberately not an extension of SUITES.

Why parallel. SUITES is the R1 pretrain catalogue and two tests pin that fact:
unit/test_evaluation.py requires sorted(SUITES) to be exactly the four pretrain
suites, and unit/test_handlers.py iterates SUITES.items() expecting every member
to be a pretrain suite with a pretrain handler. The diacritizer measures a
different task (Arabic diacritic restoration) on a different split, so folding
it in would make both of those statements false at once.

Why declared at all. The value of suites.py is not the four names, it is that
the metric set is a written-down contract instead of whatever a script happened
to print. This file gives the D-line the same property: every metric below is
produced by diacritizer/scripts/eval_report.py, and each definition string is
the one that driver writes into its artifact, so a report and its schema cannot
drift apart.

On thresholds. Only three bounds are declared, and each one is an invariant
rather than a quality target:

    unclassified_units   the schema has 15 classes; a base letter carrying a
                         mark outside them means the model emitted something
                         the head cannot represent. The reference and all ten
                         measured models report exactly 0.
    alignment_coverage   the class metrics divide by aligned positions, so a
                         low coverage makes them reportable-looking but
                         unrepresentative. The lowest measured value is 0.9667.
    bare_emission_rate   a model that emits nothing scores 1.0 here, so this
                         bound catches a silent model rather than a bad one.
                         The highest measured value is 0.3047.

Quality bounds for der_canonical and friends are NOT baked in as constants,
because a number invented here would silently rot. They belong to the
registered gate comparison against the deployed anchor, which is where the
repo already keeps that decision.

Careful with the direction vocabulary: Threshold.direction reads backwards on
purpose. "min" means the value may not EXCEED the bound (an upper bound), and
"max" means it may not fall below it (a lower bound). This file follows
suites.py, where val_loss uses "min" for its ceiling.
"""
from __future__ import annotations

from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold

__all__ = ["DIACRITIZER_SUITES", "diacritizer_suite_for"]

DIACRITIZER_METRICS_SUITE = SuiteSpec(
    evaluator="diacritizer_metrics",
    version="1",
    split_identity="wikinews2024_multiref",
    metrics=(
        MetricSpec(
            name="der_canonical",
            definition=(
                "word error rate against any '/' alternative with each mark run "
                "canonicalized to shadda-first order, so shadda+vowel and "
                "vowel+shadda compare equal as labels.py declares"
            ),
        ),
        MetricSpec(
            name="der_anyalt",
            definition=(
                "word error rate against any '/' alternative on raw mark strings "
                "(the registered eval_multi_ref gate semantics, which is "
                "byte-order sensitive)"
            ),
        ),
        MetricSpec(
            name="der_primary",
            definition=(
                "word error rate of the model's first alternative against the "
                "reference's first alternative"
            ),
        ),
        MetricSpec(
            name="ser",
            definition="share of lines where every word is correct",
        ),
        MetricSpec(
            name="cder",
            definition=(
                "character-level diacritic error rate: "
                "(substitutions+deletions+insertions)/gold diacritic count"
            ),
        ),
        MetricSpec(
            name="macro_f1",
            definition="unweighted mean F1 over the classes present in the reference",
        ),
        MetricSpec(
            name="position_accuracy",
            definition="share of aligned base positions whose 15-class label is correct",
        ),
        MetricSpec(
            name="worst_class_recall",
            definition=(
                "lowest per-class recall among classes present; catches rare-class "
                "collapse that an aggregate accuracy hides"
            ),
        ),
        MetricSpec(
            name="case_accuracy",
            definition=(
                "share of words whose final base letter's class matches the "
                "reference (the i'rab/case ending)"
            ),
        ),
        MetricSpec(
            name="density_ratio",
            definition=(
                "marks emitted per base position divided by marks in the "
                "reference; below 1 is systematic under-diacritization"
            ),
        ),
        MetricSpec(
            name="alignment_coverage",
            definition=(
                "share of words whose base letters aligned, i.e. the class "
                "metrics' denominator"
            ),
            threshold=Threshold(value=0.95, direction="max"),
        ),
        MetricSpec(
            name="unclassified_units",
            definition=(
                "base letters carrying a mark outside the 15-class schema; such "
                "units are excluded from the confusion matrix"
            ),
            threshold=Threshold(value=0.0, direction="min"),
        ),
        MetricSpec(
            name="bare_emission_rate",
            definition=(
                "share of positions the model left unmarked; a silent model "
                "scores 1.0 here"
            ),
            threshold=Threshold(value=0.90, direction="min"),
        ),
    ),
)

#: The whole diacritizer catalogue. Like SUITES, it is populated by
#: declaration only, so a suite that is not written here is unreachable.
DIACRITIZER_SUITES: dict = {}
DIACRITIZER_SUITES["diacritizer_metrics"] = DIACRITIZER_METRICS_SUITE


def diacritizer_suite_for(name: str) -> SuiteSpec:
    """One declared diacritizer suite, or a refusal naming what is declared."""
    if not isinstance(name, str) or name not in DIACRITIZER_SUITES:
        declared = ", ".join(sorted(DIACRITIZER_SUITES))
        raise KeyError(
            "unknown diacritizer suite: " + repr(name) + "; declared: " + declared
        )
    return DIACRITIZER_SUITES[name]
