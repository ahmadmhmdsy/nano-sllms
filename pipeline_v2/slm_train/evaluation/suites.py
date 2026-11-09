"""The R1 evaluator suites, declared as static data.

No evaluator is discovered at runtime. The set below is the whole catalogue,
an external gate must be declared here to be reachable at all, and a metric
that is not declared here cannot appear in a report without failing the suite
completeness check.
"""
from __future__ import annotations

from slm_train.evaluation.contracts import MetricSpec, SuiteSpec, Threshold

__all__ = ["EXTERNAL_GATE_NAMES", "SUITES", "suite_for"]

EVALUATION_EPOCHS = (1, 2)

VALIDATION_SUITE = SuiteSpec(
    evaluator="validation",
    version="1",
    split_identity="validation",
    metrics=(
        MetricSpec(
            name="val_loss",
            definition="mean cross-entropy loss on the validation split",
            threshold=Threshold(value=3.0, direction="min"),
            cadence=EVALUATION_EPOCHS,
        ),
        MetricSpec(
            name="val_perplexity",
            definition="exp of the validation mean cross-entropy loss",
            threshold=Threshold(value=20.0, direction="min"),
            cadence=EVALUATION_EPOCHS,
        ),
    ),
)

GENERATION_SUITE = SuiteSpec(
    evaluator="generation",
    version="1",
    split_identity="smoke",
    metrics=(
        MetricSpec(
            name="generation_smoke",
            definition="1 if the fixed smoke prompt produced finite decoded text, else 0",
            threshold=Threshold(value=1.0, direction="min"),
            cadence=EVALUATION_EPOCHS,
        ),
    ),
)

SYNTAX_SUITE = SuiteSpec(
    evaluator="syntax",
    version="1",
    split_identity="smoke",
    metrics=(
        MetricSpec(
            name="syntax_error_rate",
            definition="fraction of decoded samples that failed the syntax check",
            threshold=Threshold(value=0.0, direction="max"),
            cadence=EVALUATION_EPOCHS,
        ),
    ),
)

RESOURCE_SUITE = SuiteSpec(
    evaluator="resources",
    version="1",
    split_identity="none",
    metrics=(
        MetricSpec(
            name="tokens_per_second",
            definition="measured training throughput in tokens per second",
        ),
        MetricSpec(
            name="peak_vram_bytes",
            definition="peak reserved accelerator memory in bytes",
        ),
    ),
)

SUITES: dict = {}
SUITES["validation"] = VALIDATION_SUITE
SUITES["generation"] = GENERATION_SUITE
SUITES["syntax"] = SYNTAX_SUITE
SUITES["resources"] = RESOURCE_SUITE

#: External gates are registered by declaration only; nothing is discovered.
#:
#: An external gate is not a second opinion from the same bookkeeping that
#: produced the artifact. Each one below re-derives a claim the run made, using
#: the published bytes and nothing the run said about them:
#:
#: checkpoint_reload
#:     Loads the published checkpoint into a fresh model and recomputes the
#:     held-out loss, then requires it to match the loss the run recorded. This
#:     is the one that matters: it fails when a checkpoint verifies by hash but
#:     does not carry the weights its own record claims.
#: artifact_integrity
#:     Re-hashes every published payload and compares it to the manifest that
#:     vouches for it, so a run cannot certify bytes it no longer has.
#: retention_policy
#:     Re-reads the retention decision from disk and requires exactly the best
#:     checkpoint plus the configured number of most recent ones.
#:
#: The set is explicit rather than populated when one appears, so a gate that
#: is not listed here is unreachable by any config.
EXTERNAL_GATE_NAMES: tuple = (
    "artifact_integrity",
    "checkpoint_reload",
    "retention_policy",
)


def suite_for(name: str) -> SuiteSpec:
    """One declared suite, or a refusal naming what is declared."""
    if not isinstance(name, str) or name not in SUITES:
        declared = ", ".join(sorted(SUITES))
        raise KeyError("unknown evaluator suite: " + repr(name) + "; declared: " + declared)
    return SUITES[name]
