"""Built-in node definitions, listed in the fixed canonical order.

Every builtin module exposes one DEFINITION singleton; DEFINITIONS is
the canonical registration order (dataset, prepare, tokenize, train,
eval, infer) that flow.server.nodes relies on for both the legacy
compat surface and the config-order contract.
"""

from . import dataset, eval_job, infer, prepare, tokenize, train  # noqa: F401

DEFINITIONS = [
    dataset.DEFINITION,
    prepare.DEFINITION,
    tokenize.DEFINITION,
    train.DEFINITION,
    eval_job.DEFINITION,
    infer.DEFINITION,
]
