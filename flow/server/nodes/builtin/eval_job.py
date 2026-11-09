"""Eval node definition: checkpoint in, evaluation report out.

Not a config participant: the MVP train-tab config carries its fixed
eval block from the train node. eval exists in the registry for kinds,
ports and its gpu gate; execution wiring is future F5 work.
"""

from flow.server.nodes.base import NodeDefinition, PortSpec
from flow.server.nodes.ports import (B_CHECKPOINT, B_SINGLE_FILE, CKPT_DIR,
                                     REPORT, T_CKPT_DIR, T_REPORT)


class EvalNode(NodeDefinition):
    """eval: checkpoint -> evaluation report (gate: gpu)."""

    kind = "eval"
    title = "Evaluate"
    category = "train"
    config_participant = False
    gate = "gpu"
    label_semantic = None
    features = ()
    ports = [
        PortSpec(CKPT_DIR, "in", T_CKPT_DIR, B_CHECKPOINT),
        PortSpec(REPORT, "out", T_REPORT, B_SINGLE_FILE),
    ]
    required_upstream = {}  # out of the config chain: no declared link


DEFINITION = EvalNode()
