"""Infer node definition: checkpoint in, generation (chat) out.

Not a config participant (same reason as eval). Declares the
"webui-chat-button" UI feature so the frontend renders the render-only
chat shortcut from registry data instead of a kind-string check.
"""

from flow.server.nodes.base import NodeDefinition, PortSpec
from flow.server.nodes.ports import B_CHECKPOINT, CKPT_DIR, T_CKPT_DIR


class InferNode(NodeDefinition):
    """infer: checkpoint -> interactive generation (gate: gpu/cpu)."""

    kind = "infer"
    title = "Infer"
    category = "train"
    config_participant = False
    gate = "gpu/cpu"
    label_semantic = None
    features = ("webui-chat-button",)
    ports = [
        PortSpec(CKPT_DIR, "in", T_CKPT_DIR, B_CHECKPOINT),
    ]
    required_upstream = {}  # out of the config chain: no declared link


DEFINITION = InferNode()
