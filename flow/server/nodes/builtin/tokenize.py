"""Tokenize node definition: cleaned-dir in, shard-dir out.

Owns the tokenizer identity + shard granularity constants moved here
verbatim from the pre-registry config_gen. SHARD_TOKENS surface through
this node's section (the tokenizer owns the packing granularity),
merged into the shared "data" block by config_gen.
"""

from flow.server.nodes.base import NodeDefinition, PortSpec, PropSpec
from flow.server.nodes.ports import (B_FILE_LIST, CLEANED_DIR, SHARD_DIR,
                                     T_CLEANED_DIR, T_SHARD_DIR)

# Same tokenizer + shard granularity as webui/app.py build_config (shipped
# pipeline constants; not knobs in this MVP).
TOKENIZER_NAME = "codellama/CodeLlama-7b-hf"
SHARD_TOKENS = 8_000_000


class TokenizeNode(NodeDefinition):
    """tokenize: cleaned text files -> packed training shards."""

    kind = "tokenize"
    title = "Tokenize"
    category = "data"
    config_participant = True
    gate = None
    label_semantic = None
    features = ()
    ports = [
        PortSpec(CLEANED_DIR, "in", T_CLEANED_DIR, B_FILE_LIST),
        PortSpec(SHARD_DIR, "out", T_SHARD_DIR, B_FILE_LIST),
    ]
    props = [
        PropSpec("seq_len", "number", required=True),
        PropSpec("vocab", "number", required=True),
    ]
    required_upstream = {CLEANED_DIR: ("prepare", CLEANED_DIR)}

    def validate_semantic(self, node, ctx):
        """Props container + every knob (pre-registry error strings)."""
        return self._missing_knob_errors(node)

    def build_section(self, ctx):
        """The tokenizer block + the data block's shard_tokens leaf."""
        props = ctx.chain[self.kind]["props"]
        return {
            "tokenizer": {"name": TOKENIZER_NAME,
                          "vocab_size": int(props["vocab"])},
            "data": {"shard_tokens": SHARD_TOKENS},
        }


DEFINITION = TokenizeNode()
