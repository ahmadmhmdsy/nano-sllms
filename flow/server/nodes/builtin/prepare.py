"""Prepare node definition: raw-dir in, cleaned-dir out.

Owns the webui-parity MAX_ROWS hard cap (U7) moved here verbatim from
the pre-registry config_gen; the cap check runs in validate_semantic
after this node's own knob checks (config-order contract: knob misses
report before the cap, as today).
"""

from flow.server.nodes.base import NodeDefinition, PortSpec, PropSpec
from flow.server.nodes.ports import (B_FILE_LIST, CLEANED, CLEANED_DIR,
                                     RAW_DIR, T_CLEANED_DIR, T_RAW_DIR)

# Same U7 hard cap as webui/app.py MAX_ROWS: bounds net time + disk.
MAX_ROWS = 500_000


class PrepareNode(NodeDefinition):
    """prepare: dataset rows -> cleaned text files on disk."""

    kind = "prepare"
    title = "Prepare"
    category = "data"
    config_participant = True
    gate = None
    label_semantic = None
    features = ()
    ports = [
        PortSpec(RAW_DIR, "in", T_RAW_DIR, B_FILE_LIST),
        PortSpec(CLEANED_DIR, "out", T_CLEANED_DIR, B_FILE_LIST),
    ]
    # Declared sorted (min_chars, rows, val_fraction); the derived compat
    # surfaces re-sort anyway.
    props = [
        PropSpec("min_chars", "number", required=True),
        PropSpec("rows", "number", required=True),
        PropSpec("val_fraction", "number", required=True),
    ]
    required_upstream = {RAW_DIR: ("dataset", CLEANED)}

    def validate_semantic(self, node, ctx):
        """Props container + every knob + the rows hard cap.

        Error strings are exactly the pre-registry ones; the caller
        raises the first reason (sorted knob misses before the cap).
        """
        errors = self._missing_knob_errors(node)
        if not errors:
            # All knobs present: a knob-miss reason would have fired
            # first, so the cap only matters on an otherwise clean node.
            rows = node["props"]["rows"]
            if int(rows) > MAX_ROWS:
                errors.append(
                    "rows: %d is above the hard cap %d (net time + disk)"
                    % (int(rows), MAX_ROWS))
        return errors

    def build_section(self, ctx):
        """The data block's preparation knobs and slug-named dirs."""
        props = ctx.chain[self.kind]["props"]
        return {"data": {
            "rows": int(props["rows"]),
            "val_fraction": float(props["val_fraction"]),
            "dedupe": True,
            "min_chars": int(props["min_chars"]),
            "raw_dir": "data/%s/raw" % ctx.slug,
            "tokens_dir": "data/%s/tokens" % ctx.slug,
        }}


DEFINITION = PrepareNode()
