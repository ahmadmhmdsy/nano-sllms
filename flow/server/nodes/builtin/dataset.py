r"""Dataset node definition: the entry point of every flow MVP graph.

The node's user label IS the HF dataset name in the MVP config mapping
(label_semantic "hf-dataset-name"), so the name-shape rule moved here
verbatim from the pre-registry config_gen: _valid_dataset_name is the
single home of the HF-path-shape check now.
"""

import re

from flow.server.nodes.base import NodeDefinition, PortSpec
from flow.server.nodes.ports import B_ROW_BATCH, CLEANED, T_RAW_DATASET

# Basic dataset-name shape: segments of letters/digits/. _ - with at most one
# '/' (org/name) or a single name (see _valid_dataset_name).
_HF_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _valid_dataset_name(name):
    """HF-path-shape check: one segment or org/name, no empty segments."""
    parts = name.split("/")
    if len(parts) > 2:
        return False
    return all(part and _HF_NAME_RE.fullmatch(part) for part in parts)


class DatasetNode(NodeDefinition):
    """dataset: label carries the HF dataset name; takes no editable props."""

    kind = "dataset"
    title = "Dataset"
    category = "data"
    config_participant = True
    gate = None
    label_semantic = "hf-dataset-name"
    features = ()
    ports = [PortSpec(CLEANED, "out", T_RAW_DATASET, B_ROW_BATCH)]
    required_upstream = {}  # chain head: nothing feeds the dataset node

    def validate_semantic(self, node, ctx):
        """The label must be a non-empty HF-path-shaped dataset name.

        Produces exactly the pre-registry error strings (golden-locked);
        the caller raises the first reason.
        """
        label = node.get("label")
        if not isinstance(label, str) or not label.strip():
            return [
                "dataset name: node '%s' has no label - the dataset node's "
                "label is the HF dataset name in this MVP" % node.get("id")]
        ds_name = label.strip()
        if not _valid_dataset_name(ds_name):
            # Beyond non-empty: a basic HF path shape (single name or
            # org/name; letters, digits, '.', '_', '-').
            return [
                "dataset name: node '%s' label %r is not a valid dataset "
                "name - expected a single name or org/name using only "
                "letters, digits, '.', '_' and '-' with at most one '/'"
                % (node.get("id"), ds_name)]
        return []

    def build_section(self, ctx):
        """The data block's dataset_candidates entry only."""
        node = ctx.chain[self.kind]
        return {"data": {"dataset_candidates": [
            {"name": node["label"].strip()}]}}


DEFINITION = DatasetNode()
