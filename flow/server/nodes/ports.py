"""Port names, types and the compat rule, named once (promotion T1).

The MVP backend validates edges against port NAMES only (graph_schema
checks toPort/fromPort against the registry PORTS table); there is no
type enforcement yet and this module must not change today's behavior.
The PortSpec type strings that builtin definitions declare live here so
future work (F5 execution, frontend portTypes.ts) compares against data,
never against magic strings.
"""

# --- port names (edge endpoints, serialized in .flow.json) ---------------
CLEANED = "cleaned"
RAW_DIR = "raw-dir"
CLEANED_DIR = "cleaned-dir"
SHARD_DIR = "shard-dir"
CKPT_DIR = "ckpt-dir"
REPORT = "report"

# --- port types (declared per PortSpec; port_compatible rule input) -------
T_RAW_DATASET = "raw-dir: dataset name/rows"
T_RAW_DIR = "raw-dir"
T_CLEANED_DIR = "cleaned-dir"
T_SHARD_DIR = "shard-dir"
T_CKPT_DIR = "ckpt-dir"
T_REPORT = "report"

# --- burst formats (how a port's data streams, display/transport hint) ----
B_ROW_BATCH = "row-batch"
B_FILE_LIST = "file-list"
B_CHECKPOINT = "checkpoint"
B_SINGLE_FILE = "single-file"


def port_compatible(source_type, target_type):
    """True when a source port type may feed a target port.

    Strict equality helper, named once: the backend does not yet enforce
    edge port TYPES (names only), so nothing calls this today - it is the
    ready replacement when F5 execution lands, not a behavior change.
    """
    return source_type == target_type
