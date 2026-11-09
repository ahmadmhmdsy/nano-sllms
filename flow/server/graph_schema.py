"""Pure validation of .flow.json graph documents (flow/ MVP Task 2).

`validate(g) -> list[str]` performs NO I/O and returns [] for a valid
document, otherwise one human-readable reason string per violation.

The node-kind registry (flow.server.nodes, Task 3) is the source of truth
for kinds AND per-kind ports, reached through the lazy-import hooks
_known_kinds() and _known_ports(): when the registry answers, toPort and
fromPort are both checked against the endpoint kinds' ports and the
registry-free "out*"/"in*" prefix heuristic is skipped.
"""

# Placeholder valid kinds. The real registry (flow/server/nodes/ package)
# is the source of truth; the lazy import in _known_kinds() reads from it
# and this set is the fallback if the package is missing/unstable.
VALID_KINDS_PLACEHOLDER = frozenset(
    {"dataset", "prepare", "tokenize", "train", "eval", "infer"}
)

SCHEMA_STRING = "flow/0.1"

# Strings that show up as unfilled editor placeholders and must never be
# stored in a saved document.
_PLACEHOLDER_PORTS = frozenset({"", "<from_port>", "<to_port>", "todo", "?", "?"})


def _known_kinds() -> "frozenset | None":
    """Return the set of valid node kinds, or None when unavailable.

    Lazy-imports the Task 3 registry so graph_schema never depends on it
    existing. While the registry is missing/unstable we fall back to the
    placeholder set above.
    """
    try:  # lazy: the registry package (flow/server/nodes/)
        from flow.server import nodes as _registry  # noqa: F401  # type: ignore

        kinds = getattr(_registry, "VALID_KINDS", None)
        if isinstance(kinds, (set, frozenset)) and kinds:
            return frozenset(kinds)
    except (ImportError, AttributeError):
        pass
    return None


def _known_ports(kind: str):
    """Return (inputs, outputs) port-name sets for a kind, if the registry
    can answer. None means "kind ports cannot be checked yet" (fallback
    until Task 3)."""
    try:
        from flow.server import nodes as _registry  # type: ignore

        ports = getattr(_registry, "PORTS", None)
        if isinstance(ports, dict) and kind in ports:
            entry = ports[kind]
            return (frozenset(entry.get("in", ())), frozenset(entry.get("out", ())))
    except (ImportError, AttributeError):
        pass
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bad_port_name(name) -> bool:
    if not isinstance(name, str):
        return True
    return name.strip().lower() in _PLACEHOLDER_PORTS


def validate(g) -> "list[str]":
    """Validate a .flow.json document. Pure function: no I/O, returns []."""
    errors = _validate(g)
    return errors


def _validate(g) -> "list[str]":
    if not isinstance(g, dict):
        return ["document: expected a JSON object"]

    errors = []  # never mutate list lazily; append in order

    # --- schema -------------------------------------------------------
    schema = g.get("schema")
    if not isinstance(schema, str):
        errors.append(
            "schema: document must declare schema string %r" % SCHEMA_STRING
        )
    elif schema != SCHEMA_STRING:
        errors.append(
            "schema: unsupported schema %r — only %r is accepted; "
            "forward-migration to newer documents is not supported, "
            "re-save the graph with %r" % (schema, SCHEMA_STRING, SCHEMA_STRING)
        )

    # --- meta ----------------------------------------------------------
    meta = g.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta: missing or not an object; meta.name is required")
    else:
        name = meta.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append("meta.name: required non-empty string — missing meta.name")

    # --- graph container ------------------------------------------------
    graph = g.get("graph")
    if not isinstance(graph, dict):
        errors.append("graph: missing or not an object")
        return errors

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        errors.append("graph.nodes: missing or not a list")
        nodes = []
    elif len(nodes) == 0:
        errors.append("graph.nodes: graph is empty — at least one node is required")

    # A present-but-mistyped edges key is an error; a graph whose "edges"
    # key is entirely missing is accepted and treated as an empty edge list.
    if "edges" in graph and not isinstance(graph["edges"], list):
        errors.append("graph.edges: missing or not a list")
        edges = []
    else:
        edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

    # --- nodes -----------------------------------------------------------
    known_kinds = _known_kinds() or VALID_KINDS_PLACEHOLDER

    ids = []  # list of (id, node) preserving order
    seen_node_ids = set()
    node_by_id = {}
    for node in nodes:
        if not isinstance(node, dict):
            errors.append("graph.nodes: each node must be an object")
            continue

        nid = node.get("id")
        if nid is None:
            errors.append("graph.nodes[].id: missing node id")
            nid = None  # cannot reference this node from edges sanely
        elif not isinstance(nid, str) or not nid.strip():
            errors.append("graph.nodes[].id: node id must be a non-empty string")
            nid = None
        elif nid in seen_node_ids:
            errors.append(
                "graph.nodes: duplicate node id %r — duplicate ids are not allowed"
                % nid
            )
        else:
            seen_node_ids.add(nid)
            node_by_id[nid] = node
        ids.append(nid)

        kind = node.get("kind")
        if kind is None:
            errors.append(
                "graph.nodes[%s].kind: missing — node must declare a kind" % (nid or "?")
            )
        elif not isinstance(kind, str):
            errors.append(
                "graph.nodes[%s].kind: must be a string" % (nid or "?")
            )
        elif kind not in known_kinds:
            errors.append(
                "graph.nodes[%s]: unknown kind %r" % (nid or "?", kind)
            )

        if "props" not in node:
            errors.append(
                "graph.nodes[%s].props: missing — props may be {} but must exist"
                % (nid or "?")
            )
        elif not isinstance(node["props"], dict):
            errors.append(
                "graph.nodes[%s].props: must be an object" % (nid or "?")
            )

        pos = node.get("position")
        if not isinstance(pos, dict):
            errors.append(
                "graph.nodes[%s].position: missing — node must store position"
                % (nid or "?")
            )
        else:
            for axis in ("x", "y"):
                if axis not in pos:
                    errors.append(
                        "graph.nodes[%s].position.%s: missing" % (nid or "?", axis)
                    )
                elif not _is_number(pos[axis]):
                    errors.append(
                        "graph.nodes[%s].position.%s: must be a number"
                        % (nid or "?", axis)
                    )

    # --- edges -----------------------------------------------------------
    seen_edge_ids = set()
    adjacency = {}  # nid -> list of target nids (for Kahn)
    indegree = {nid: 0 for nid in node_by_id}
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("graph.edges: each edge must be an object")
            continue

        eid = edge.get("id")
        label = eid if isinstance(eid, str) and eid else "?"
        if eid is None:
            errors.append(f"graph.edges[{label}].id: missing edge id")
        elif not isinstance(eid, str) or not eid.strip():
            errors.append(f"graph.edges[{label}].id: edge id must be a non-empty string")
        elif eid in seen_edge_ids:
            errors.append(
                "graph.edges: duplicate edge id %r — duplicate ids are not allowed" % eid
            )
        else:
            seen_edge_ids.add(eid)

        frm = edge.get("from")
        to = edge.get("to")

        frm_ok = isinstance(frm, str) and frm in node_by_id
        to_ok = isinstance(to, str) and to in node_by_id
        if not isinstance(frm, str) or not frm:
            errors.append(f"graph.edges[{label}].from: missing source node id")
        elif not frm_ok:
            errors.append(
                "graph.edges[%s].from: edge connects from unknown node id %r"
                % (label, frm)
            )

        if not isinstance(to, str) or not to:
            errors.append(f"graph.edges[{label}].to: missing target node id")
        elif not to_ok:
            errors.append(
                "graph.edges[%s].to: edge connects to unknown node id %r"
                % (label, to)
            )

        if frm_ok and to_ok and frm == to:
            errors.append(
                "graph.edges[%s]: a node cannot be connected to itself" % label
            )

        from_port = edge.get("fromPort")
        to_port = edge.get("toPort")

        if from_port is None:
            errors.append(f"graph.edges[{label}].fromPort: missing port name")
        elif _bad_port_name(from_port):
            errors.append(
                "graph.edges[%s].fromPort: no-placeholder port names required, "
                "got %r" % (label, from_port)
            )
        if to_port is None:
            errors.append(f"graph.edges[{label}].toPort: missing port name")
        elif _bad_port_name(to_port):
            errors.append(
                "graph.edges[%s].toPort: no-placeholder port names required, "
                "got %r" % (label, to_port)
            )

        src_ports = _known_ports(node_by_id[frm].get("kind")) if frm_ok else None
        dst_ports = _known_ports(node_by_id[to].get("kind")) if to_ok else None

        if dst_ports is not None:
            inputs, _outputs = dst_ports
            if isinstance(to_port, str) and to_port and to_port not in inputs:
                errors.append(
                    "graph.edges[%s].toPort: %r is not an input port of kind %r"
                    % (label, to_port, node_by_id[to].get("kind"))
                )

        if src_ports is not None:
            _inputs, src_out = src_ports
            if isinstance(from_port, str) and from_port and from_port not in src_out:
                errors.append(
                    "graph.edges[%s].fromPort: %r is not an output port of kind %r"
                    % (label, from_port, node_by_id[frm].get("kind"))
                )

        if src_ports is None and dst_ports is None:
            # Fallback (registry-unknown) port semantics until the registry
            # can answer for BOTH endpoint kinds: an edge leaves a
            # source-side output port ("out..."-prefixed) and lands on an
            # input-side port ("in..."-prefixed). Any other pairing is a
            # mismatched connection. When the registry answers, it is the
            # sole source of truth and this heuristic is skipped.
            if isinstance(from_port, str) and not from_port.strip().lower().startswith("out"):
                errors.append(
                    "graph.edges[%s]: mismatched port names — fromPort %r "
                    "must be an output-side port of the source node"
                    % (label, from_port)
                )
            elif isinstance(to_port, str) and not to_port.strip().lower().startswith("in"):
                errors.append(
                    "graph.edges[%s]: mismatched port names — toPort %r "
                    "must be an input-side port of the target node"
                    % (label, to_port)
                )

        if frm_ok and to_ok and frm != to:
            adjacency.setdefault(frm, []).append(to)
            indegree[to] = indegree.get(to, 0) + 1

    # --- cycle detection (Kahn's algorithm) -------------------------------
    ready = [nid for nid, deg in indegree.items() if deg == 0]
    ordered = []
    while ready:
        nid = ready.pop()
        ordered.append(nid)
        for nxt in adjacency.get(nid, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(ordered) < len(indegree):
        cyclic = sorted(set(indegree) - set(ordered))
        errors.append(
            "graph.edges: cycle detected in execution graph involving node(s) %s"
            % (", ".join(repr(c) for c in cyclic))
        )

    return errors
