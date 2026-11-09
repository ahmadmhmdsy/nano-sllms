"""Node-kind registry for the flow/ MVP (registry promotion T1).

Single source of truth for node kinds, per-kind ports, editable props,
gates and now per-kind BEHAVIOR: every built-in kind is a NodeDefinition
(flow/server/nodes/base.py) registered in the fixed canonical order
dataset, prepare, tokenize, train, eval, infer. The legacy compat
surface (VALID_KINDS / PORTS / PORT_SPECS / PROPS / NUMERIC_PROPS /
GATES / NODES / validate_props) is DERIVED from the registry so the
lazy-importing graph_schema, flows.py, app.py and /api/nodes keep
their exact dict shapes (plus the label_semantic / features snapshot
fields the T3 frontend contract expects).

Pure data + pure validation - no I/O.
"""

import math

from flow.server.nodes import builtin  # noqa: F401 - registers builtins
from flow.server.nodes.base import NodeDefinition, PortSpec, PropSpec  # noqa: F401


class NodeRegistryError(Exception):
    """Registry misuse: unknown kind on get(), duplicate on register."""


class NodeRegistry:
    """Insertion-ordered registry of NodeDefinitions keyed by kind."""

    def __init__(self):
        self._definitions = {}  # kind -> NodeDefinition (insertion order)

    def register(self, defn):
        """Register one NodeDefinition; duplicate kind is a registry bug."""
        if not isinstance(defn, NodeDefinition):
            raise NodeRegistryError(
                "register() wants a NodeDefinition instance, got %r"
                % (defn,))
        if not defn.kind:
            raise NodeRegistryError(
                "node definition needs a non-empty kind")
        if defn.kind in self._definitions:
            raise NodeRegistryError(
                "node kind %r registered twice" % (defn.kind,))
        self._definitions[defn.kind] = defn

    def get(self, kind):
        """The definition for kind; NodeRegistryError when unknown."""
        try:
            return self._definitions[kind]
        except KeyError:
            raise NodeRegistryError(
                "unknown node kind %r (registered: %s)"
                % (kind, ", ".join(self.kinds()))) from None

    def kinds(self):
        """Registered kinds in canonical registration order."""
        return tuple(self._definitions)

    def definitions(self):
        """Registered definitions in canonical registration order."""
        return tuple(self._definitions.values())


REGISTRY = NodeRegistry()
for _definition in builtin.DEFINITIONS:  # canonical order: nodes.builtin
    REGISTRY.register(_definition)

# --- legacy compat surface (derived from the registry, same shapes) ------
VALID_KINDS = frozenset(REGISTRY.kinds())

PORT_SPECS = {kind: [port.to_dict() for port in REGISTRY.get(kind).ports]
              for kind in REGISTRY.kinds()}

# The shape graph_schema expects: kind -> {"in": [names], "out": [names]}.
PORTS = {
    kind: {
        "in": [p["name"] for p in specs if p["direction"] == "in"],
        "out": [p["name"] for p in specs if p["direction"] == "out"],
    }
    for kind, specs in PORT_SPECS.items()
}

# Allowed prop keys per kind (dataset/eval/infer take no editable props).
PROPS = {kind: frozenset(spec.name for spec in REGISTRY.get(kind).props)
         for kind in REGISTRY.kinds()}

# Props that, when present, must be finite real numbers (bool excluded);
# derived from the declared prop type ("number" is the numeric contract).
NUMERIC_PROPS = {
    kind: frozenset(spec.name for spec in REGISTRY.get(kind).props
                    if spec.type == "number")
    for kind in REGISTRY.kinds()
}

# Execution gate per kind; None means no gate (dataset row wins: gate none).
GATES = {kind: REGISTRY.get(kind).gate for kind in REGISTRY.kinds()}

NODES = {
    kind: {
        "ports": specs,
        "props": sorted(PROPS[kind]),
        "numeric_props": sorted(NUMERIC_PROPS[kind]),
        "gate": GATES[kind],
        "label_semantic": REGISTRY.get(kind).label_semantic,
        "features": list(REGISTRY.get(kind).features),
    }
    for kind, specs in PORT_SPECS.items()
}


def _finite_number(value) -> bool:
    """True for finite real numbers; bools and NaN/inf excluded."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return bool(math.isfinite(value))


def validate_props(kind, props):
    """Validate props for kind: key-existence and finite numbers only.

    Returns [] when acceptable, otherwise one human-readable reason string
    per violation. props must be a dict whose keys are all declared for the
    kind; numeric props present in props must be finite real numbers
    (bools and NaN/inf rejected). Generic on purpose: kind-LOCAL and
    cross-node semantic checks live on the NodeDefinitions.
    """
    if kind not in VALID_KINDS:
        return ["kind: unknown kind %r" % (kind,)]
    if not isinstance(props, dict):
        return ["props: expected an object for kind %r" % (kind,)]
    allowed = PROPS[kind]
    numeric = NUMERIC_PROPS[kind]
    errors = []
    for key in sorted(props):
        if key not in allowed:
            errors.append(
                "props: unknown prop %r for kind %r" % (key, kind)
            )
        elif key in numeric and not _finite_number(props[key]):
            errors.append(
                "props: %r of kind %r must be a finite number, got %r"
                % (key, kind, props[key])
            )
    return errors
