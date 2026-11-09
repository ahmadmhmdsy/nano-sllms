"""Abstract node definition + spec records for the flow/ node registry.

A NodeDefinition is the behavior-carrying unit that replaces the old
dict-entry registry (the pre-promotion flow/server/nodes.py): each
built-in kind declares its ports/props/gate as data and may override
validate_semantic (kind-local checks on top of the generic
validate_props) and build_section (its slice of the generated
train-tab config).

Pure declarations + pure methods - no I/O.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PortSpec:
    """One declarative port; to_dict matches the legacy port dict shape.

    direction is "in" or "out"; burst_format is an optional noun for how
    the port's data streams (e.g. "row-batch", "file-list").
    """

    name: str
    direction: str
    type: str
    burst_format: "str | None" = None

    def to_dict(self):
        return {
            "name": self.name,
            "direction": self.direction,
            "type": self.type,
            "burst-format": self.burst_format,
        }


@dataclass(frozen=True)
class PropSpec:
    """One editable prop declaration.

    type is "number", "string" or "select" (the numeric_props compat
    surface derives from type == "number"); widget_hint/help are display
    metadata for registry snapshot consumers.
    """

    name: str
    type: str
    required: bool = False
    widget_hint: "str | None" = None
    help: "str | None" = None

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "widget_hint": self.widget_hint,
            "help": self.help,
        }


class NodeDefinition:
    """Abstract definition of one node kind (builtin modules subclass).

    Declarative class fields (every builtin overrides what it needs):

    - kind: the .flow.json kind string; the registry key (required)
    - title / category: display metadata for registry snapshot consumers
    - config_participant: True when the kind takes part in the linear
      config-mapping chain (dataset/prepare/tokenize/train today)
    - gate: execution gate string (runner queue key); None means ungated
    - label_semantic: what the node's user label carries when that is
      not a plain display name (dataset: "hf-dataset-name" = the HF
      dataset name drives the config mapping); None otherwise
    - features: registry-declared UI feature flags (e.g. infer's
      "webui-chat-button"); UI chrome keys off these, never kind strings
    - ports / props: PortSpec / PropSpec lists (order-independent: the
      derived compat surface is sorted)
    - required_upstream: {in-port name: (upstream kind, upstream
      out-port name)} - config participants declare the chain through
      this; config_gen derives the chain order from it.

    Definitions are registered as stateless singletons; class-level
    fields must be treated as read-only.
    """

    kind = ""
    title = ""
    category = ""
    config_participant = False
    gate = None
    label_semantic = None
    features = ()
    ports: "list[PortSpec]" = ()
    props: "list[PropSpec]" = ()
    # Immutable shared default (review minor 4: a mutable class-level dict
    # is a footgun for future definitions); participants override with
    # their own dict AFTER class creation (or via a plain dict assignment
    # that then must never be mutated in place).
    required_upstream: "dict[str, tuple[str, str]]" = MappingProxyType({})

    def validate_semantic(self, node, ctx):
        """Kind-local semantic checks beyond validate_props (base: none).

        `node` is the .flow.json node dict; `ctx` is the config_gen
        context (.phase "early"/"full" per the config-order contract,
        .chain = {kind: node} for the derived chain, .slug = run name).
        Returns [] when acceptable, otherwise one human-readable reason
        string per violation IN REPORT ORDER - the caller raises the
        first reason, so reason order encodes the historic message
        ordering (config-order contract).
        """
        return []

    def build_section(self, ctx):
        """This node's slice of the generated config (base: abstract).

        Returns {top-level key: dict}. The caller merges sections in the
        derived chain order (config_gen._merge_sections); only config
        participants are ever asked.
        """
        raise NotImplementedError(
            "build_section: kind %r does not build a config section"
            % (self.kind,))

    def _missing_knob_errors(self, node):
        """Container + per-knob presence reasons (error strings locked).

        Shared by prepare/tokenize: the historic build_config refused a
        props object that is missing or not a dict, then reported missing
        knobs one sorted key at a time. Returns [] when the node declares
        every knob on a real object; the caller raises the first reason.
        """
        props = node.get("props")
        if not isinstance(props, dict):
            return [
                "props: missing - %s node '%s' has no props object and the "
                "config mapping requires every %s knob"
                % (self.kind, node.get("id"), self.kind)]
        errors = []
        for spec in sorted(self.props, key=lambda spec: spec.name):
            if spec.name not in props:
                errors.append(
                    "%s: missing - %s node '%s' requires the %s knob"
                    % (spec.name, self.kind, node.get("id"), spec.name))
        return errors
