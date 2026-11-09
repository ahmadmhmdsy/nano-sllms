"""Config generator for the flow/ MVP (Task 5; registry promotion T2).

Bridges a validated .flow.json graph to the repo's EXISTING config
format (the webui train-tab contract, same wrapper via run_custom.py):
a linear dataset -> prepare -> tokenize -> train chain produces a
configs/flow_<slug>.yaml whose key structure is copied verbatim from
webui/app.py build_config / configs/smoke.yaml (never invented keys).

Public API:

- build_config(g) -> dict
    Pure function: enforces the linear-chain mapping, validates the
    document with flow.server.graph_schema plus
    flow.server.nodes.validate_props per node, and returns the nested
    config dict. Raises ValueError with human-readable reasons on any
    violation (nothing is ever written).

- generate(g, out_dir=None) -> str
    Calls build_config, then writes YAML (deterministic byte structure for
    future diffing) to <out_dir>/flow_<slug>.yaml and fsyncs. out_dir
    defaults to the repo's configs/; tests inject tempfile dirs and never
    write there. Returns the full written path.

De-hardcoding (registry promotion): the participant chain order and
every "requires upstream" message derive from the nodes REGISTRY
(each config participant declares required_upstream); the per-kind
knobs, presets and hard caps live in the owning builtin node modules.
This module keeps re-exporting the constants the Task-5 parity tests
read, and documents the config-order contract below.
"""

import os
from pathlib import Path

import yaml

from flow.server import graph_schema, nodes
from flow.server.flows import _SLUG_RE

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"

# Compat re-exports: these constants are OWNED by the node definitions
# now (flow/server/nodes/builtin/{prepare,tokenize,train}.py); the
# Task-5 webui-parity tests read them from here as before.
from flow.server.nodes.builtin.prepare import MAX_ROWS  # noqa: E402
from flow.server.nodes.builtin.tokenize import (  # noqa: E402
    SHARD_TOKENS, TOKENIZER_NAME)
from flow.server.nodes.builtin.train import (  # noqa: E402
    LR_PRESETS, PRESETS)

# Same guard as webui/app.py build_config: a run named like a shipped config
# would collide with it and with the auto-resume state (runs/<name>,
# data/<name>).
RESERVED_RUN_NAMES = frozenset(
    {"smoke", "pilot", "target", "sft_t1", "custom_example"})


def _participant_chain_order():
    """Config-participant kinds in topological dependency order.

    DERIVED from the registry: config_participant kinds, ordered so every
    kind's required_upstream predecessors come first (dataset depends on
    nothing; train is the terminal participant nothing depends ON).
    Deterministic Kahn walk in canonical REGISTRY order so ties never
    depend on dict internals. A dependency cycle is a registry bug and
    refuses the module import itself. Edges to non-participants are
    ignored for the ordering (the structural walk still enforces them).
    """
    registry = nodes.REGISTRY
    participants = [kind for kind in registry.kinds()
                    if registry.get(kind).config_participant]
    participant_set = set(participants)
    parents = {}
    for kind in participants:
        declared = set()
        for up_kind, _port in registry.get(kind).required_upstream.values():
            if up_kind in participant_set:
                declared.add(up_kind)
        parents[kind] = declared
    order = []
    placed = set()
    remaining = list(participants)  # canonical order for stable ties
    while remaining:
        ready = [kind for kind in remaining if parents[kind] <= placed]
        if not ready:
            raise ValueError(
                "registry bug: cycle in config_participant required_"
                "upstream edges among %s" % (remaining,))
        for kind in ready:
            order.append(kind)
            placed.add(kind)
            remaining.remove(kind)
    return tuple(order)


_PARTICIPANT_ORDER = _participant_chain_order()
  # dataset->prepare->tokenize->train
_CHAIN_ARROW = "->".join(_PARTICIPANT_ORDER)

# The only execution chain the MVP maps to a config. Branching is rejected
# cleanly (never silently ignored) until the future F5 execution task.
# The arrow text is DERIVED from the registry chain order (never a second
# hardcoded list); the wording is golden-locked.
_LINEAR_MSG = (
    "flow/%s: linear chains only - the MVP maps exactly one linear chain "
    "(linear chain mapping: " + _CHAIN_ARROW + ") to a config; "
    "branching graphs run via advanced YAML only in this MVP (future F5 "
    "execution)"
)


def _slug(name) -> str:
    """Return the validated run-name slug (flows.py regex, not reexported
    logic); every rejection names 'slug' for a stable error substring."""
    if not isinstance(name, str):
        raise ValueError("flow name: meta.name must be a slug string, got %r"
                         % (name,))
    if "/" in name or chr(92) in name or ".." in name:
        raise ValueError(
            "flow name: %r is not a slug (contains a path separator or '..')" 
            "- path traversal is not allowed" % (name,))
    if not _SLUG_RE.match(name):
        raise ValueError(
            "flow name: %r is not a slug - lowercase [a-z0-9-]{1,64} required"
            % (name,))
    return name


class _LinearChain:
    """Structural view: branching/merging/foreign kinds, or linear path."""

    def __init__(self, g):
        self.node_list = [n for n in g["graph"].get("nodes", [])
                          if isinstance(n, dict)]
        self.by_id = {n.get("id"): n for n in self.node_list
                      if isinstance(n.get("id"), str) and n["id"]}
        self.in_edges = {}   # node id -> list of (edge, from_id)
        self.out_edges = {}  # node id -> list of (edge, to_id)
        for edge in g["graph"].get("edges", []) or []:
            if not isinstance(edge, dict):
                continue  # graph_schema already reported the shape error
            frm, to = edge.get("from"), edge.get("to")
            if frm in self.by_id and to in self.by_id:
                self.out_edges.setdefault(frm, []).append((edge, to))
                self.in_edges.setdefault(to, []).append((edge, frm))

    def linear(self) -> bool:
        """True when nothing structurally branches. Missing chain kinds are
        NOT a linearity failure here: the dependency walk reports exactly
        which required upstream node (and port) is absent."""
        if not self.node_list:
            return False  # no chain at all is not a linear chain
        kinds = [n.get("kind") for n in self.node_list]
        if any(k not in _PARTICIPANT_ORDER for k in kinds if k is not None):
            return False  # node kinds outside the train chain
        present = [k for k in kinds if k is not None]
        if len(set(present)) != len(present):
            return False  # duplicated chain kind is not a linear mapping
        for _nid, edges in self.out_edges.items():
            if len(edges) > 1:
                return False  # branching
        for _nid, edges in self.in_edges.items():
            if len(edges) > 1:
                return False  # merge (diamond)
        return True


def _build_chain(g):
    """Return {kind: node} for the registry-derived participant chain.

    Raises the linear-only ValueError for branching graphs and the
    missing-input ValueError naming each absent required upstream node.
    The walk starts at the terminal participant (nothing depends on it:
    train) and follows each child's required_upstream declaration from
    the nodes REGISTRY - no hardcoded CHAIN list, no _DEPENDENCIES.
    """
    view = _LinearChain(g)
    if not view.linear():
        raise ValueError(_LINEAR_MSG % (g.get("meta") or {}).get("name"),)
    terminal = _PARTICIPANT_ORDER[-1]
    terminal_node = next((n for n in view.node_list
                          if n.get("kind") == terminal), None)
    if terminal_node is None:
        raise ValueError(
            "%s: no %s node found - the MVP maps exactly one linear chain "
            "(linear chain mapping: %s) to a config"
            % (terminal, terminal, _CHAIN_ARROW))
    chain = [terminal_node]
    # Walk backwards: every link must satisfy the child's declared
    # upstream requirements. MVP participants declare exactly one link
    # each, so the historic per-step messages (kind/port quotes preserved)
    # are reproduced verbatim.
    while len(chain) < len(_PARTICIPANT_ORDER):
        child = chain[-1]
        child_defn = nodes.REGISTRY.get(child.get("kind"))
        if not child_defn.required_upstream:
            break  # chain head reached (its own declaration says so)
        dep_kind, port = next(iter(child_defn.required_upstream.values()))
        ins = view.in_edges.get(child.get("id"), [])
        if not ins:
            raise ValueError(
                "%s requires upstream %s node providing %s; none found "
                "(node '%s')" % (child.get('kind'), dep_kind, port,
                                 child.get("id")))
        parent_id = ins[0][1]
        parent = view.by_id.get(parent_id, {})
        if parent.get("kind") != dep_kind:
            raise ValueError(
                "%s requires upstream %s node providing %s; node '%s' has "
                "kind %r instead (node '%s')"
                % (child.get("kind"), dep_kind, port, parent_id,
                   parent.get("kind"), child.get("id")))
        chain.append(parent)
    chain.reverse()  # dataset, prepare, tokenize, train
    return {node.get("kind"): node for node in chain}


class _NodeContext:
    """Read-only context handed to NodeDefinition.validate_semantic and
    build_section (duck-typed in flow/server/nodes/base.py; defined here
    because it is config-assembly state, not registry data).

    - phase: the config-order contract pass, "early" or "full"; only the
      terminal participant (train) distinguishes them today
    - chain: {kind: node} for the registry-derived linear chain
    - slug: the validated run-name slug
    """

    def __init__(self, phase, chain, slug):
        self.phase = phase
        self.chain = chain
        self.slug = slug


def _semantic_passes(chain, slug):
    """(kind, node, ctx) semantic passes in the config-order contract order.

    Config-order contract (pinned by the goldens + the config_gen tests):
    semantic checks raise in the EXACT sequence the pre-registry
    build_config used -

      1. the terminal participant's early pass (train steps-knob presence)
      2. every other participant's full pass in derived chain order
         (dataset label shape -> prepare knobs -> tokenize knobs; new
         participant kinds slot in here by registry order)
      3. the terminal participant's full pass (preset / lr_preset / ctx
         choices; the ctx check is cross-node via ctx.chain)

    Raising on the FIRST reason of the FIRST failing pass reproduces the
    historic first-message ordering byte for byte.
    """
    terminal = _PARTICIPANT_ORDER[-1]
    plan = ([(terminal, "early")]
            + [(kind, "full") for kind in _PARTICIPANT_ORDER[:-1]]
            + [(terminal, "full")])
    return [(kind, chain[kind], _NodeContext(phase, chain, slug))
            for kind, phase in plan]


def _merge_sections(cfg, chain, slug):
    """Merge each participant's build_section into cfg, in chain order.

    Section-merge contract: a top-level key may be emitted by several
    sections ONLY as a dict on both sides - today the "data" block is
    assembled from the dataset, prepare and tokenize sections - and
    such shared dicts merge one level deep. Any OTHER duplicate
    top-level key is a registry bug and raises; duplicate LEAF keys
    inside a shared dict are impossible in the MVP but are asserted
    anyway. Sections merge in the derived chain order, so the merged
    config grows data -> tokenizer -> model -> train -> eval.
    """
    ctx = _NodeContext("full", chain, slug)
    for kind in _PARTICIPANT_ORDER:
        section = nodes.REGISTRY.get(kind).build_section(ctx)
        for key, value in section.items():
            existing = cfg.get(key)
            if existing is None:
                cfg[key] = value
            elif isinstance(existing, dict) and isinstance(value, dict):
                for leaf_key, leaf in value.items():
                    # "data" leaf collision impossible in MVP; assert
                    # anyway (= the documented section contract).
                    assert leaf_key not in existing, (
                        "registry bug: %r section re-emits %s.%s"
                        % (kind, key, leaf_key))
                    existing[leaf_key] = leaf
            else:
                raise ValueError(
                    "registry bug: %r section re-emits top-level key %r"
                    % (kind, key))


def build_config(g) -> dict:
    """Validate graph + props and return the nested config dict (pure).

    Raises ValueError (joined graph-schema reasons, registry prop reasons,
    linear-chain structural errors, or missing knob errors).
    """
    malformed = not (isinstance(g, dict)
                     and isinstance(g.get("graph"), dict)
                     and isinstance(g["graph"].get("nodes"), list))
    if malformed:
        # Degenerate document: the schema reasons are the honest answer.
        errors = list(graph_schema.validate(g))
        raise ValueError("\n".join(errors))

    # 1. Linear-chain structural mapping (BEFORE schema error emission so a
    #    branching graph gets the linear-only message, not a port quote).
    _build_chain(g)

    # 2. Document schema (graph_schema.validate) + per-node registry props -
    #    the same dual-validator boundary flows.py uses at save.
    errors = list(graph_schema.validate(g))
    for node in g["graph"]["nodes"]:
        if not isinstance(node, dict):
            continue  # graph_schema already reported the shape error
        for reason in nodes.validate_props(node.get("kind"),
                                           node.get("props")):
            errors.append("graph.nodes[%s]: %s" % (node.get("id", "?"),
                                                   reason))
    if errors:
        raise ValueError("\n".join(errors))

    # 3. Run-name slug - flows.py slug rule reused, not reimplemented; the
    #    webui reserved-name guard rejects shipped-config collisions. The
    #    raw-name check runs first: webui's own slug rule accepts '_' (so
    #    'custom_example' is a legit webui name), while the flow slug rule
    #    is stricter - reservation must win over the slug message.
    raw_name = (g.get("meta") or {}).get("name")
    if (isinstance(raw_name, str)
            and raw_name.strip().lower() in RESERVED_RUN_NAMES):
        slug = raw_name.strip().lower()
    else:
        slug = _slug(raw_name)
    if slug in RESERVED_RUN_NAMES:
        raise ValueError(
            "flow name: '%s' is a reserved shipped-config name - it would "
            "collide with the shipped configs and auto-resume state "
            "(runs/%s, data/%s)" % (slug, slug, slug))

    # 4. Per-node semantic validation ( knob + label checks on the node
    #    definitions ) walked in the config-order contract sequence and
    #    raising on the FIRST reason - one human-readable message, as
    #    before. Then the nested config: per-node build_section outputs
    #    merged in the same canonical order (_merge_sections); key
    #    structure is still copied verbatim from webui build_config minus
    #    the interactive-only data_mode knob (matches shipped
    #    configs/smoke.yaml key structure exactly).
    chain = _build_chain(g)
    for kind, node, ctx in _semantic_passes(chain, slug):
        defn = nodes.REGISTRY.get(kind)
        reasons = defn.validate_semantic(node, ctx)
        if reasons:
            raise ValueError(reasons[0])
    cfg = {"name": slug}
    _merge_sections(cfg, chain, slug)
    return cfg


def generate(g, out_dir=None) -> str:
    """Validate, then write configs/flow_<slug>.yaml and fsync; return path.

    Nothing is touched on disk until build_config accepts the graph.
    out_dir defaults to the repo's configs/; tests pass tempfile dirs ONLY.
    """
    cfg = build_config(g)  # every validation happens here, no I/O yet
    slug = cfg["name"]
    target = Path(out_dir) if out_dir is not None else DEFAULT_CONFIGS_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / ("flow_" + slug + ".yaml")
    payload = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())  # data-preservation rule: durable before OK
    return str(path)
