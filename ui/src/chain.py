"""U-line v1 chain <-> json-render flat-spec codec.

Chain grammar (fixed ops, one op per line; structured payloads are compact JSON
so the codec is escape-free):

    UI1 <catalog-id>
    TASK <free rest-of-line>
    ROOT <key>
    EL   <key> <Type> <parentKey-or-dash>
    PROP <key> <prop> <json value>       (literals only; bindings use BIND)
    BIND <key> <prop> <kind> <pointer-or-field>   kind: state|bindState|item|bindItem|index|template
    COND <key> visible <json cond>
    REPT <key> <json repeat> <json children-refs array>
    EVNT <key> <event> <json action-binding>
    STAX <json state object>

Canonical order: DFS pre-order (parent before child), children in spec order.
v1 scope exclusions (documented, deliberately): $cond/$computed recursion chains,
multi-slot trees, and element-level "watch" are NOT chain-representable; the
generator never emits them and chain_to_spec ignores unknown element fields.
A later WACH op is a pre-registered extension (see research report section 7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from ui.src.catalog import CATALOG_FROZEN


@dataclass
class Issue:
    layer: str   # L1 | L2 | L3
    code: str
    element: object = None
    detail: str = ""

    def __str__(self):
        return "[{0}] {1} @ {2}: {3}".format(self.layer, self.code, self.element, self.detail)


def _dumps(v):
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def spec_to_chain(spec, task_text=""):
    out = ["UI1 " + CATALOG_FROZEN, "TASK " + task_text]
    if spec.get("root") is not None:
        out.append("ROOT " + str(spec["root"]))
    if spec.get("state") is not None:
        out.append("STAX " + _dumps(spec["state"]))

    def emit(key, parent):
        el = spec["elements"][key]
        out.append("EL {0} {1} {2}".format(key, el["type"], parent if parent is not None else "-"))
        for pk, pv in (el.get("props") or {}).items():
            if isinstance(pv, dict):
                k0 = list(pv)[0]
                kind = k0.lstrip("$")
                arg = pv[k0]
                if isinstance(arg, (dict, list)):
                    arg = _dumps(arg)
                out.append("BIND {0} {1} {2} {3}".format(key, pk, kind, arg))
            else:
                out.append("PROP {0} {1} {2}".format(key, pk, _dumps(pv)))
        if "visible" in el:
            out.append("COND {0} visible {1}".format(key, _dumps(el["visible"])))
        if "repeat" in el:
            out.append("REPT {0} {1} {2}".format(key, _dumps(el["repeat"]), _dumps(el["children"])))
        if "on" in el:
            for ev, act in el["on"].items():
                binds = act if isinstance(act, list) else [act]
                for b in binds:
                    out.append("EVNT {0} {1} {2}".format(key, ev, _dumps(b)))
        for ch in el["children"]:
            emit(ch, key)

    root = spec["root"]
    if root not in spec["elements"]:
        raise ValueError("root {0} not in elements".format(root))
    emit(root, None)
    return "\n".join(out)


def chain_to_spec(chain):
    """Returns (spec, issues). Inverse of spec_to_chain on the representable subset."""
    issues = []
    root = None
    els = {}          # key -> record: type, props, children(from REPT), visible, on, parent
    parents = {}
    state = None
    has_state = False
    seen_keys = set()
    rept_child = set()

    for raw in chain.splitlines():
        ln = raw.rstrip()
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        T = ln.split()
        op = T[0]

        if op == "TASK":
            continue
        elif op == "ROOT":
            root = T[1] if len(T) == 2 else None
            if len(T) != 2:
                issues.append(Issue("L1", "bad_root_line", detail=ln))
        elif op == "STAX":
            try:
                state = json.loads(ln[5:])
                has_state = True
            except json.JSONDecodeError as e:
                issues.append(Issue("L1", "bad_state_json", detail=str(e)))
        elif op == "EL":
            if len(T) != 4:
                issues.append(Issue("L1", "bad_el_line", detail=ln))
                continue
            key, ctype, parent = T[1], T[2], T[3]
            if key in seen_keys:
                issues.append(Issue("L1", "duplicate_key", key))
            seen_keys.add(key)
            els[key] = {"type": ctype, "props": {}, "children": []}
            els[key]["_p"] = parent
        elif op in ("PROP", "BIND"):
            key = T[1] if len(T) > 1 else None
            if key is None or key not in els:
                issues.append(Issue("L1", op.lower() + "_unknown_key", key, detail=ln))
                continue
            if op == "PROP":
                parts = ln.split(None, 3)
                if len(parts) != 4:
                    issues.append(Issue("L1", "bad_prop_line", key, detail=ln))
                    continue
                try:
                    els[key]["props"][parts[2]] = json.loads(parts[3])
                except json.JSONDecodeError as e:
                    issues.append(Issue("L1", "bad_prop_json", key, detail=str(e)))
            else:
                if len(T) != 5:
                    issues.append(Issue("L1", "bad_bind_line", key, detail=ln))
                    continue
                els[key]["props"][T[2]] = {"$" + T[3].lstrip("$"): T[4]}
        elif op == "COND":
            parts = ln.split(None, 3)
            if len(parts) != 4 or parts[2] != "visible":
                issues.append(Issue("L1", "bad_cond_line", detail=ln))
                continue
            key = parts[1]
            if key not in els:
                issues.append(Issue("L1", "cond_unknown_key", key, detail=ln))
                continue
            try:
                els[key]["visible"] = json.loads(parts[3])
            except json.JSONDecodeError as e:
                issues.append(Issue("L1", "bad_cond_json", key, detail=str(e)))
        elif op == "REPT":
            parts = ln.split(None, 3)
            if len(parts) != 4:
                issues.append(Issue("L1", "bad_rept_line", detail=ln))
                continue
            key = parts[1]
            if key not in els:
                issues.append(Issue("L1", "rept_unknown_key", key, detail=ln))
                continue
            try:
                els[key]["repeat"] = json.loads(parts[2])
                els[key]["children"] = list(json.loads(parts[3]))
                rept_child.update(els[key]["children"])
            except json.JSONDecodeError as e:
                issues.append(Issue("L1", "bad_rept_json", key, detail=str(e)))
        elif op == "EVNT":
            parts = ln.split(None, 3)
            if len(parts) != 4:
                issues.append(Issue("L1", "bad_evnt_line", detail=ln))
                continue
            key, ev = parts[1], parts[2]
            if key not in els:
                issues.append(Issue("L1", "evnt_unknown_key", key, detail=ln))
                continue
            try:
                els[key].setdefault("on", {})[ev] = json.loads(parts[3])
            except json.JSONDecodeError as e:
                issues.append(Issue("L1", "bad_evnt_json", key, detail=str(e)))
        elif op == "UI1":
            pass
        else:
            issues.append(Issue("L1", "bad_op", detail=ln))

    # parent-driven fill: attach every element to its parent unless it is already
    # addressed as a repeat-child (REPT arrays own those references)
    visited = set()
    for key in els:
        el = els[key]
        p = el.pop("_p")
        if p is None:
            el.pop("_p", None)
        if p == "-" or p not in els:
            if p != "-":
                issues.append(Issue("L3", "missing_child", key, "parent {0} undefined".format(p)))
            continue
        if key in rept_child:
            continue  # reference already owned by the repeat container
        if key in visited:
            issues.append(Issue("L3", "duplicate_ref", key, "parent {0}".format(p)))
            continue
        visited.add(key)
        els[p]["children"].append(key)

    spec = {"root": root, "elements": els}
    if has_state:
        spec["state"] = state
    return spec, issues
