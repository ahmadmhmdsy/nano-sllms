"""U-line 3-layer validator.

L1 = chain structural parse (chain_to_spec issues)
L2 = schema/catalog: type exists, props known + required, value kinds, events, actions
L3 = semantic: root exists, children resolve + acyclic, binding/pointer paths
     resolve against state, $item scope rules, repeat/openPath path checks
"""

from __future__ import annotations

from ui.src.chain import chain_to_spec, Issue
from ui.src.catalog import COMPONENTS, ACTIONS, FREE_ACTIONS

_SCALARS = {"str": str, "num": (int, float), "bool": bool}


def validate_spec(spec):
    issues = []
    els = spec.get("elements") or {}
    root = spec.get("root")
    state = spec.get("state") or {}

    # parent map + child resolution
    parent_of = {}
    for key, el in els.items():
        if el.get("type") not in COMPONENTS:
            issues.append(["L2", "unknown_type", key, str(el.get("type"))])
        for ch in el.get("children", []):
            parent_of[ch] = key
            if ch not in els:
                issues.append(["L3", "missing_child", key, "child key {0} undefined".format(ch)])

    if root not in els:
        issues.append(["L3", "missing_root", root, ""])
    else:
        # acyclicity + reachability: every element must trace to root without a loop
        for key in els:
            seen = set()
            cur = key
            while cur is not None:
                if cur in seen:
                    issues.append(["L3", "cycle", cur, ""])
                    break
                seen.add(cur)
                cur = parent_of.get(cur)
                if cur is None and cur not in parent_of and key != root:
                    pass  # missing parent already reported above
        # reachability from root
        reach = set()
        stack = [root]
        while stack:
            k = stack.pop()
            if k in reach:
                continue
            reach.add(k)
            # a child id that is not in els is already reported as L3 missing_child
            # above; skip it here instead of raising KeyError on untrusted model output
            node = els.get(k)
            if isinstance(node, dict):
                stack.extend(node.get("children", []))
        for key in els:
            if key not in reach:
                issues.append(["L3", "unreached_element", key, "not reachable from root"])

    def pointer_ok(p):
        if not isinstance(p, str) or not p:
            return False
        node = state
        for tok in p.lstrip("/").split("/"):
            tok = tok.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict):
                if tok not in node:
                    return False
                node = node[tok]
            elif isinstance(node, list):
                try:
                    node = node[int(tok)]
                except (ValueError, IndexError):
                    return False
            else:
                return False
        return True

    for key, el in els.items():
        ctype = el.get("type")
        comp = COMPONENTS.get(ctype)
        if comp is None:
            continue  # already reported at L2
        props = el.get("props") or {}
        for pk, pv in props.items():
            if pk not in comp.props:
                issues.append(["L2", "unknown_prop", key, pk])
                continue
            spec2 = comp.props[pk]
            if isinstance(spec2, tuple):
                kind, opts = spec2[0], (spec2[1] if len(spec2) > 1 else ())
                required = True if len(spec2) > 2 else False
            else:
                kind, opts, required = spec2, (), False
            if isinstance(pv, dict):
                k0 = list(pv)[0]
                if not k0.startswith("$"):
                    issues.append(["L2", "bad_prop_object", key, pk])
                    continue
                kexpr = k0.lstrip("$")
                if kexpr not in ("state", "bindState", "item", "bindItem", "index", "cond", "then", "else", "template", "computed", "format", "math", "concat", "count", "truncate", "pluralize", "join"):
                    issues.append(["L2", "unknown_expr_kind", key, k0])
                elif kexpr in ("state", "bindState") and isinstance(pv[k0], str):
                    arg = pv[k0]
                    if arg and not pointer_ok(arg):
                        issues.append(["L3", "binding_path_missing", key, "{0} -> {1}".format(pk, arg)])
            elif kind == "enum":
                if pv is not None and pv not in opts:
                    issues.append(["L2", "bad_enum_value", key, "{0}={1}".format(pk, pv)])
            elif kind == "str":
                if pv is not None and not isinstance(pv, str):
                    issues.append(["L2", "bad_prop_kind", key, "{0} not a string".format(pk)])
            elif kind == "num":
                if pv is not None and not isinstance(pv, (int, float)):
                    issues.append(["L2", "bad_prop_kind", key, "{0} not numeric".format(pk)])
            elif kind == "bool":
                if pv is not None and not isinstance(pv, bool):
                    issues.append(["L2", "bad_prop_kind", key, "{0} not boolean".format(pk)])
            elif kind == "as":
                if not isinstance(pv, list) or any(not isinstance(x, str) for x in pv):
                    issues.append(["L2", "bad_prop_kind", key, "{0} not string-array".format(pk)])
            elif kind == "ao":
                if not isinstance(pv, list) or any(not isinstance(x, dict) for x in pv):
                    issues.append(["L2", "bad_prop_kind", key, "{0} not object-array".format(pk)])
            elif kind == "ptr":
                if pv is not None and not (isinstance(pv, str) and pv.startswith("/")):
                    issues.append(["L2", "bad_pointer", key, "{0}={1}".format(pk, pv)])
        if "openPath" in props and isinstance(props["openPath"], str) and props["openPath"]:
            if not pointer_ok(props["openPath"]):
                issues.append(["L3", "openPath_missing", key, props["openPath"]])
        for ev in el.get("on", {}):
            if ev not in comp.events:
                issues.append(["L2", "unknown_event_for_type", key, ev])
        if "repeat" in el:
            if not el.get("children"):
                issues.append(["L3", "repeat_without_children", key, ""])
            rp = el["repeat"]
            for ok in rp:
                if ok not in ("statePath", "key"):
                    issues.append(["L3", "bad_repeat_field", key, ok])
            sp = rp.get("statePath", "")
            if sp and not pointer_ok(sp):
                issues.append(["L3", "repeat_state_mismatch", key, sp])
        if "visible" in el:
            vis = el["visible"]
            k0 = list(vis)[0] if vis else ""
            if not k0.startswith("$") or k0.lstrip("$") not in ("state", "item", "index"):
                issues.append(["L3", "bad_visible_kind", key, k0])
            for ok, ov in vis.items():
                if ok.startswith("$") and ok.lstrip("$") in ("state",):
                    if isinstance(ov, str) and ov and not pointer_ok(ov):
                        issues.append(["L3", "visible_path_missing", key, ov])
                elif not ok.startswith("$"):
                    if ok not in ("eq", "neq", "gt", "gte", "lt", "lte", "not"):
                        issues.append(["L3", "bad_visible_op", key, ok])
        for ev, act in el.get("on", {}).items():
            binds = act if isinstance(act, list) else [act]
            for b in binds:
                if not isinstance(b, dict) or "action" not in b:
                    issues.append(["L2", "bad_action_binding", key, ev])
                    continue
                if b["action"] not in ACTIONS + FREE_ACTIONS:
                    issues.append(["L2", "unknown_action", key, b["action"]])
                for pk2, pv2 in (b.get("params") or {}).items():
                    if isinstance(pv2, dict) and list(pv2) and list(pv2)[0] == "$state":
                        p2 = pv2["$state"]
                        if isinstance(p2, str) and p2 and not pointer_ok(p2):
                            issues.append(["L3", "action_param_path_missing", key, p2])
        # $item / $bindItem legal only under a repeat ancestor
        def has_repeat_ancestor(k):
            cur = k
            while cur is not None:
                el2 = els.get(cur)
                if el2 is None:
                    return False
                if "repeat" in el2 or (el2.get("props") or {}).get("repeat") is not None:
                    return True
                cur = parent_of.get(cur)
            return False

        for pk, pv in props.items():
            if isinstance(pv, dict) and list(pv) and list(pv)[0].lstrip("$") in ("item", "bindItem", "index"):
                if not has_repeat_ancestor(key):
                    issues.append(["L3", "repeat_item_outside_scope", key, pk])
    return issues


def _norm(issues):
    out = []
    for i in issues:
        if isinstance(i, list):
            out.append(Issue(i[0], i[1], i[2] if len(i) > 2 else None, i[3] if len(i) > 3 else ""))
        elif i is not None:
            out.append(i)
    return out


def validate_chain(chain):
    spec, issues = chain_to_spec(chain)
    return _norm(issues) + _norm(validate_spec(spec))
