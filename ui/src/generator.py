"""U-line v1 synthetic spec generator: 8 archetypes, seeded, valid by construction.

Constraints (Constrained-GT, research report section 6):
- tree depth <= 4, elements <= 16, siblings <= 6
- v1 expression subset only: {"$state"}, {"$bindState"}, {"$item"} (no $cond/$computed/watch)
- optional props: kept, dropped, or set to None (never invented)
- build() asserts L1-L3 validity before returning (validator mirror of renderer contract).
"""

from __future__ import annotations

import random

from ui.src.validate import validate_spec

_NAMES = ["Jane Cooper", "Devon Lane", "Marcus Reed", "Priya Anand", "Cody Fisher", "Lena Mora"]
_SKILLS = ["TypeScript", "React", "Python", "Node.js", "Design", "SQL"]
_TODO_TITLE = ["Buy milk", "Walk the dog", "Send report", "Fix the login bug", "Review PR"]
_TITLES = ["Overview", "Brokerage", "Team pulse"]
_QUARTERS = [("Q1", "$10k"), ("Q2", "$14k"), ("Q3", "$18k"), ("Q4", "$21k")]
_STATS = (["Paid", 0.5], ["Pending", 0.3], ["Refunded", 0.2])

# ---- R2 diversity escalation (task ui/task-u3-diversity, 2026-09-27) ----
# The R2 dataset audit (research/ui_dataset_adequacy_r2.md) pinned the weak
# rung to surface values: 19 titles / constant STAX numbers over 40k rows.
# Every pool below is sampled with the per-row generator, so a new corpus
# seed instantiate tens of thousands of distinct value sets with NO change
# to the chain contract or the validator surface.
_MONEY = [9, 12, 15, 18, 21, 24, 28, 33, 37, 41, 45, 52, 58, 64, 71]
_PERC = ["+{0}.{1}%".format(a, b) for a in range(1, 30) for b in range(0, 9, 3)]
_CHANGE = ["-{0}.{1}%".format(a, b) for a in range(1, 12) for b in range(0, 9, 3)]
_NULLISH = ["0", "-1.2%", "+3.1%"]
_METRIC_LABELS = [["Revenue", "Active users"], ["Sales", "Downloads"],
                  ["MRR", "Trials"], ["Orders", "Visitors"], ["Leads", "Signups"]]
BAR_LABELS = [["Q1", "Q2", "Q3", "Q4"], ["Mon", "Tue", "Wed", "Thu"],
              ["Jan", "Feb", "Mar", "Apr"]]
_CHART_TITLES = ["Sales by quarter", "Weekly trend", "Monthly revenue",
                 "Quarterly outlook", "Monthly sales", "Weekly volume"]
_SECTION_HEADS = ["Overview", "Workspace", "Command center", "Summary view",
                  "Home base", "Pulse", "Snapshot"]


def _money(gen):
    return "$" + str(gen.choice(_MONEY) * 1000 + gen.choice([0, 118, 204, 318, 462, 551, 748]))


def _state_panel(gen, labels):
    a, b = labels
    return {
        a.replace(" ", "").lower(): _money(gen),
        a.replace(" ", "").lower() + "Change": gen.choice(_PERC),
        b.replace(" ", "").lower(): _money(gen),
        b.replace(" ", "").lower() + "Change": gen.choice(_PERC + _NULLISH[:1]),
    }


def _wpick(gen, pairs):
    names = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return gen.choices(names, weights=weights)[0]


def _k(gen):
    return "".join(gen.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(4))


def _nz(gen, props):
    """Optional-prop discipline for Prop dicts: value kept, dropped, or None."""
    out = {}
    for k, v in props.items():
        if v is not None:
            out[k] = v
        elif gen.random() < 0.5:
            out[k] = None
    return out


def _dashboard(gen):
    root, grid = "card" + _k(gen), "grid" + _k(gen)
    m1, m2 = "m1" + _k(gen), "m2" + _k(gen)
    ch = "ch" + _k(gen)
    with_chart = gen.random() < 0.7
    kids = [m1, m2, ch] if with_chart else [m1, m2]
    # R2 diversity: metric labels, money values, and quarterly series are
    # all sampled with the row generator now.
    labels = gen.choice(_METRIC_LABELS)
    quarters = gen.choice(BAR_LABELS)
    state = {
        labels[0].replace(" ", "").lower(): _money(gen),
        labels[0].replace(" ", "").lower() + "Change": gen.choice(_PERC),
        labels[1].replace(" ", "").lower(): _money(gen),
        labels[1].replace(" ", "").lower() + "Change": gen.choice(_PERC + _NULLISH[:1]),
        "sales": [{"label": a, "value": "$" + str(gen.choice(_MONEY)) + "k"} for a in quarters],
    }
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": gen.choice(_SECTION_HEADS), "maxWidth": "md", "description": None, "centered": None}), "children": [grid]},
        grid: {"type": "Grid", "props": {"columns": 2 if len(kids) == 2 else gen.choice([2, 2, 3]), "gap": "md" if gen.random() < 0.6 else "lg"}, "children": kids},
        m1: {"type": "Metric", "props": {"label": labels[0], "value": {"$state": "/" + labels[0].replace(" ", "").lower()}, "change": {"$state": "/" + labels[0].replace(" ", "").lower() + "Change"}, "changeType": "1" if gen.random() < 0.7 else "0"}, "children": []},
        m2: {"type": "Metric", "props": {"label": labels[1], "value": {"$state": "/" + labels[1].replace(" ", "").lower()}, "change": {"$state": "/" + labels[1].replace(" ", "").lower() + "Change"}, "changeType": "0"}, "children": []},
    }
    if with_chart:
        els[ch] = {"type": "BarGraph" if gen.random() < 0.5 else "LineGraph",
                   "props": {"title": gen.choice(_CHART_TITLES), "data": {"$state": "/sales"}}, "children": []}
    return {"root": root, "elements": els, "state": state}


def _settings(gen):
    root, form = "card" + _k(gen), "form" + _k(gen)
    ni, ei, ro = "ni" + _k(gen), "ei" + _k(gen), "ro" + _k(gen)
    sw1, sw2, save = "sw1" + _k(gen), "sw2" + _k(gen), "sv" + _k(gen)
    state = {"name": "Ada Lovelace", "email": "ada@demo.io", "role": "", "notifications": True, "darkMode": False}
    form_kids = [ni, ei, ro, sw1, sw2, save]
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": "Account settings", "description": "Manage your profile", "maxWidth": "md", "centered": None}), "children": [form]},
        form: {"type": "Stack", "props": {"direction": "vertical", "gap": "md", "align": None, "justify": None}, "children": form_kids},
        ni: {"type": "Input", "props": {"label": "Full name", "name": "name", "type": "text", "placeholder": "Your name", "value": {"$bindState": "/name"}, "checks": [{"type": "required", "message": "Name required"}]}, "children": []},
        ei: {"type": "Input", "props": {"label": "Email", "name": "email", "type": "email", "placeholder": "you@demo.io", "value": {"$bindState": "/email"}}, "children": []},
        ro: {"type": "Select", "props": {"label": "Role", "name": "role", "options": ["Engineer", "Designer", "Manager"], "placeholder": "Choose a role", "value": {"$bindState": "/role"}, "checks": [{"type": "required", "message": "Role required"}]}, "children": []},
        sw1: {"type": "Switch", "props": {"label": "Email notifications", "name": "notifications", "checked": {"$bindState": "/notifications"}}, "children": []},
        sw2: {"type": "Switch", "props": {"label": "Dark mode", "name": "darkMode", "checked": {"$bindState": "/darkMode"}}, "children": []},
        save: {"type": "Button", "props": {"label": "Save changes", "variant": gen.choice(["primary", "primary", "secondary"])}, "children": [],
                "on": {"press": {"action": "save_form", "params": {}}}},
    }
    if gen.random() < 0.4:
        els.pop(sw2)
        form_kids.remove(sw2)
    return {"root": root, "elements": els, "state": state}


def _todo(gen):
    root = "card" + _k(gen)
    new, add = "nw" + _k(gen), "ad" + _k(gen)
    lst, item = "ls" + _k(gen), "im" + _k(gen)
    n = gen.randint(2, 5)
    state = {"newTodoText": "", "todos": [{"id": "t{0}".format(i + 1), "title": gen.choice(_TODO_TITLE), "done": gen.random() < 0.3} for i in range(n)]}
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": "Todo list", "maxWidth": "md", "description": None, "centered": None}), "children": [new, add, lst]},
        new: {"type": "Input", "props": {"label": "New task", "name": "newTodo", "type": "text", "placeholder": "What needs doing?", "value": {"$bindState": "/newTodoText"}}, "children": []},
        add: {"type": "Button", "props": {"label": "Add", "variant": "primary"}, "children": [],
               "on": {"press": {"action": "pushState", "params": {"statePath": "/todos", "value": {"$state": "/newTodoText"}, "title": {"$state": "/newTodoText"}, "id": "$id"}}}},
        lst: {"type": "Stack", "props": {"direction": "vertical", "gap": "sm", "align": None, "justify": None},
               "repeat": {"statePath": "/todos", "key": "id"}, "children": [item]},
        item: {"type": "Card", "props": {"title": {"$item": "title"}, "maxWidth": "full", "description": None, "centered": None}, "children": []},
    }
    return {"root": root, "elements": els, "state": state}


def _profile(gen):
    name = gen.choice(_NAMES)
    root, stk = "card" + _k(gen), "stk" + _k(gen)
    head, txt = "hd" + _k(gen), "tx" + _k(gen)
    b1, b2, prg = "b1" + _k(gen), "b2" + _k(gen), "pg" + _k(gen)
    kids = [head, txt, b1, b2]
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": "User profile", "maxWidth": "md", "description": None, "centered": None}), "children": [stk]},
        stk: {"type": "Stack", "props": {"direction": "vertical", "gap": "md", "align": None, "justify": None}, "children": kids},
        head: {"type": "Heading", "props": {"text": name, "level": gen.choice(["h2", "h3"])}, "children": []},
        txt: {"type": "Text", "props": {"text": gen.choice(["Senior engineer in San Francisco.", "Designer who builds tools."]), "variant": "muted"}, "children": []},
        b1: {"type": "Badge", "props": {"text": gen.choice(_SKILLS), "variant": "default"}, "children": []},
        b2: {"type": "Badge", "props": {"text": gen.choice(["Remote", "Full-time"]), "variant": "outline"}, "children": []},
    }
    if gen.random() < 0.7:
        els[prg] = {"type": "Progress", "props": {"value": gen.randint(40, 95), "max": 100, "label": "Profile completion"}, "children": []}
        kids.append(prg)
    return {"root": root, "elements": els}


def _notifications(gen):
    root = "card" + _k(gen)
    sw, al, al2 = "sw" + _k(gen), "al" + _k(gen), "a2" + _k(gen)
    state = {"alertsOn": True}
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": "Notifications", "maxWidth": "md", "description": None, "centered": None}), "children": [sw, al, al2]},
        sw: {"type": "Switch", "props": {"label": "Show alerts", "name": "alerts", "checked": {"$bindState": "/alertsOn"}}, "children": []},
        al: {"type": "Alert", "props": {"title": "Unsaved changes", "message": "You have unsaved edits.", "type": "warning"},
              "visible": {"$state": "/alertsOn", "eq": True}, "children": []},
        al2: {"type": "Alert", "props": {"title": "All clear", "message": "No notifications today.", "type": "success"},
               "visible": {"$state": "/alertsOn", "eq": False}, "children": []},
    }
    return {"root": root, "elements": els, "state": state}


def _pricing(gen):
    root, grid = "card" + _k(gen), "gd" + _k(gen)
    names = gen.sample(["Starter", "Pro", "Team", "Plus", "Solo"], 3)
    els = {root: {"type": "Card", "props": _nz(gen, {"title": "Pricing", "maxWidth": "lg", "description": None, "centered": None}), "children": [grid]},
           grid: {"type": "Grid", "props": {"columns": 3, "gap": "md"}, "children": []}}
    for i, pn in enumerate(names):
        ck, bn = "pc{0}{1}".format(i, _k(gen)), "pb{0}{1}".format(i, _k(gen))
        els[ck] = {"type": "Card", "props": _nz(gen, {"title": pn, "maxWidth": "full", "description": None, "centered": None}), "children": [bn]}
        els[bn] = {"type": "Button", "props": {"label": "Choose " + pn, "variant": "primary"}, "children": []}
        els[grid]["children"].append(ck)
        if i == 1 and gen.random() < 0.6:
            bd = "bd{0}{1}".format(i, _k(gen))
            els[ck]["children"] = [bd, bn]
            els[bd] = {"type": "Badge", "props": {"text": "Popular", "variant": "default"}, "children": []}
    return {"root": root, "elements": els}


def _search(gen):
    root = "card" + _k(gen)
    inp, lst, pag = "in" + _k(gen), "ls" + _k(gen), "pa" + _k(gen)
    item = "im" + _k(gen)
    n = gen.randint(2, 4)
    state = {"search": "", "page": 1,
             "results": [{"id": str(i + 1), "name": gen.choice(["Wireless mouse", "USB hub", "Desk lamp", "Monitor stand"]), "price": gen.randint(9, 99)} for i in range(n)]}
    els = {
        root: {"type": "Card", "props": _nz(gen, {"title": "Search products", "maxWidth": "md", "description": None, "centered": None}), "children": [inp, lst, pag]},
        inp: {"type": "Input", "props": {"label": "Search", "name": "search", "type": "text", "placeholder": "Type to search", "value": {"$bindState": "/search"}}, "children": []},
        lst: {"type": "Stack", "props": {"direction": "vertical", "gap": "sm", "align": None, "justify": None},
               "repeat": {"statePath": "/results", "key": "id"}, "children": [item]},
        item: {"type": "Card", "props": {"title": {"$item": "name"}, "maxWidth": "full", "description": None, "centered": None}, "children": []},
        pag: {"type": "Pagination", "props": {"totalPages": gen.randint(3, 9), "page": 1}, "children": [],
               "on": {"change": {"action": "load_data", "params": {}}}},
    }
    return {"root": root, "elements": els, "state": state}


def _orders_table(gen):
    root, tbl = "card" + _k(gen), "tb" + _k(gen)
    cols = ["Order", "Customer", "Total", "Status"]
    rows = [[str(1000 + i), gen.choice(_NAMES), "$" + str(gen.randint(10, 500)), _wpick(gen, _STATS)] for i in range(gen.randint(3, 5))]
    state = {"cols": cols, "rows": rows}
    els = {root: {"type": "Card", "props": _nz(gen, {"title": "Recent orders", "maxWidth": "lg", "description": None, "centered": None}), "children": [tbl]},
           tbl: {"type": "Table", "props": {"columns": {"$state": "/cols"}, "rows": {"$state": "/rows"}, "caption": None}, "children": []}}
    return {"root": root, "elements": els, "state": state}


ARCHETYPES = {
    "dashboard_metrics": _dashboard,
    "settings_form": _settings,
    "todo_list": _todo,
    "profile_card": _profile,
    "notifications_center": _notifications,
    "pricing_cards": _pricing,
    "search_results": _search,
    "orders_table": _orders_table,
}

# prompt composition (R2 diversity): template x subject pool. Register rolls
# (plain / "make me" / "can you build" / "I need ... for my app") come from
# the per-row generator, giving hundreds of natural phrasings instead of the
# fixed 36 lines the dataset audit flagged.
PROMPTS = {
    "dashboard_metrics": ["a dashboard with {s}", "a KPI screen for {s}"],
    "settings_form": ["settings form: {s}", "a settings screen for {s}"],
    "todo_list": ["a todo widget for {s}", "a task list for {s}"],
    "profile_card": ["a profile card for {s}", "a card page about {s}"],
    "notifications_center": ["an alerts panel for {s}", "conditional alerts with a toggle ({s})"],
    "pricing_cards": ["a pricing section for {s}", "three plan cards for {s}"],
    "search_results": ["a search page for {s}", "a product finder for {s}"],
    "orders_table": ["a table of orders for {s}", "sales as a table for {s}"],
}

_SUBJECTS = {
    "dashboard_metrics": ["kpi data", "our numbers", "revenue charts", "live stats",
                          "team metrics", "this week's numbers", "quarterly stats",
                          "the demo", "current sales"],
    "settings_form": ["account settings", "email preferences", "profile options"],
    "todo_list": ["daily tasks", "the shopping list", "sprint tasks"],
    "profile_card": ["the team lead", "contributors", "the speakers"],
    "notifications_center": ["unread alerts", "system notices"],
    "pricing_cards": ["the three tiers", "the plans", "billing tiers"],
    "search_results": ["catalog items", "the listings", "product matches"],
    "orders_table": ["recent orders", "this week's orders", "open orders"],
}

_PROJECTS = ["my app", "the review", "sprint demo", "the client", "our page", "the project"]


def _task_phrase(gen, arch):
    """One natural phrasing: template x subject, plus a register roll."""
    base = gen.choice(PROMPTS[arch])
    phrase = base.format(s=gen.choice(_SUBJECTS[arch]))
    roll = gen.random()
    if roll < 0.25:
        return "make me " + base.format(s=gen.choice(_SUBJECTS[arch])) + " (" + gen.choice(_PROJECTS) + ")"
    if roll < 0.45:
        return "can you build " + phrase + "?"
    if roll < 0.6:
        return "I need " + phrase + " for " + gen.choice(_PROJECTS)
    if roll < 0.7:
        return phrase[0].upper() + phrase[1:]
    return phrase

def build(seed, arch):
    gen = random.Random(seed * 7919 + 17)
    spec = ARCHETYPES[arch](gen)
    task = _task_phrase(gen, arch)
    problems = [p for p in validate_spec(spec) if p is not None]
    if problems:
        raise RuntimeError("archetype {0} invalid: {1}".format(arch, problems))
    return task, spec
