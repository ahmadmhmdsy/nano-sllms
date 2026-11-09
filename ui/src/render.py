"""spec -> real, standalone HTML ("json-render" step of the U-line).

The model emits a FLAT spec: {elements, root, state?}. Elements are keyed by id and
each carries {type, props, children, on?}. This module walks that graph from the root
and emits a self-contained HTML document - no CDN, no build step, no framework - so a
generated spec can be opened straight in a browser.

SECURITY: every string here originates from a language model. All text is HTML-escaped,
no value is ever interpolated into a script or a style attribute, and the only JS in the
output is a fixed, literal block that only ever writes via textContent. A component type
outside the frozen catalog renders as a visible placeholder instead of being guessed at.

Usage:
    from ui.src.render import render_document
    html = render_document(spec, title="My page")
"""
import html
import json
import re

from ui.src.catalog import CATALOG_FROZEN, COMPONENTS

_UNSAFE = re.compile(r"[<>]|javascript:", re.I)


def esc(v):
    """HTML-escape any value, including hostile model output."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    s = str(v)
    return html.escape(s, quote=True)


def resolve(v, state, scope):
    """Resolve a prop value that may be a binding pointer.

    {.."$bindState": "/cart/items"} -> state value
    {.."$item": "name"}            -> current repeat-scope value
    """
    if isinstance(v, dict):
        if "$bindState" in v:
            return dig(state, str(v["$bindState"]))
        if "$state" in v:
            return dig(state, str(v["$state"]))
        if "$item" in v:
            return scope.get(str(v["$item"]), "")
        if "$const" in v:
            return v["$const"]
        return json.dumps(v, ensure_ascii=False)
    return v


def dig(obj, pointer):
    """RFC-6901-ish lookup that never raises."""
    node = obj
    for tok in str(pointer).lstrip("/").split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and tok in node:
            node = node[tok]
        elif isinstance(node, list):
            try:
                node = node[int(tok)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def _num(v, default=0.0):
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return default


def _series(data, state, scope):
    """Normalise a chart payload to [(label, float_value)]."""
    out = []
    for row in data or []:
        if isinstance(row, dict):
            lbl = resolve(row.get("label", row.get("name", "")), state, scope)
            val = resolve(row.get("value", row.get("y", 0)), state, scope)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            lbl, val = row[0], row[1]
        else:
            continue
        out.append((str(lbl), _num(val)))
    return out


def _opts(props, key, state, scope):
    raw = resolve(props.get(key), state, scope) or []
    out = []
    for o in raw:
        if isinstance(o, dict):
            lbl = resolve(o.get("label", o.get("text", o.get("name", ""))), state, scope)
            val = resolve(o.get("value", lbl), state, scope)
            out.append((str(lbl), str(val)))
        else:
            out.append((str(o), str(o)))
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0e1117;color:#e6edf3;
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px}
.card{background:#161b22;border:1px solid #262d36;border-radius:12px;padding:18px 20px}
.card>.hd{display:flex;flex-direction:column;gap:3px;margin-bottom:14px}
.card .t{font-weight:650;font-size:16px}
.card .d{color:#8b949e;font-size:13px}
.stack{display:flex;gap:14px}
.stack.row{flex-direction:row}
.stack.col{flex-direction:column}
.grid{display:grid;gap:14px}
h1,h2,h3,h4{margin:0 0 8px;line-height:1.25}
h1{font-size:24px}h2{font-size:20px}h3{font-size:17px}h4{font-size:15px}
.text{color:#c9d1d9}
.text.muted{color:#8b949e;font-size:13px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;
 border:1px solid #30363d;background:#21262d}
.badge.success{color:#3fb950;border-color:#23863633;background:#3fb9501f}
.badge.warning{color:#d29922;background:#d299221f}
.badge.danger{color:#f85149;background:#f851491f}
.badge.info{color:#58a6ff;background:#58a6ff1f}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:8px;
 border:1px solid #30363d;background:#21262d;color:#e6edf3;cursor:pointer;font-size:14px}
.btn.primary{background:#238636;border-color:#2ea043;color:#fff}
.btn.danger{background:#8b2c2c;border-color:#b04a4a;color:#fff}
.btn:disabled{opacity:.45;cursor:not-allowed}
.field{display:flex;flex-direction:column;gap:5px;margin-bottom:11px}
.field label{font-size:13px;color:#8b949e}
input[type=text],input[type=email],input[type=number],input[type=password],textarea,select{
 width:100%;padding:7px 10px;border-radius:7px;border:1px solid #30363d;
 background:#0d1117;color:#e6edf3;font:inherit}
input:focus,textarea:focus,select:focus{outline:2px solid #1f6feb33;border-color:#1f6feb}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase;
 letter-spacing:.04em;padding:7px 10px;border-bottom:1px solid #262d36}
td{padding:8px 10px;border-bottom:1px solid #21262d}
.metric{background:#0d1117;border:1px solid #262d36;border-radius:10px;padding:13px 15px}
.metric .lbl{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.metric .val{font-size:27px;font-weight:680;margin-top:3px;letter-spacing:-.02em}
.metric .chg{font-size:12px;margin-top:2px}
.up{color:#3fb950}.down{color:#f85149}
.alert{display:flex;flex-direction:column;gap:3px;padding:12px 15px;border-radius:9px;
 border:1px solid #30363d;background:#161b22}
.alert.success{border-left:3px solid #3fb950}
.alert.warning{border-left:3px solid #d29922}
.alert.error,.alert.danger{border-left:3px solid #f85149}
.alert.info{border-left:3px solid #58a6ff}
.alert .ttl{font-weight:650}
.progress{height:8px;background:#21262d;border-radius:999px;overflow:hidden}
.progress>i{display:block;height:100%;background:#1f6feb;border-radius:999px}
svg.chart{width:100%;height:auto;display:block}
.acc summary{cursor:pointer;padding:9px 0;font-weight:600}
.acc{border-bottom:1px solid #21262d}
.tabs{display:flex;gap:4px;border-bottom:1px solid #262d36;margin-bottom:11px}
.tabs button{background:none;border:0;color:#8b949e;padding:8px 13px;cursor:pointer;
 font:inherit;border-bottom:2px solid transparent}
.tabs button[aria-selected=true]{color:#e6edf3;border-bottom-color:#1f6feb}
.avatar{width:34px;height:34px;border-radius:50%;background:#1f6feb33;color:#79c0ff;
 display:flex;align-items:center;justify-content:center;font-weight:650}
.skel{background:linear-gradient(90deg,#161b22,#21262d,#161b22);border-radius:7px;
 animation:sh 1.4s infinite}
@keyframes sh{0%{opacity:.55}50%{opacity:1}100%{opacity:.55}}
.pill{font-size:12px;color:#8b949e;padding:2px 8px;border:1px solid #30363d;border-radius:999px}
.statebox{margin-top:26px;background:#0d1117;border:1px solid #262d36;border-radius:10px;padding:14px}
.statebox h4{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.statebox pre{margin:0;font-size:12px;color:#7ee787;white-space:pre-wrap;word-break:break-word}
.unknown{border:1px dashed #d29922;border-radius:8px;padding:9px 12px;color:#d29922;font-size:13px}
"""

JS = """
// Literal, fixed script. Model output never reaches eval/innerHTML - only textContent.
document.addEventListener('click', function(e){
  var b = e.target.closest('[data-tabgroup]');
  if(!b) return;
  var gid = b.getAttribute('data-tabgroup');
  document.querySelectorAll('[data-tabgroup="'+gid+'"] > button').forEach(function(x){
    x.setAttribute('aria-selected', String(x === b));
  });
  document.querySelectorAll('[data-panelgroup="'+gid+'"] > [data-panel]').forEach(function(p){
    p.hidden = p.getAttribute('data-tab') !== b.getAttribute('data-tab');
  });
});
"""


def _bar_chart(series, w=680, h=190):
    if not series:
        return '<p class="text muted">no data</p>'
    mx = max(abs(v) for _, v in series) or 1.0
    n = len(series)
    gap, pad = 8, 26
    bw = max(4, (w - pad * 2 - gap * (n - 1)) / n)
    bars = []
    for i, (lbl, v) in enumerate(series):
        bh = max(2, abs(v) / mx * (h - 52))
        x = pad + i * (bw + gap)
        y = h - 26 - bh
        col = "#3fb950" if v >= 0 else "#f85149"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"><title>{esc(lbl)}: {esc(v)}</title></rect>'
        )
        bars.append(
            f'<text x="{x + bw / 2:.1f}" y="{h - 9}" fill="#8b949e" font-size="10" text-anchor="middle">{esc(lbl)}</text>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" role="img">'
        + "".join(bars)
        + "</svg>"
    )


def _line_chart(series, w=680, h=190):
    if not series:
        return '<p class="text muted">no data</p>'
    mx = max(abs(v) for _, v in series) or 1.0
    pad = 28
    n = max(len(series) - 1, 1)
    step = (w - pad * 2) / n
    pts = [
        (pad + i * step, h - 26 - (v / mx) * (h - 52))
        for i, (_, v) in enumerate(series)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = (
        f"{pad},{h - 26} " + line + f" {pts[-1][0]:.1f},{h - 26}"
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#58a6ff"><title>{esc(l)}: {esc(v)}</title></circle>'
        for (x, y), (l, v) in zip(pts, series)
    )
    labels = "".join(
        f'<text x="{x:.1f}" y="{h - 9}" fill="#8b949e" font-size="10" text-anchor="middle">{esc(l)}</text>'
        for (x, _), (l, _) in zip(pts, series)
    )
    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" role="img">'
        f'<polygon points="{area}" fill="#58a6ff1f"/>'
        f'<polyline points="{line}" fill="none" stroke="#58a6ff" stroke-width="2"/>'
        f"{dots}{labels}</svg>"
    )


def _children(el, els, state, scope, seen):
    kids = el.get("children") or []
    if not isinstance(kids, list):
        return ""
    out = []
    for kid in kids:
        if isinstance(kid, str) and kid in els:
            out.append(_node(kid, els, state, scope, seen))
    return "".join(out)


def _node(eid, els, state, scope, seen):
    el = els.get(eid)
    if not isinstance(el, dict):
        return ""
    if eid in seen:
        return f'<div class="unknown">cycle at {esc(eid)}</div>'
    seen = seen | {eid}
    t = el.get("type")
    p = el.get("props") or {}
    p = {k: resolve(v, state, scope) for k, v in p.items()}
    kids = _children(el, els, state, scope, seen)
    on = el.get("on") or {}
    ev = " ".join(
        f"data-on-{esc(k)}={esc(json.dumps(v, ensure_ascii=False))}" for k, v in on.items()
    ) if isinstance(on, dict) and on else ""
    comp = COMPONENTS.get(t)
    if comp is None:
        return (
            f'<div class="unknown">unknown component type '
            f'<b>{esc(t)}</b> (not in {esc(CATALOG_FROZEN)})</div>'
        )
    fn = _RENDER.get(t, _generic)
    return fn(p, kids, el, state, scope, ev)


def _generic(p, kids, el, state, scope, ev):
    inner = kids or "".join(
        f'<div class="text">{esc(v)}</div>' for v in p.values() if not isinstance(v, (dict, list))
    )
    return f'<div class="card" {ev}>{inner}</div>'


def _card(p, kids, el, state, scope, ev):
    t = p.get("title")
    d = p.get("description")
    hd = ""
    if t or d:
        hd = (
            '<div class="hd">'
            + (f'<div class="t">{esc(t)}</div>' if t else "")
            + (f'<div class="d">{esc(d)}</div>' if d else "")
            + "</div>"
        )
    return f'<div class="card" {ev}>{hd}{kids}</div>'


def _stack(p, kids, el, state, scope, ev):
    d = str(p.get("direction") or "col").lower()
    cls = "stack " + ("row" if d in ("row", "horizontal", "h") else "col")
    st = []
    if p.get("gap"):
        st.append(f"gap:{esc(p['gap'])}")
    if p.get("align"):
        st.append(f"align-items:{esc(p['align'])}")
    if p.get("justify"):
        st.append(f"justify-content:{esc(p['justify'])}")
    sty = f' style="{";".join(st)}"' if st else ""
    if not kids:
        kids = "".join(f'<div class="text">{esc(v)}</div>' for v in p.values())
    return f'<div class="{cls}"{sty} {ev}>{kids}</div>'


def _grid(p, kids, el, state, scope, ev):
    cols = p.get("columns") or 2
    try:
        cols = int(cols)
    except (TypeError, ValueError):
        cols = 2
    cols = max(1, min(cols, 6))
    st = f' style="grid-template-columns:repeat({cols},minmax(0,1fr))"'
    if p.get("gap"):
        st = f' style="grid-template-columns:repeat({cols},minmax(0,1fr));gap:{esc(p["gap"])}"'
    if not kids:
        kids = "".join(
            f'<div class="card"><div class="text">{esc(v)}</div></div>'
            for v in p.values()
            if not isinstance(v, (dict, list))
        )
    return f'<div class="grid"{st} {ev}>{kids}</div>'


def _metric(p, kids, el, state, scope, ev):
    chg = p.get("change")
    ctype = str(p.get("changeType") or "")
    ch = ""
    if chg not in (None, ""):
        up = str(ctype) in ("1", "up", "increase", "+")
        arrow = "&#9650;" if up else "&#9660;"
        ch = f'<div class="chg {"up" if up else "down"}">{arrow} {esc(chg)}</div>'
    val = esc(p.get("prefix", "")) + esc(p.get("value", "")) + esc(p.get("suffix", ""))
    return (
        f'<div class="metric" {ev}><div class="lbl">{esc(p.get("label"))}</div>'
        f'<div class="val">{val}</div>{ch}</div>'
    )


def _table(p, kids, el, state, scope, ev):
    cols = p.get("columns") or []
    if isinstance(cols, str):
        cols = [cols]
    rows = p.get("rows") or []
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for r in rows:
        if isinstance(r, dict):
            cells = "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols)
        elif isinstance(r, (list, tuple)):
            cells = "".join(f"<td>{esc(x)}</td>" for x in r)
        else:
            cells = f"<td>{esc(r)}</td>"
        body.append(f"<tr>{cells}</tr>")
    cap = p.get("caption")
    cap_html = f"<caption style='caption-side:bottom;color:#8b949e;font-size:12px;padding-top:8px;text-align:left'>{esc(cap)}</caption>" if cap else ""
    return (
        f'<div style="overflow:auto" {ev}><table>{cap_html}'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _alert(p, kids, el, state, scope, ev):
    t = str(p.get("type") or "info").lower()
    t = "error" if t == "danger" else t
    return (
        f'<div class="alert {esc(t)}" {ev}>'
        f'<div class="ttl">{esc(p.get("title"))}</div>'
        f'<div class="text">{esc(p.get("message"))}</div></div>'
    )


def _button(p, kids, el, state, scope, ev):
    v = str(p.get("variant") or "default").lower()
    cls = "btn" + ("" if v in ("default", "") else " " + esc(v))
    dis = " disabled" if p.get("disabled") else ""
    return f'<button class="{cls}"{dis} {ev}>{esc(p.get("label"))}</button>'


def _button_group(p, kids, el, state, scope, ev):
    sel = str(p.get("selected") or "")
    out = []
    for b in p.get("buttons") or []:
        if isinstance(b, dict):
            lbl = b.get("label")
            val = b.get("value", lbl)
        else:
            lbl = val = b
        on = ' aria-pressed="true"' if str(val) == sel else ' aria-pressed="false"'
        out.append(f'<button class="btn"{on}>{esc(lbl)}</button>')
    return f'<div style="display:flex;gap:8px" {ev}>{"".join(out) or kids}</div>'


def _field(label, control, name):
    lb = f"<label>{esc(label)}</label>" if label else ""
    return f'<div class="field">{lb}{control}</div>'


def _input(p, kids, el, state, scope, ev):
    t = str(p.get("type") or "text")
    val = esc(p.get("value", ""))
    ph = f' placeholder="{esc(p["placeholder"])}"' if p.get("placeholder") else ""
    nm = f' name="{esc(p["name"])}"' if p.get("name") else ""
    return _field(p.get("label"), f'<input type="{esc(t)}" value="{val}"{ph}{nm} {ev}>', p.get("name"))


def _textarea(p, kids, el, state, scope, ev):
    rows = p.get("rows") or 3
    val = esc(p.get("value", ""))
    ph = f' placeholder="{esc(p["placeholder"])}"' if p.get("placeholder") else ""
    return _field(p.get("label"), f"<textarea rows=\"{esc(rows)}\"{ph}>{val}</textarea>", p.get("name"))


def _select(p, kids, el, state, scope, ev):
    opts = "".join(
        f'<option value="{esc(v)}"{" selected" if str(v) == str(p.get("value")) else ""}>{esc(l)}</option>'
        for l, v in _opts(p, "options", state, scope)
    )
    ph = f'<option value="">{esc(p["placeholder"])}</option>' if p.get("placeholder") else ""
    return _field(p.get("label"), f"<select {ev}>{ph}{opts}</select>", p.get("name"))


def _radio(p, kids, el, state, scope, ev):
    cur = str(p.get("value") or "")
    nm = esc(p.get("name") or "radio")
    out = "".join(
        f'<label style="display:flex;gap:7px;align-items:center;padding:3px 0">'
        f'<input type="radio" name="{nm}" value="{esc(v)}"'
        f'{" checked" if str(v) == cur else ""}><span>{esc(l)}</span></label>'
        for l, v in _opts(p, "options", state, scope)
    )
    return _field(p.get("label"), f"<div>{out}</div>", nm)


def _checkbox(p, kids, el, state, scope, ev):
    checks = p.get("checks") or []
    if not checks:
        return _field(p.get("label"), '<input type="checkbox">', p.get("name"))
    nm = esc(p.get("name") or "check")
    out = "".join(
        f'<label style="display:flex;gap:7px;align-items:center;padding:3px 0">'
        f'<input type="checkbox" name="{nm}" value="{esc(c)}"><span>{esc(c)}</span></label>'
        for c in checks
    )
    return _field(p.get("label"), f"<div>{out}</div>", nm)


def _switch(p, kids, el, state, scope, ev):
    chk = " checked" if p.get("checked") else ""
    return (
        f'<div class="field" style="flex-direction:row;align-items:center;gap:9px">'
        f'<input type="checkbox" role="switch"{chk} style="width:auto" {ev}>'
        f'<label style="margin:0">{esc(p.get("label"))}</label></div>'
    )


def _slider(p, kids, el, state, scope, ev):
    mn, mx = p.get("min", 0), p.get("max", 100)
    v = p.get("value", mn)
    st = p.get("step")
    sa = f' step="{esc(st)}"' if st else ""
    return _field(
        f'{p.get("label")} ({esc(mn)}-{esc(mx)})',
        f'<input type="range" min="{esc(mn)}" max="{esc(mx)}" value="{esc(v)}"{sa} style="width:100%" {ev}>',
        p.get("name"),
    )


def _rating(p, kids, el, state, scope, ev):
    mx = int(_num(p.get("max", 5), 5) or 5)
    v = int(_num(p.get("value", 0), 0) or 0)
    stars = "".join(
        ("&#9733;" if i <= v else "&#9734;") for i in range(1, max(mx, 1) + 1)
    )
    return f'<div class="field"><label>{esc(p.get("label"))}</label><div style="color:#d29922;font-size:19px">{stars}</div></div>'


def _heading(p, kids, el, state, scope, ev):
    lvl = p.get("level") or 2
    try:
        lvl = max(1, min(int(lvl), 6))
    except (TypeError, ValueError):
        lvl = 2
    return f"<h{lvl} {ev}>{esc(p.get('text'))}</h{lvl}>"


def _text(p, kids, el, state, scope, ev):
    v = str(p.get("variant") or "")
    cls = "text muted" if v in ("muted", "secondary", "subtle") else "text"
    return f'<div class="{cls}" {ev}>{esc(p.get("text"))}</div>'


def _badge(p, kids, el, state, scope, ev):
    v = str(p.get("variant") or "default")
    cls = "badge" if v in ("default", "") else "badge " + esc(v)
    return f'<span class="{cls}" {ev}>{esc(p.get("text"))}</span>'


def _link(p, kids, el, state, scope, ev):
    href = str(p.get("href") or "#")
    if _UNSAFE.search(href) and not href.startswith("#"):
        href = "#"
    return f'<a class="btn" href="{esc(href)}" rel="noopener noreferrer" {ev}>{esc(p.get("label"))}</a>'


def _bargraph(p, kids, el, state, scope, ev):
    t = p.get("title")
    return (
        (f'<div class="t" style="margin-bottom:6px">{esc(t)}</div>' if t else "")
        + _bar_chart(_series(p.get("data"), state, scope))
    )


def _linegraph(p, kids, el, state, scope, ev):
    t = p.get("title")
    return (
        (f'<div class="t" style="margin-bottom:6px">{esc(t)}</div>' if t else "")
        + _line_chart(_series(p.get("data"), state, scope))
    )


def _progress(p, kids, el, state, scope, ev):
    v, mx = _num(p.get("value", 0)), _num(p.get("max", 100), 100) or 100
    pct = max(0, min(100, v / mx * 100))
    lbl = p.get("label")
    return (
        (f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-bottom:4px"><span>{esc(lbl)}</span><span>{pct:.0f}%</span></div>' if lbl else "")
        + f'<div class="progress" {ev}><i style="width:{pct:.1f}%"></i></div>'
    )


def _tabs(p, kids, el, state, scope, ev):
    gid = "tg" + str(abs(hash(str(p))) % 100000)
    cur = str(p.get("value") or p.get("defaultValue") or "")
    tabs = p.get("tabs") or []
    if isinstance(tabs, str):
        tabs = [{"label": tabs, "value": tabs}]
    btns = []
    first = True
    for t in tabs:
        if isinstance(t, dict):
            lbl, val = t.get("label"), t.get("value", t.get("label"))
        else:
            lbl = val = t
        sel = "true" if (str(val) == cur or (first and not cur)) else "false"
        first = False
        btns.append(
            f'<button role="tab" aria-selected="{sel}" data-tab="{esc(val)}" data-tabgroup="{gid}">{esc(lbl)}</button>'
        )
    return (
        f'<div><div class="tabs" data-tabgroup="{gid}" {ev}>{"".join(btns)}</div>'
        f'<div data-panelgroup="{gid}">{kids}</div></div>'
    )


def _tab_panel(p, kids, el, state, scope, ev):
    return f'<div data-panel hidden>{kids}</div>'


def _accordion(p, kids, el, state, scope, ev):
    out = []
    for it in p.get("items") or []:
        if isinstance(it, dict):
            t = it.get("title")
            c = it.get("content", it.get("description", ""))
        else:
            t, c = str(it), ""
        out.append(f"<details class='acc'><summary>{esc(t)}</summary><div class='text'>{esc(c)}</div></details>")
    return f'<div {ev}>{"".join(out) or kids}</div>'


def _collapsible(p, kids, el, state, scope, ev):
    op = " open" if p.get("defaultOpen") else ""
    return f'<details class="acc"{op} {ev}><summary>{esc(p.get("title"))}</summary>{kids}</details>'


def _carousel(p, kids, el, state, scope, ev):
    cards = []
    for it in p.get("items") or []:
        if isinstance(it, dict):
            t = it.get("title")
            d = it.get("description", it.get("content", ""))
        else:
            t, d = str(it), ""
        cards.append(f'<div class="card"><div class="t">{esc(t)}</div><div class="text">{esc(d)}</div></div>')
    return f'<div style="display:flex;gap:12px;overflow:auto" {ev}>{"".join(cards) or kids}</div>'


def _dialog(p, kids, el, state, scope, ev):
    return (
        f'<div class="card" {ev}><div class="t">{esc(p.get("title"))}</div>'
        f'<div class="d">{esc(p.get("description"))}</div>{kids}</div>'
    )


def _drawer(p, kids, el, state, scope, ev):
    return f'<div class="card" style="border-left:3px solid #1f6feb" {ev}><div class="t">{esc(p.get("title"))}</div>{kids}</div>'


def _dropdown(p, kids, el, state, scope, ev):
    opts = _opts(p, "items", state, scope)
    cur = str(p.get("value") or "")
    sel = next((l for l, v in opts if v == cur), opts[0][0] if opts else "")
    return _field(p.get("label"), f'<select {ev}>' + "".join(
        f'<option{" selected" if l == sel else ""}>{esc(l)}</option>' for l, _ in opts
    ) + "</select>", None)


def _toggle_group(p, kids, el, state, scope, ev):
    cur = str(p.get("value") or "")
    out = []
    for it in p.get("items") or []:
        if isinstance(it, dict):
            lbl, val = it.get("label"), it.get("value", it.get("label"))
        else:
            lbl = val = it
        on = " aria-pressed='true'" if str(val) == cur else " aria-pressed='false'"
        out.append(f"<button class='btn'{on}>{esc(lbl)}</button>")
    return f"<div style='display:flex;gap:8px' {ev}>{''.join(out) or kids}</div>"


def _toggle(p, kids, el, state, scope, ev):
    on = " aria-pressed='true'" if p.get("pressed") else " aria-pressed='false'"
    return f"<button class='btn' style='border-radius:999px'{on} {ev}>{esc(p.get('label'))}</button>"


def _avatar(p, kids, el, state, scope, ev):
    nm = str(p.get("name") or "")
    ini = "".join(w[0] for w in nm.split()[:2]).upper() or "?"
    sz = p.get("size") or 34
    return (
        f'<div class="avatar" style="width:{esc(sz)}px;height:{esc(sz)}px;font-size:{esc(int(_num(sz,34)*0.4))}px" title="{esc(nm)}" {ev}>{esc(ini)}</div>'
    )


def _icon(p, kids, el, state, scope, ev):
    return f'<span class="pill" style="color:#79c0ff" {ev}>&#9635; {esc(p.get("name"))}</span>'


def _image(p, kids, el, state, scope, ev):
    return (
        f'<div class="skel" style="height:{esc(p.get("height") or 140)}px;width:{esc(p.get("width") or "100%")}" '
        f'title="{esc(p.get("alt") or "image")}" {ev}></div>'
    )


def _spinner(p, kids, el, state, scope, ev):
    return f'<div class="text muted" {ev}>&#9696; {esc(p.get("label") or "loading")}</div>'


def _skeleton(p, kids, el, state, scope, ev):
    return (
        f'<div class="skel" style="height:{esc(p.get("height") or 16)}px;'
        f'width:{esc(p.get("width") or "100%")};border-radius:{esc(p.get("rounded") or 7)}px" {ev}></div>'
    )


def _separator(p, kids, el, state, scope, ev):
    o = str(p.get("orientation") or "horizontal")
    if o == "vertical":
        return f'<div style="width:1px;align-self:stretch;background:#262d36" {ev}></div>'
    return f'<hr style="border:0;border-top:1px solid #262d36;margin:4px 0" {ev}>'


def _pagination(p, kids, el, state, scope, ev):
    cur = int(_num(p.get("page", 1), 1) or 1)
    tot = int(_num(p.get("totalPages", 1), 1) or 1)
    btns = "".join(
        f'<button class="btn"{" disabled" if i == cur else ""}>{i}</button>'
        for i in range(1, min(tot, 9) + 1)
    )
    return f'<div style="display:flex;gap:6px" {ev}>{btns}</div>'


def _tooltip(p, kids, el, state, scope, ev):
    return f'<span class="pill" title="{esc(p.get("content"))}" {ev}>{esc(p.get("text"))}</span>'


def _popover(p, kids, el, state, scope, ev):
    return f'<span class="pill" {ev}>{esc(p.get("trigger"))} &#8599;</span>'


_RENDER = {
    "Card": _card, "Stack": _stack, "Grid": _grid, "Metric": _metric,
    "Table": _table, "Alert": _alert, "Button": _button, "ButtonGroup": _button_group,
    "Input": _input, "Textarea": _textarea, "Select": _select, "Radio": _radio,
    "Checkbox": _checkbox, "Switch": _switch, "Slider": _slider, "Rating": _rating,
    "Heading": _heading, "Text": _text, "Badge": _badge, "Link": _link,
    "BarGraph": _bargraph, "LineGraph": _linegraph, "Progress": _progress,
    "Tabs": _tabs, "Accordion": _accordion, "Collapsible": _collapsible,
    "Carousel": _carousel, "Dialog": _dialog, "Drawer": _drawer,
    "DropdownMenu": _dropdown, "ToggleGroup": _toggle_group, "Toggle": _toggle,
    "Avatar": _avatar, "Icon": _icon, "Image": _image, "Spinner": _spinner,
    "Skeleton": _skeleton, "Separator": _separator, "Pagination": _pagination,
    "Tooltip": _tooltip, "Popover": _popover,
}


def spec_to_html(spec, show_state=True):
    """Render a flat U-line spec to an HTML fragment."""
    if not isinstance(spec, dict):
        return '<div class="unknown">not a spec object</div>'
    els = spec.get("elements") or {}
    root = spec.get("root")
    state = spec.get("state") or {}
    if not isinstance(els, dict) or not els:
        return '<div class="unknown">empty spec</div>'
    if root not in els:
        known = list(els)[:4]
        return (
            f'<div class="unknown">root <b>{esc(root)}</b> is not defined '
            f'(known ids: {esc(", ".join(known))})</div>'
        )
    body = _node(root, els, state, {}, frozenset())
    if show_state and state:
        body += (
            '<div class="statebox"><h4>state (from the generated spec)</h4>'
            f"<pre>{esc(json.dumps(state, indent=1, ensure_ascii=False))}</pre></div>"
        )
    return body


def render_document(spec, title="Generated UI", show_state=True, meta=""):
    """Full standalone HTML document for a spec."""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class='wrap'>"
        + (f"<p class='pill' style='display:inline-block;margin-bottom:14px'>{esc(meta)}</p>" if meta else "")
        + spec_to_html(spec, show_state=show_state)
        + f"<script>{JS}</script></div></body></html>"
    )
