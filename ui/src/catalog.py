"""U-line: frozen 41-component catalog (v1). Source: vercel-labs/json-render
shadcn set + chat example (research/ui_json_render_tiny_codegen_report.md §1/§5).
Frozen id UI1-2026-09-24-41c. Catalog changes must be pre-registered.
"""

from __future__ import annotations

from dataclasses import dataclass

CATALOG_FROZEN = "UI1-2026-09-24-41c"


@dataclass(frozen=True)
class Prop:
    kind: str          # str | num | bool | enum | as | ao | bx | ptr
    required: bool = False
    opts: tuple = ()


@dataclass(frozen=True)
class Comp:
    props: dict
    slot: bool = False
    events: tuple = ()


def _c(name, props, slot=False, events=()):
    return name, Comp(props, slot, events)


S = dict  # shorthand below

COMPONENTS: dict[str, Comp] = dict([
    _c("Card", {"title": "str", "description": "str", "maxWidth": ("enum", ("sm", "md", "lg", "full")), "centered": "bool"}, slot=True),
    _c("Stack", {"direction": ("enum", ("horizontal", "vertical"), True), "gap": ("enum", ("none", "sm", "md", "lg", "xl")), "align": ("enum", ("start", "center", "end", "stretch")), "justify": ("enum", ("start", "center", "end", "between", "around"))}, slot=True),
    _c("Grid", {"columns": "num", "gap": ("enum", ("sm", "md", "lg", "xl"))}, slot=True),
    _c("Separator", {"orientation": ("enum", ("horizontal", "vertical"))}),
    _c("Heading", {"text": ("str", True), "level": ("enum", ("h1", "h2", "h3", "h4"))}),
    _c("Text", {"text": ("str", True), "variant": ("enum", ("body", "caption", "muted", "lead", "code"))}),
    _c("Icon", {"name": ("str", True), "size": ("enum", ("sm", "md", "lg")), "color": "str"}),
    _c("Image", {"alt": ("str", True), "width": "num", "height": "num"}),
    _c("Avatar", {"src": "str", "name": ("str", True), "size": ("enum", ("sm", "md", "lg"))}),
    _c("Badge", {"text": ("str", True), "variant": ("enum", ("default", "secondary", "destructive", "outline"))}),
    _c("Alert", {"title": ("str", True), "message": ("str", True), "type": ("enum", ("info", "success", "warning", "error"))}),
    _c("Progress", {"value": ("num", True), "max": "num", "label": "str"}),
    _c("Skeleton", {"width": "str", "height": "str", "rounded": "bool"}),
    _c("Spinner", {"size": ("enum", ("sm", "md", "lg")), "label": "str"}),
    _c("Metric", {"label": ("str", True), "value": ("str", True), "change": "str", "changeType": ("enum", ("-1", "0", "1")), "prefix": "str", "suffix": "str"}),
    _c("Table", {"columns": ("as", True), "rows": ("ao", True), "caption": "str"}),
    _c("Tabs", {"tabs": ("ao", True), "defaultValue": "str", "value": "str"}, slot=True, events=("change",)),
    _c("Accordion", {"items": ("ao", True), "type": ("enum", ("single", "multiple"))}),
    _c("Collapsible", {"title": ("str", True), "defaultOpen": "bool"}, slot=True),
    _c("Carousel", {"items": ("ao", True)}),
    _c("Dialog", {"title": ("str", True), "description": "str", "openPath": "ptr"}, slot=True),
    _c("Drawer", {"title": ("str", True), "description": "str", "openPath": "ptr"}, slot=True),
    _c("Tooltip", {"content": ("str", True), "text": ("str", True)}),
    _c("Popover", {"trigger": ("str", True), "content": ("str", True)}),
    _c("Input", {"label": ("str", True), "name": ("str", True), "type": ("enum", ("text", "email", "password", "number")), "placeholder": "str", "value": "bx", "checks": "ao"}, events=("submit", "focus", "blur")),
    _c("Textarea", {"label": ("str", True), "name": ("str", True), "rows": "num", "value": "bx", "placeholder": "str", "checks": "ao"}, events=("submit", "focus", "blur")),
    _c("Select", {"label": ("str", True), "name": ("str", True), "options": ("as", True), "placeholder": "str", "value": "bx", "checks": "ao"}, events=("change",)),
    _c("Checkbox", {"label": ("str", True), "name": ("str", True), "checked": "bx", "checks": "ao"}, events=("change",)),
    _c("Radio", {"label": ("str", True), "name": ("str", True), "options": ("as", True), "value": "bx", "checks": "ao"}, events=("change",)),
    _c("Switch", {"label": ("str", True), "name": ("str", True), "checked": "bx"}, events=("change",)),
    _c("Slider", {"label": "str", "min": "num", "max": "num", "step": "num", "value": "num"}, events=("change",)),
    _c("Button", {"label": ("str", True), "variant": ("enum", ("primary", "secondary", "danger")), "disabled": "bool"}, events=("press",)),
    _c("Link", {"label": ("str", True), "href": ("str", True)}, events=("press",)),
    _c("Toggle", {"label": ("str", True), "pressed": "bx", "variant": ("enum", ("default", "outline"))}, events=("change",)),
    _c("DropdownMenu", {"label": ("str", True), "items": ("ao", True), "value": "str"}, events=("select",)),
    _c("ToggleGroup", {"items": ("ao", True), "type": ("enum", ("single", "multiple")), "value": "str"}, events=("change",)),
    _c("ButtonGroup", {"buttons": ("ao", True), "selected": "str"}, events=("change",)),
    _c("Pagination", {"totalPages": ("num", True), "page": "bx"}, events=("change",)),
    _c("Rating", {"value": ("num", True), "max": "num", "label": "str"}, events=("change",)),
    _c("BarGraph", {"data": ("ao", True), "title": "str"}),
    _c("LineGraph", {"data": ("ao", True), "title": "str"}),
])

ACTIONS = ("setState", "pushState", "removeState", "validateForm")
FREE_ACTIONS = ("save_form", "export_report", "refresh_data", "load_data")
BIND_KINDS = ("state", "bindState", "item", "bindItem", "index")
