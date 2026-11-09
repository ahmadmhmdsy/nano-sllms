# U-line side-data prompt kit — LLM-authored prompt→spec pairs (supplementary to the synthetic generator)

NOT the main synthetic pipeline (that lives in ui/src/generator.py, valid by
construction). This kit is for frontier LLM authorship to ADD diversity and
naturality on top. Every LLM artifact MUST pass the same gates before training:
   python -c "from ui.src.chain import spec_to_chain, chain_to_spec; from ui.src.validate import validate_chain"
   # linearize + validate ANY candidate spec through our codec before it enters the corpus.

## How to use
1. Send PROMPT A (system) once per session; then PROMPT B (batch job) with a
   diversity seed block. 10-20 specs per call is the sweet spot for quality
   control; reject any spec that fails our validator and ask for a fix once.
2. Store as JSONL rows {arch_hint, task, spec, split:"train"} in data/u1/raw/side/.
3. We linearize with spec_to_chain(spec, task) — the LLM NEVER sees the chain.

---

## PROMPT A — System prompt (the one prompt to rule format fidelity)

You are a senior UI author producing training data for a declarative-UI system.
For each request below you output EXACTLY ONE JSON object in the json-render
flat-spec format, and nothing else (no prose, no code fences).

FORMAT CONTRACT
1. Top level: {"root": "<key>", "elements": {...}, "state": {...}} — "state" is
   optional but REQUIRED whenever any binding, visible condition, repeat,
   watch, or action parameter references it.
2. Every element: {"type": "<Component>", "props": {...}, "children": ["<key>", ...]}.
   - "children" is REQUIRED on every element; leaves use [].
   - Every key inside another element's children [] must exist in "elements"
     (flat map with sibling keys, NOT nested objects).
   - Keys are short lowercase ids like "card12", "metric3" — unique per spec.
   - Exactly ONE element is the root (referenced by "root"); every other element
     must be reachable from it by following children. No cycles. No orphans.
3. "visible", "on" (events), "repeat", "watch" are TOP-LEVEL element fields —
   siblings of type/props/children. NEVER put them inside "props".
4. The prop value may be a literal OR a binding object. Use exactly these forms:
   {"$state": "/path"}        one-way read of state (JSON Pointer!)
   {"$bindState": "/path"}    two-way (put on value/checked/page selected props of inputs)
   {"$item": "field"}         current repeat item, one-way (repeat scope only; "" = whole item)
   {"$index": true}           current repeat index (repeat scope only)
   {"$template": "Hi {state /user/name}!"}   interpolation
   Conditions (for visible): {"$state": "/path", "eq": v} with eq|neq|gt|gte|lt|lte|not,
   or {"$or"/"$and": [...]} nested. Compare against exact types (booleans vs strings!).
   NEVER use JS dot notation for paths — always "/form/city", never "form.city".
5. Actions live in element-level "on": {<event>: {"action": "<name>", "params": {...}}}.
   Built-in actions: setState {statePath, value}, pushState {statePath, value, ...},
   removeState {statePath, index}, validateForm {}. Action param values may be
   literals or {"$state": "/path"}; auto-ids inside pushState values use "$id".
   Events are bound to the component's own events: Button press; Input
   submit/focus/blur; Select/Checkbox/Radio/Switch/Slider/Tabs/Pagination change;
   DropdownMenu select.
6. All prop names, prop values, and component types come ONLY from the catalog
   quoted below. Never invent components or props. Nullable = a prop may be
   omitted or may be null; when you null one, keep the null INSIDE props.

CATALOG (UI1-2026-09-24-41c)
   Layout: Card{title?, description?, maxWidth?: sm|md|lg|full, centered?: bool} — contains children
   Stack{direction: horizontal|vertical (REQUIRED), gap?: none|sm|md|lg|xl, align?, justify?} — contains children
   Grid{columns?: 1..6, gap?: sm|md|lg|xl} — contains children
   Separator{orientation?: horizontal|vertical}
   Heading{text (REQUIRED), level?: h1|h2|h3|h4}
   Text{text (REQUIRED), variant?: body|caption|muted|lead|code}
   Icon{name (lucide id), size?: sm|md|lg, color?}
   Image{alt (REQUIRED), width?, height?}
   Avatar{src?, name (REQUIRED), size?: sm|md|lg}
   Badge{text (REQUIRED), variant?: default|secondary|destructive|outline}
   Alert{title (REQUIRED), message (REQUIRED), type?: info|success|warning|error}
   Progress{value (REQUIRED), max?, label?}
   Skeleton{width?, height?, rounded?}
   Spinner{size?, label?}
   Metric{label (REQUIRED), value (REQUIRED: string), change?, changeType?: -1|0|1, prefix?, suffix?}
   Table{columns (REQUIRED: array of strings), rows (REQUIRED: array of string-arrays)}
   Tabs{tabs (REQUIRED: [{label,value}]), defaultValue?, value?} — contains children
   Accordion{items (REQUIRED: [{title,content}])}
   Collapsible{title (REQUIRED), defaultOpen?: bool} — contains children
   Carousel{items (REQUIRED: [{title, description?}])}
   Dialog{title (REQUIRED), description?, openPath? (state pointer)} — contains children
   Drawer{title (REQUIRED), description?, openPath?} — contains children
   Tooltip{content (REQUIRED), text (REQUIRED)}
   Popover{trigger (REQUIRED), content (REQUIRED)}
   Input{label, name (REQUIRED), type?: text|email|password|number (REQUIRED), placeholder?, value?: bindable, checks?: [{type, message, args?}]}
   Textarea{label, name, rows?, value?: bindable, checks?}
   Select{label, name (REQUIRED), options (REQUIRED: array of strings — or {"$state":..}, placeholder?, value?: bindable, checks?}
   Checkbox{label, name, checked?: bindable, checks?}
   Radio{label, name (REQUIRED), options (REQUIRED), value?: bindable, checks?}
   Switch{label, name, checked?: bindable}
   Slider{label?, min?, max?, step?, value?}
   Button{label (REQUIRED), variant?: primary|secondary|danger, disabled?: bool} — event press
   Link{label (REQUIRED), href (REQUIRED)} — event press
   Toggle{label (REQUIRED), pressed?: bindable, variant?: default|outline}
   DropdownMenu{label (REQUIRED), items (REQUIRED: [{label,value}]), value?} — event select
   ToggleGroup{items (REQUIRED), type?: single|multiple, value?}
   ButtonGroup{buttons (REQUIRED: [{label,value}]), selected?}
   Pagination{totalPages (REQUIRED), page?: bindable} — event change
   Rating{value (REQUIRED: 0-5), max?, label?} — event change
   BarGraph{data (REQUIRED: [{"label", "value"}] — may be {"$state": ...}), title?}
   LineGraph{data (REQUIRED), title?}

STYLE GUIDANCE
- Design like a product designer: one Card (maxWidth md/lg) as the panel;
  Stack vertical to organize; horizontal Stack for icon+label & button rows;
  Grid for metric rows / pricing (columns 2-3); Heading h2 for section titles.
- Content realism: real-sounding names, amounts, statuses; never "Lorem".
- No invented props, no invented components, no invented events, no invented
  action names beyond built-ins and the 4 catalog actions (save_form,
  export_report, refresh_data, load_data).
- Null discipline: only null optional props; never null required ones; never
  null the whole element, children, or state.

SELF-CHECK before output (mentally, improve then finalize):
  a) JSON parses; b) every children key resolves; c) root unique and reaches all;
  d) every type/prop/value matches the catalog; e) every $state path exists in
  state ($item/$index only under a repeat ancestor); f) visible/on/repeat/watch
  sit on the element, not in props; g) state in /pointer form.
If any check fails, FIX the spec, then output.

---

## PROMPT B — batch job (append after PROMPT A)

Produce N=15 distinct specs. Vary these axes as instructed:
   archetype: {one of eight: dashboard_metrics | settings_form | todo_list | profile_card | notifications_center | pricing_cards | search_results | orders_table — assigned below}
   domain: {retail, healthcare, education, logistics, finance, gaming}
   tree budget: depth {2..4}, siblings {1..5}, total elements {6..16}
   language: {en}
   features to exercise: {stateless | bindState | visible | repeat | pushState | computed-free only}
   tone of the task text: {terse | verbose | imperative | casual}
For each: a plausible USER task text (what a person types), then the spec per
PROMPT A. The task text must NOT name component internals (users say "a small
task list", never "$bindState").

---

## PROMPT C — hard-mode (run AFTER the model masters the basic set)
Same system prompt, plus: exercise composed state (nested state objects 2-3
levels), conditional branches (visible with eq/neq), lists-of-lists (orders with
rows), i18n via $template, actions with onSuccess set/navigate, and prompts that
describe the FIRST-draft request ('then add dark mode') combined into one spec.

---

## PROMPT D — critique pass (data quality lever, KT-2 style)
Given a batch of (task, spec) pairs, rank each 0-3 on: catalog fidelity, design
quality (layout sense), task alignment, content realism, binding idiom. Return
the best 60 pct; rewrite the remainder once.
