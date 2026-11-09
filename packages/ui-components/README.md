# @nano-slms/ui-components

Shared React UI components for the nano_SLMs GUI apps (`viz/`, `flow/`).
This is the ONE place the common surfaces are implemented — the apps
import from here and keep owning layout/typography in their own
stylesheets via the semantic class names (`.spinner`, `.modal`,
`.toast`, `.error-boundary`, ...).

## Components

| Component | Props | Notes |
|---|---|---|
| `Spinner` | `className?` | `role="status"`, `aria-label="Loading"`, pure CSS rotation |
| `ConfirmDialog` | `open, title, message, onConfirm, onDismiss, className?` | Renders null when closed; Escape + backdrop dismiss |
| `Toast` | `message, className?, role?` | Presentational; renders null for an empty message |
| `ErrorBoundary` | `children, className?` | Class boundary with `getDerivedStateFromError` + `componentDidCatch` |
| `Tabs` | `tabs: TabItem[], active, onChange, className?, ariaLabel?` | Full ARIA pattern (`tablist` / `tab` / `tabpanel`), roving tabindex |
| `SearchInput` | `id, value, onChange, onKeyDown?, placeholder?, label?, ariaLabel?, className?` | Visible `<label htmlFor>` or `ariaLabel` (default `"Search"`) |

## Theming

Every component takes a `className` prop and themes through CSS
variables. The names follow `viz/src/global.css` (the only existing
definitions in the repo) with fallbacks matching flow's light palette, so
the same markup themes correctly in both apps:

| Variable | Used for | Fallback |
|---|---|---|
| `--bg` | surfaces (search box, error fallback) | `#fff` |
| `--panel` | raised surfaces (dialog, inactive tab) | `#fff` / `#f6f6f6` |
| `--ink` | primary text | `#333` |
| `--dim` | secondary text / icons | `#888` / `#555` |
| `--line`, `--line2` | borders | `#ccc` / `#999` |
| `--out`, `--on-out` | toast background / text | `#c62828` / `#fff` |
| `--mono` | tab font stack | `monospace` |
| `--color-primary` | spinner accent | `currentColor` |
| `--backdrop` | modal scrim | `rgba(0, 0, 0, 0.15)` |
| `--tab-active`, `--tab-active-ink` | active tab colors | `#e9e7e2` / `#101014` |

## Running the tests

The suite runs standalone inside the package and is also pulled into both
app suites (`pnpm test` in `flow/` and `viz/` include
`../packages/ui-components/test/**`):

    pnpm test            # inside packages/ui-components
    pnpm test            # inside flow/ or viz/ (runs app + shared tests)
