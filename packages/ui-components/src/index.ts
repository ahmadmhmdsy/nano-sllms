// Shared UI component library — the ONE place the common React
// surfaces are implemented. flow/ and viz/ import from here (see
// flow/src/App.tsx and viz/src/App.tsx); the apps keep owning layout and
// typography in their own stylesheets via the semantic class names
// (.spinner, .modal, .toast, .error-boundary, ...).
//
// Theming contract: every component takes a className prop and themes
// through CSS variables. The names follow viz/src/global.css (--bg,
// --panel, --ink, --dim, --line, --line2, --out, --mono) — the only
// existing definitions in the repo — with fallbacks matching flow's light
// palette, so the same markup themes correctly in both apps. Variables
// without an existing definition extend the same bare-semantic convention:
// --color-primary (spinner accent), --backdrop (modal scrim),
// --on-out (text on the --out accent), --tab-active / --tab-active-ink
// (active tab colors).

export { default as Spinner } from "./Spinner";
export { default as ConfirmDialog } from "./ConfirmDialog";
export type { ConfirmDialogProps } from "./ConfirmDialog";
export { default as Toast } from "./Toast";
export type { ToastProps } from "./Toast";
export { default as ErrorBoundary } from "./ErrorBoundary";
export { default as Tabs } from "./Tabs";
export type { TabItem, TabsProps } from "./Tabs";
export { default as SearchInput } from "./SearchInput";
export type { SearchInputProps } from "./SearchInput";
