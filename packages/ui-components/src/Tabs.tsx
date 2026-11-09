import type { CSSProperties, ReactNode } from "react";

/** One tab: a stable id, the button label, and the panel body rendered
 * while the tab is active. */
export interface TabItem<T extends string = string> {
  id: T;
  label: string;
  content: ReactNode;
}

/** Shared tab bar (new for viz/, whose hand-rolled nav had no ARIA roles).
 * Renders the tablist plus the active tab's panel; inactive panels are not
 * mounted, so per-tab component state resets on switch (same semantics as
 * viz's previous conditional rendering). The buttons carry the full ARIA
 * tab pattern: role="tab", aria-selected, aria-controls pointing at the
 * panel, and a roving tabindex (active tab tabIndex 0, others -1).
 * Button colors come from CSS variables (--panel / --dim / --line2 /
 * --tab-active / --tab-active-ink; viz defines the first three, the
 * fallbacks reproduce viz's current .tab / .on look) so the same markup
 * themes correctly in both apps. */
export interface TabsProps<T extends string = string> {
  tabs: TabItem<T>[];
  active: T;
  onChange: (id: T) => void;
  /** Extra class for the wrapper element. */
  className?: string;
  /** Accessible name for the tablist. */
  ariaLabel?: string;
}

const tabBaseStyle: CSSProperties = {
  border: "1px solid var(--line2, #ccc)",
  borderRadius: 10,
  cursor: "pointer",
  font: "14px var(--mono, monospace)",
  padding: "9px 22px",
};

const tabInactiveStyle: CSSProperties = {
  ...tabBaseStyle,
  background: "var(--panel, #f6f6f6)",
  color: "var(--dim, #555)",
};

const tabActiveStyle: CSSProperties = {
  ...tabBaseStyle,
  background: "var(--tab-active, #e9e7e2)",
  borderColor: "var(--tab-active, #e9e7e2)",
  color: "var(--tab-active-ink, #101014)",
  fontWeight: 700,
};

export default function Tabs<T extends string = string>({ tabs, active, onChange, className, ariaLabel }: TabsProps<T>) {
  if (tabs.length === 0) return null;
  const activeTab = tabs.find((t) => t.id === active) ?? tabs[0];
  return (
    <div className={"ui-tabs" + (className ? " " + className : "")}>
      <div
        role="tablist"
        aria-label={ariaLabel}
        style={{ display: "flex", gap: 8, justifyContent: "center", padding: "22px 0 0" }}
      >
        {tabs.map((t) => {
          const isActive = t.id === active;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={isActive}
              aria-controls={`tabpanel-${t.id}`}
              tabIndex={isActive ? 0 : -1}
              className={"ui-tab" + (isActive ? " ui-tab-active" : "")}
              style={isActive ? tabActiveStyle : tabInactiveStyle}
              onClick={() => onChange(t.id)}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        id={`tabpanel-${activeTab.id}`}
        aria-labelledby={`tab-${activeTab.id}`}
        className="ui-tabs-panel"
      >
        {activeTab.content}
      </div>
    </div>
  );
}
